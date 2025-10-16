//! Loudness-related DSP utilities for combined LUFS and peak metering.

use super::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::util::audio::DEFAULT_SAMPLE_RATE;
use std::collections::VecDeque;

const MIN_MEAN_SQUARE: f64 = 1e-12;
const LOG10_FACTOR: f64 = 10.0;
const DB_FACTOR: f32 = 20.0;
const LUFS_OFFSET: f64 = -0.691;
const NOMINAL_SAMPLE_RATE: f32 = DEFAULT_SAMPLE_RATE;
const SAMPLE_RATE_TOLERANCE: f32 = 0.1;
const DEFAULT_SHORT_TERM_WINDOW: f32 = 3.0;
const DEFAULT_RMS_FAST_WINDOW: f32 = 0.3;
const DEFAULT_FLOOR_LUFS: f32 = -99.9;

// ITU-R BS.1770-5: https://www.itu.int/rec/R-REC-BS.1770
const PRE_B_COEFFS_48K: [f64; 3] = [
    1.535_124_859_586_97,
    -2.691_696_189_406_38,
    1.198_392_810_852_85,
];
const PRE_A_COEFFS_48K: [f64; 3] = [1.0, -1.690_659_293_182_41, 0.732_480_774_215_85];
const HP_B_COEFFS_48K: [f64; 3] = [1.0, -2.0, 1.0];
const HP_A_COEFFS_48K: [f64; 3] = [1.0, -1.990_047_454_833_98, 0.990_072_250_366_21];

#[inline]
fn mean_square_to_db(mean_square: f64, floor: f32) -> f32 {
    if mean_square <= MIN_MEAN_SQUARE {
        floor
    } else {
        (LOG10_FACTOR * mean_square.log10() + LUFS_OFFSET).max(floor as f64) as f32
    }
}

#[inline]
fn peak_to_db(peak: f32, floor: f32) -> f32 {
    if peak <= f32::EPSILON {
        floor
    } else {
        (DB_FACTOR * peak.log10()).max(floor)
    }
}

#[inline]
fn window_length(sample_rate: f32, window_secs: f32) -> usize {
    (sample_rate * window_secs).max(1.0) as usize
}

#[derive(Debug, Clone)]
struct RollingMeanSquare {
    samples: VecDeque<f64>,
    capacity: usize,
    sum: f64,
}

impl RollingMeanSquare {
    fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "rolling window capacity must be positive");
        Self {
            samples: VecDeque::with_capacity(capacity),
            capacity,
            sum: 0.0,
        }
    }

    fn push(&mut self, value: f64) {
        if self.samples.len() == self.capacity
            && let Some(oldest) = self.samples.pop_front()
        {
            self.sum -= oldest;
        }
        self.samples.push_back(value);
        self.sum += value;
    }

    #[inline]
    fn mean(&self) -> f64 {
        if self.samples.is_empty() {
            0.0
        } else {
            self.sum / self.samples.len() as f64
        }
    }
}

#[derive(Debug, Clone)]
struct ChannelState {
    short_term: RollingMeanSquare,
    rms_fast: RollingMeanSquare,
    filter: KWeightingFilter,
    peak_linear: f32,
}

impl ChannelState {
    fn new(short_term_capacity: usize, rms_capacity: usize, sample_rate: f32) -> Self {
        Self {
            short_term: RollingMeanSquare::new(short_term_capacity),
            rms_fast: RollingMeanSquare::new(rms_capacity),
            filter: KWeightingFilter::new(sample_rate),
            peak_linear: 0.0,
        }
    }
}

/// Combined loudness statistics produced by the loudness processor.
#[derive(Debug, Clone, Default)]
pub struct LoudnessSnapshot {
    pub short_term_lufs: Vec<f32>,
    pub rms_fast_db: Vec<f32>,
    pub true_peak_db: Vec<f32>,
}

impl LoudnessSnapshot {
    fn with_channels(channels: usize, floor_lufs: f32) -> Self {
        Self {
            short_term_lufs: vec![floor_lufs; channels],
            rms_fast_db: vec![floor_lufs; channels],
            true_peak_db: vec![floor_lufs; channels],
        }
    }
}

/// Configuration options for the loudness processor.
#[derive(Debug, Clone, Copy)]
pub struct LoudnessConfig {
    pub sample_rate: f32,
    /// Window size in seconds for LUFS short-term (~3.0s).
    pub short_term_window: f32,
    /// Window size in seconds for RMS fast (~0.3s).
    pub rms_fast_window: f32,
    /// Floor applied to LUFS/peak values to avoid `-inf`.
    pub floor_lufs: f32,
}

impl Default for LoudnessConfig {
    fn default() -> Self {
        Self {
            sample_rate: DEFAULT_SAMPLE_RATE,
            short_term_window: DEFAULT_SHORT_TERM_WINDOW,
            rms_fast_window: DEFAULT_RMS_FAST_WINDOW,
            floor_lufs: DEFAULT_FLOOR_LUFS,
        }
    }
}

/// Loudness processor that tracks per-channel LUFS, RMS, and true-peak values.
#[derive(Debug, Clone)]
pub struct LoudnessProcessor {
    config: LoudnessConfig,
    channels: Vec<ChannelState>,
    snapshot: LoudnessSnapshot,
}

impl LoudnessProcessor {
    pub fn new(config: LoudnessConfig) -> Self {
        Self {
            channels: Vec::new(),
            snapshot: LoudnessSnapshot::default(),
            config,
        }
    }

    pub fn config(&self) -> LoudnessConfig {
        self.config
    }

    pub fn snapshot(&self) -> &LoudnessSnapshot {
        &self.snapshot
    }

    fn ensure_state(&mut self, requested_channels: usize, sample_rate: f32) {
        let channels = requested_channels.max(1);
        let rate_changed = sample_rate.is_finite()
            && sample_rate > 0.0
            && (self.config.sample_rate - sample_rate).abs() > f32::EPSILON;

        if rate_changed {
            self.config.sample_rate = sample_rate;
        }

        if rate_changed || self.channels.len() != channels {
            self.rebuild_state(channels);
        }
    }

    fn rebuild_state(&mut self, channels: usize) {
        let short_term_capacity =
            window_length(self.config.sample_rate, self.config.short_term_window);
        let rms_capacity = window_length(self.config.sample_rate, self.config.rms_fast_window);

        self.channels = (0..channels)
            .map(|_| ChannelState::new(short_term_capacity, rms_capacity, self.config.sample_rate))
            .collect();
        self.snapshot = LoudnessSnapshot::with_channels(channels, self.config.floor_lufs);
    }
}

impl AudioProcessor for LoudnessProcessor {
    type Output = LoudnessSnapshot;

    fn process_block(&mut self, block: &AudioBlock<'_>) -> ProcessorUpdate<Self::Output> {
        if block.channels == 0 || block.frame_count() == 0 {
            return ProcessorUpdate::None;
        }

        self.ensure_state(block.channels, block.sample_rate);

        if self.channels.is_empty() {
            return ProcessorUpdate::None;
        }

        for channel in &mut self.channels {
            channel.peak_linear = 0.0;
        }

        for frame in block.samples.chunks_exact(block.channels) {
            for (channel_state, &sample) in self.channels.iter_mut().zip(frame) {
                let filtered = channel_state.filter.process(sample);
                let energy = (filtered as f64).powi(2);
                channel_state.short_term.push(energy);
                channel_state.rms_fast.push(energy);
                channel_state.peak_linear = channel_state.peak_linear.max(sample.abs());
            }
        }

        let mut combined_short_term_energy = 0.0;

        for (index, channel_state) in self.channels.iter().enumerate() {
            let short_term_mean = channel_state.short_term.mean().max(MIN_MEAN_SQUARE);
            let rms_mean = channel_state.rms_fast.mean().max(MIN_MEAN_SQUARE);

            combined_short_term_energy += short_term_mean;
            self.snapshot.rms_fast_db[index] = mean_square_to_db(rms_mean, self.config.floor_lufs);
            self.snapshot.true_peak_db[index] =
                peak_to_db(channel_state.peak_linear, self.config.floor_lufs);
        }

        let combined_short_term_lufs =
            mean_square_to_db(combined_short_term_energy, self.config.floor_lufs);

        self.snapshot.short_term_lufs.fill(combined_short_term_lufs);

        ProcessorUpdate::Snapshot(self.snapshot.clone())
    }

    fn reset(&mut self) {
        let channels = self.channels.len();
        self.channels.clear();
        self.snapshot = if channels == 0 {
            LoudnessSnapshot::default()
        } else {
            LoudnessSnapshot::with_channels(channels, self.config.floor_lufs)
        };
    }
}

impl Reconfigurable<LoudnessConfig> for LoudnessProcessor {
    fn update_config(&mut self, config: LoudnessConfig) {
        self.config = config;
        let channels = self.channels.len();
        if channels > 0 {
            self.rebuild_state(channels);
        }
    }
}

#[derive(Debug, Clone)]
struct KWeightingFilter {
    pre: Biquad,
    high_pass: Biquad,
}

impl KWeightingFilter {
    fn new(sample_rate: f32) -> Self {
        let pre = Biquad::k_weighting_pre(sample_rate);
        let high_pass = Biquad::k_weighting_high_pass(sample_rate);
        Self { pre, high_pass }
    }

    fn process(&mut self, sample: f32) -> f32 {
        let stage1 = self.pre.process(sample);
        self.high_pass.process(stage1)
    }
}

#[derive(Debug, Clone)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1: f64,
    z2: f64,
}

impl Biquad {
    fn from_coefficients(b: [f64; 3], a: [f64; 3]) -> Self {
        debug_assert!(a[0] != 0.0, "digital biquad a0 must be non-zero");
        let inv_a0 = 1.0 / a[0];

        Self {
            b0: b[0] * inv_a0,
            b1: b[1] * inv_a0,
            b2: b[2] * inv_a0,
            a1: a[1] * inv_a0,
            a2: a[2] * inv_a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    #[inline]
    fn prewarp(freq_hz: f64, sample_rate: f64) -> f64 {
        (std::f64::consts::PI * freq_hz / sample_rate).tan() * 2.0 * sample_rate
    }

    fn new(analog_b: [f64; 3], analog_a: [f64; 3], sample_rate: f32) -> Self {
        let k = 2.0 * sample_rate as f64;
        let k2 = k * k;

        let (a0, a1, a2) = (analog_a[0], analog_a[1], analog_a[2]);
        let (b0, b1, b2) = (analog_b[0], analog_b[1], analog_b[2]);

        let a0d = a0 * k2 + a1 * k + a2;
        let a1d = 2.0 * (a2 - a0 * k2);
        let a2d = a0 * k2 - a1 * k + a2;

        let b0d = b0 * k2 + b1 * k + b2;
        let b1d = 2.0 * (b2 - b0 * k2);
        let b2d = b0 * k2 - b1 * k + b2;

        Self::from_coefficients([b0d, b1d, b2d], [a0d, a1d, a2d])
    }

    fn k_weighting_pre(sample_rate: f32) -> Self {
        if (sample_rate - NOMINAL_SAMPLE_RATE).abs() <= SAMPLE_RATE_TOLERANCE {
            return Self::from_coefficients(PRE_B_COEFFS_48K, PRE_A_COEFFS_48K);
        }

        let sr = sample_rate as f64;
        let w0 = Self::prewarp(15.915, sr);
        let w1 = Self::prewarp(4.078, sr);
        Self::new([1.0, w0, w0 * w0], [1.0, w1, w1 * w1], sample_rate)
    }

    fn k_weighting_high_pass(sample_rate: f32) -> Self {
        if (sample_rate - NOMINAL_SAMPLE_RATE).abs() <= SAMPLE_RATE_TOLERANCE {
            return Self::from_coefficients(HP_B_COEFFS_48K, HP_A_COEFFS_48K);
        }

        let sr = sample_rate as f64;
        let wh = Self::prewarp(38.1358, sr);
        Self::new([1.0, 0.0, 0.0], [1.0, wh, wh * wh], sample_rate)
    }

    #[inline]
    fn process(&mut self, sample: f32) -> f32 {
        let x = sample as f64;
        let y = x * self.b0 + self.z1;
        self.z1 = x * self.b1 + self.z2 - self.a1 * y;
        self.z2 = x * self.b2 - self.a2 * y;
        y as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ebur128::{EbuR128, Mode};
    use std::time::Instant;

    fn sine_wave(sample_rate: f32, duration: f32, freq: f32, amplitude: f32) -> Vec<f32> {
        let frames = (sample_rate * duration) as usize;
        (0..frames)
            .map(|n| {
                let phase = 2.0 * std::f32::consts::PI * freq * n as f32 / sample_rate;
                phase.sin() * amplitude
            })
            .collect()
    }

    #[test]
    fn rolling_mean_square_tracks_average() {
        let mut window = RollingMeanSquare::new(4);
        window.push(1.0);
        window.push(9.0);
        assert!((window.mean() - 5.0).abs() < f64::EPSILON);

        window.push(16.0);
        window.push(25.0);
        window.push(36.0);
        // Now should hold 9,16,25,36
        assert!((window.mean() - 21.5).abs() < f64::EPSILON);
    }

    #[test]
    fn processor_estimates_short_term_and_rms() {
        fn measure(amp: f32) -> (f32, f32) {
            let samples = sine_wave(DEFAULT_SAMPLE_RATE, 3.0, 1_000.0, amp);
            let mut processor = LoudnessProcessor::new(LoudnessConfig::default());
            let block = AudioBlock::new(&samples, 1, DEFAULT_SAMPLE_RATE, Instant::now());
            match processor.process_block(&block) {
                ProcessorUpdate::Snapshot(s) => (s.short_term_lufs[0], s.rms_fast_db[0]),
                ProcessorUpdate::None => panic!("expected snapshot"),
            }
        }

        let (st_low, rms_low) = measure(0.25);
        let (st_high, rms_high) = measure(0.5);

        assert!(st_high > st_low);
        assert!(rms_high > rms_low);

        let st_delta = st_high - st_low;
        let rms_delta = rms_high - rms_low;

        assert!(st_delta > 5.0 && st_delta < 7.0);
        assert!(rms_delta > 5.0 && rms_delta < 7.0);
    }

    #[test]
    fn processor_tracks_peak() {
        let mut processor = LoudnessProcessor::new(LoudnessConfig::default());
        let mut samples = vec![0.0; 2048];
        samples[0] = 0.9;
        let block = AudioBlock::new(&samples, 2, DEFAULT_SAMPLE_RATE, Instant::now());
        let snapshot = match processor.process_block(&block) {
            ProcessorUpdate::Snapshot(s) => s,
            ProcessorUpdate::None => panic!("expected snapshot"),
        };
        assert!(snapshot.true_peak_db[0] > -1.0);
    }

    #[test]
    fn processor_sums_channel_energy_before_log() {
        let mono = sine_wave(DEFAULT_SAMPLE_RATE, 3.0, 1_000.0, 0.5);
        let stereo: Vec<f32> = mono.iter().flat_map(|&s| [s, s]).collect();

        let mut mono_processor = LoudnessProcessor::new(LoudnessConfig::default());
        let mut stereo_processor = LoudnessProcessor::new(LoudnessConfig::default());

        let mono_block = AudioBlock::new(&mono, 1, DEFAULT_SAMPLE_RATE, Instant::now());
        let stereo_block = AudioBlock::new(&stereo, 2, DEFAULT_SAMPLE_RATE, Instant::now());

        let mono_snapshot = match mono_processor.process_block(&mono_block) {
            ProcessorUpdate::Snapshot(s) => s,
            ProcessorUpdate::None => panic!("expected mono snapshot"),
        };

        let stereo_snapshot = match stereo_processor.process_block(&stereo_block) {
            ProcessorUpdate::Snapshot(s) => s,
            ProcessorUpdate::None => panic!("expected stereo snapshot"),
        };

        assert_eq!(stereo_snapshot.short_term_lufs.len(), 2);
        let stereo_left = stereo_snapshot.short_term_lufs[0];
        let stereo_right = stereo_snapshot.short_term_lufs[1];
        assert!((stereo_left - stereo_right).abs() < 1e-3);

        let diff = stereo_left - mono_snapshot.short_term_lufs[0];

        // Correlated stereo content should increase loudness by ~3.01 dB compared to mono.
        assert!(diff > 2.9 && diff < 3.1, "diff was {diff}");
    }

    #[test]
    fn processor_matches_ebur128_short_term_within_0_01_db() {
        let mono = sine_wave(DEFAULT_SAMPLE_RATE, 4.0, 1_000.0, 0.5);
        let interleaved: Vec<f32> = mono.iter().flat_map(|&s| [s, s]).collect();

        let mut processor = LoudnessProcessor::new(LoudnessConfig::default());
        let block = AudioBlock::new(&interleaved, 2, DEFAULT_SAMPLE_RATE, Instant::now());
        let snapshot = match processor.process_block(&block) {
            ProcessorUpdate::Snapshot(s) => s,
            ProcessorUpdate::None => panic!("expected stereo snapshot"),
        };

        let mut reference = EbuR128::new(2, DEFAULT_SAMPLE_RATE as u32, Mode::S).unwrap();
        reference
            .add_frames_planar_f32(&[&mono, &mono])
            .expect("failed to feed reference meter");
        let reference_short_term = reference
            .loudness_shortterm()
            .expect("reference short-term loudness unavailable");

        let ours = snapshot.short_term_lufs[0] as f64;
        let diff = (ours - reference_short_term).abs();
        assert!(
            diff < 0.01,
            "short-term loudness mismatch: ours={ours:.4} LUFS, reference={reference_short_term:.4} LUFS, diff={diff:.4}"
        );
    }
}
