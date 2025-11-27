//! ITU-R BS.1770-5 compliant loudness and true peak metering.

use super::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::util::audio::DEFAULT_SAMPLE_RATE;
use std::f64::consts::PI;

const MIN_MEAN_SQUARE: f64 = 1e-12;
const LOUDNESS_OFFSET: f64 = -0.691;
const DEFAULT_FLOOR_DB: f32 = -99.9;

const DEFAULT_WINDOWS: [f32; 4] = [3.0, 0.4, 0.3, 1.0];

// Window indices
const WIN_SHORT_TERM: usize = 0;
const WIN_MOMENTARY: usize = 1;
const WIN_RMS_FAST: usize = 2;
const WIN_RMS_SLOW: usize = 3;

// ITU-R BS.1770-5 K-weighting
fn k_weighting_coefficients(fs: f64) -> ([f64; 5], [f64; 5]) {
    // High-shelf (pre-filter): f0=1681.97Hz, G=+4dB, Q=0.7072
    let (f0, g, q) = (
        1_681.974_450_955_533,
        3.999_843_853_973_347,
        0.707_175_236_955_419_6,
    );
    let k = (PI * f0 / fs).tan();
    let vh = 10.0_f64.powf(g / 20.0);
    let vb = vh.powf(0.499_666_774_154_541_6);
    let a0 = 1.0 + k / q + k * k;
    let pb = [
        (vh + vb * k / q + k * k) / a0,
        2.0 * (k * k - vh) / a0,
        (vh - vb * k / q + k * k) / a0,
    ];
    let pa = [1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0];

    // Highpass: f0=38.14Hz, Q=0.5003
    let (f0, q) = (38.135_470_876_024_44, 0.500_327_037_323_877_3);
    let k = (PI * f0 / fs).tan();
    let a0 = 1.0 + k / q + k * k;
    let rb = [1.0, -2.0, 1.0];
    let ra = [1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0];

    // Convolve biquad coefficients to 4th-order
    let conv = |p: [f64; 3], r: [f64; 3]| {
        [
            p[0] * r[0],
            p[0] * r[1] + p[1] * r[0],
            p[0] * r[2] + p[1] * r[1] + p[2] * r[0],
            p[1] * r[2] + p[2] * r[1],
            p[2] * r[2],
        ]
    };
    (conv(pb, rb), conv(pa, ra))
}

#[inline]
fn mean_square_to_lufs(mean_square: f64, floor: f32) -> f32 {
    if mean_square <= MIN_MEAN_SQUARE {
        floor
    } else {
        mean_square
            .log10()
            .mul_add(10.0, LOUDNESS_OFFSET)
            .max(f64::from(floor)) as f32
    }
}

#[inline]
fn linear_to_db(linear: f32, floor: f32) -> f32 {
    if linear <= f32::EPSILON {
        floor
    } else {
        linear.log10().mul_add(20.0, 0.0).max(floor)
    }
}

#[inline]
const fn window_length(sample_rate: f32, window_secs: f32) -> usize {
    let len = sample_rate * window_secs;
    if len < 1.0 { 1 } else { len as usize }
}

// True peak (ITU-R BS.1770-5 Annex 2)
const TAPS: usize = 48;
const PHASES: usize = 4;
const TAPS_PER_PHASE: usize = TAPS / PHASES;

fn compute_fir_coefficients() -> [[f32; PHASES]; TAPS_PER_PHASE] {
    let mut h = [[0.0_f32; PHASES]; TAPS_PER_PHASE];
    let n = (TAPS + 1) as f64; // Window length for symmetric Hann
    for j in 0..TAPS {
        let m = j as f64 - (n - 1.0) / 2.0;
        let w = 0.5 * (1.0 - (2.0 * PI * j as f64 / (n - 1.0)).cos());
        let sinc = if m.abs() > 1e-6 {
            let x = m * PI / PHASES as f64;
            x.sin() / x
        } else {
            1.0
        };
        h[j / PHASES][j % PHASES] = (w * sinc) as f32;
    }
    h
}

#[derive(Debug, Clone)]
struct TruePeakMeter {
    buf: [f32; TAPS_PER_PHASE * 2],
    pos: usize,
    fir: [[f32; PHASES]; TAPS_PER_PHASE],
    peak: f32,
}

impl TruePeakMeter {
    fn new() -> Self {
        Self {
            buf: [0.0; TAPS_PER_PHASE * 2],
            pos: TAPS_PER_PHASE,
            fir: compute_fir_coefficients(),
            peak: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, sample: f32) {
        self.pos = if self.pos == 0 {
            TAPS_PER_PHASE - 1
        } else {
            self.pos - 1
        };
        self.buf[self.pos] = sample;
        self.buf[self.pos + TAPS_PER_PHASE] = sample;
        let mut out = [0.0_f32; PHASES];
        for (s, h) in self.buf[self.pos..self.pos + TAPS_PER_PHASE]
            .iter()
            .zip(&self.fir)
        {
            for (o, &c) in out.iter_mut().zip(h) {
                *o += s * c;
            }
        }
        for &v in &out {
            self.peak = self.peak.max(v.abs());
        }
    }

    #[inline]
    fn take_peak(&mut self) -> f32 {
        std::mem::take(&mut self.peak)
    }
}

#[derive(Debug, Clone)]
struct KWeightingFilter {
    b: [f64; 5],
    a: [f64; 5],
    z: [f64; 4],
}

impl KWeightingFilter {
    fn new(sample_rate: f64) -> Self {
        let (b, a) = k_weighting_coefficients(sample_rate);
        Self { b, a, z: [0.0; 4] }
    }

    #[inline]
    fn process(&mut self, sample: f32) -> f32 {
        let x = f64::from(sample);
        let y = self.b[0].mul_add(x, self.z[0]);
        self.z[0] = self.b[1].mul_add(x, self.z[1]) - self.a[1] * y;
        self.z[1] = self.b[2].mul_add(x, self.z[2]) - self.a[2] * y;
        self.z[2] = self.b[3].mul_add(x, self.z[3]) - self.a[3] * y;
        self.z[3] = self.b[4] * x - self.a[4] * y;
        y as f32
    }
}

#[derive(Debug, Clone)]
struct RollingMeanSquare {
    buffer: Box<[f64]>,
    head: usize,
    sum: f64,
    filled: bool,
}

impl RollingMeanSquare {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: vec![0.0; capacity.max(1)].into_boxed_slice(),
            head: 0,
            sum: 0.0,
            filled: false,
        }
    }

    #[inline]
    fn push(&mut self, value: f64) {
        if self.filled {
            self.sum -= self.buffer[self.head];
        }
        self.buffer[self.head] = value;
        self.sum += value;
        self.head += 1;
        if self.head >= self.buffer.len() {
            self.head = 0;
            self.filled = true;
        }
    }

    #[inline]
    fn mean(&self) -> f64 {
        let count = if self.filled {
            self.buffer.len()
        } else {
            self.head
        };
        if count == 0 {
            0.0
        } else {
            self.sum / count as f64
        }
    }
}

#[derive(Debug, Clone)]
struct ChannelState {
    windows: [RollingMeanSquare; 4],
    filter: KWeightingFilter,
    true_peak: TruePeakMeter,
}

impl ChannelState {
    fn new(capacities: [usize; 4], sample_rate: f64) -> Self {
        Self {
            windows: capacities.map(RollingMeanSquare::with_capacity),
            filter: KWeightingFilter::new(sample_rate),
            true_peak: TruePeakMeter::new(),
        }
    }
}

/// only supports up to stereo. will be expanded later, fuck off
pub const MAX_CHANNELS: usize = 2;

#[derive(Debug, Clone, Copy, Default)]
pub struct LoudnessSnapshot {
    pub short_term_loudness: [f32; MAX_CHANNELS],
    pub momentary_loudness: [f32; MAX_CHANNELS],
    pub rms_fast_db: [f32; MAX_CHANNELS],
    pub rms_slow_db: [f32; MAX_CHANNELS],
    pub true_peak_db: [f32; MAX_CHANNELS],
}

impl LoudnessSnapshot {
    fn with_floor(floor_db: f32) -> Self {
        Self {
            short_term_loudness: [floor_db; MAX_CHANNELS],
            momentary_loudness: [floor_db; MAX_CHANNELS],
            rms_fast_db: [floor_db; MAX_CHANNELS],
            rms_slow_db: [floor_db; MAX_CHANNELS],
            true_peak_db: [floor_db; MAX_CHANNELS],
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LoudnessConfig {
    pub sample_rate: f32,
    pub windows: [f32; 4],
    pub floor_db: f32,
}

impl Default for LoudnessConfig {
    fn default() -> Self {
        Self {
            sample_rate: DEFAULT_SAMPLE_RATE,
            windows: DEFAULT_WINDOWS,
            floor_db: DEFAULT_FLOOR_DB,
        }
    }
}

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
        let channels = channels.min(MAX_CHANNELS);
        let capacities = self
            .config
            .windows
            .map(|w| window_length(self.config.sample_rate, w));
        let sample_rate = f64::from(self.config.sample_rate);
        self.channels = (0..channels)
            .map(|_| ChannelState::new(capacities, sample_rate))
            .collect();
        self.snapshot = LoudnessSnapshot::with_floor(self.config.floor_db);
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

        for frame in block.samples.chunks_exact(block.channels) {
            for (ch, &sample) in self.channels.iter_mut().zip(frame) {
                let energy = f64::from(ch.filter.process(sample)).powi(2);
                for w in &mut ch.windows {
                    w.push(energy);
                }
                ch.true_peak.process(sample);
            }
        }

        let floor = self.config.floor_db;
        let num_channels = self.channels.len();
        let mut combined_short_term = 0.0;
        let mut combined_momentary = 0.0;

        for (i, ch) in self.channels.iter().enumerate() {
            let short_term = ch.windows[WIN_SHORT_TERM].mean().max(MIN_MEAN_SQUARE);
            let momentary = ch.windows[WIN_MOMENTARY].mean().max(MIN_MEAN_SQUARE);
            combined_short_term += short_term;
            combined_momentary += momentary;
            self.snapshot.rms_fast_db[i] =
                mean_square_to_lufs(ch.windows[WIN_RMS_FAST].mean().max(MIN_MEAN_SQUARE), floor);
            self.snapshot.rms_slow_db[i] =
                mean_square_to_lufs(ch.windows[WIN_RMS_SLOW].mean().max(MIN_MEAN_SQUARE), floor);
        }

        for (i, ch) in self.channels.iter_mut().enumerate() {
            self.snapshot.true_peak_db[i] = linear_to_db(ch.true_peak.take_peak(), floor);
        }

        let combined_short_term_loudness = mean_square_to_lufs(combined_short_term, floor);
        let combined_momentary_loudness = mean_square_to_lufs(combined_momentary, floor);

        self.snapshot.short_term_loudness[..num_channels].fill(combined_short_term_loudness);
        self.snapshot.momentary_loudness[..num_channels].fill(combined_momentary_loudness);

        ProcessorUpdate::Snapshot(self.snapshot)
    }

    fn reset(&mut self) {
        self.channels.clear();
        self.snapshot = LoudnessSnapshot::with_floor(self.config.floor_db);
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

#[cfg(test)]
mod tests {
    use super::*;
    use ebur128::{EbuR128, Mode};
    use std::time::Instant;

    fn sine_wave(rate: f32, secs: f32, freq: f32, amp: f32) -> Vec<f32> {
        (0..(rate * secs) as usize)
            .map(|n| (2.0 * std::f32::consts::PI * freq * n as f32 / rate).sin() * amp)
            .collect()
    }

    fn unwrap_snapshot(update: ProcessorUpdate<LoudnessSnapshot>) -> LoudnessSnapshot {
        match update {
            ProcessorUpdate::Snapshot(s) => s,
            ProcessorUpdate::None => panic!("expected snapshot"),
        }
    }

    #[test]
    fn rolling_mean_square_tracks_average() {
        let mut window = RollingMeanSquare::with_capacity(4);
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
        let measure = |amp| {
            let samples = sine_wave(DEFAULT_SAMPLE_RATE, 3.0, 1000.0, amp);
            let block = AudioBlock::new(&samples, 1, DEFAULT_SAMPLE_RATE, Instant::now());
            let s = unwrap_snapshot(
                LoudnessProcessor::new(LoudnessConfig::default()).process_block(&block),
            );
            (s.short_term_loudness[0], s.rms_fast_db[0])
        };
        let (st_low, rms_low) = measure(0.25);
        let (st_high, rms_high) = measure(0.5);
        let (st_delta, rms_delta) = (st_high - st_low, rms_high - rms_low);
        assert!(st_high > st_low && rms_high > rms_low);
        assert!((5.0..7.0).contains(&st_delta) && (5.0..7.0).contains(&rms_delta));
    }

    #[test]
    fn processor_tracks_peak() {
        let mut samples = vec![0.0; 2048];
        samples[0] = 0.9;
        let block = AudioBlock::new(&samples, 2, DEFAULT_SAMPLE_RATE, Instant::now());
        let s = unwrap_snapshot(
            LoudnessProcessor::new(LoudnessConfig::default()).process_block(&block),
        );
        assert!(s.true_peak_db[0] > -1.0);
    }

    #[test]
    fn processor_sums_channel_energy_before_log() {
        let mono = sine_wave(DEFAULT_SAMPLE_RATE, 3.0, 1000.0, 0.5);
        let stereo: Vec<f32> = mono.iter().flat_map(|&s| [s, s]).collect();
        let mono_s = unwrap_snapshot(
            LoudnessProcessor::new(LoudnessConfig::default()).process_block(&AudioBlock::new(
                &mono,
                1,
                DEFAULT_SAMPLE_RATE,
                Instant::now(),
            )),
        );
        let stereo_s = unwrap_snapshot(
            LoudnessProcessor::new(LoudnessConfig::default()).process_block(&AudioBlock::new(
                &stereo,
                2,
                DEFAULT_SAMPLE_RATE,
                Instant::now(),
            )),
        );
        assert!((stereo_s.short_term_loudness[0] - stereo_s.short_term_loudness[1]).abs() < 1e-3);
        // Correlated stereo content should increase loudness by ~3.01 dB compared to mono
        let diff = stereo_s.short_term_loudness[0] - mono_s.short_term_loudness[0];
        assert!((2.9..3.1).contains(&diff), "diff was {diff}");
    }

    #[test]
    fn processor_matches_ebur128_short_term() {
        for sample_rate in [44100.0_f32, 48000.0, 96000.0] {
            let mono = sine_wave(sample_rate, 4.0, 1000.0, 0.5);
            let stereo: Vec<f32> = mono.iter().flat_map(|&s| [s, s]).collect();
            let block = AudioBlock::new(&stereo, 2, sample_rate, Instant::now());
            let cfg = LoudnessConfig {
                sample_rate,
                ..Default::default()
            };
            let ours = unwrap_snapshot(LoudnessProcessor::new(cfg).process_block(&block))
                .short_term_loudness[0] as f64;

            let mut reference = EbuR128::new(2, sample_rate as u32, Mode::S).unwrap();
            reference.add_frames_planar_f32(&[&mono, &mono]).unwrap();
            let expected = reference.loudness_shortterm().unwrap();
            let diff = (ours - expected).abs();
            assert!(
                diff < 0.01,
                "{sample_rate}Hz mismatch: {ours:.4} vs {expected:.4} (diff={diff:.4})"
            );
        }
    }

    #[test]
    fn processor_matches_ebur128_true_peak() {
        let sample_rate = 48000.0_f32;
        let mono = sine_wave(sample_rate, 0.5, 17000.0, 0.9); // Near Nyquist for inter-sample peaks
        let stereo: Vec<f32> = mono.iter().flat_map(|&s| [s, s]).collect();
        let cfg = LoudnessConfig {
            sample_rate,
            ..Default::default()
        };
        let ours = unwrap_snapshot(LoudnessProcessor::new(cfg).process_block(&AudioBlock::new(
            &stereo,
            2,
            sample_rate,
            Instant::now(),
        )))
        .true_peak_db[0] as f64;

        let mut reference = EbuR128::new(2, sample_rate as u32, Mode::TRUE_PEAK | Mode::S).unwrap();
        reference.add_frames_planar_f32(&[&mono, &mono]).unwrap();
        let ref_db = 20.0 * reference.true_peak(0).unwrap().log10();
        let sample_peak_db =
            20.0 * (mono.iter().map(|x| x.abs()).fold(0.0_f32, f32::max) as f64).log10();

        assert!(
            (ours - ref_db).abs() < 1.0,
            "true peak: {ours:.2} vs {ref_db:.2} dBTP"
        );
        assert!(ours >= sample_peak_db - 0.1 && ref_db >= sample_peak_db - 0.1);
    }

    #[test]
    fn true_peak_meter_detects_inter_sample_peaks() {
        let mut meter = TruePeakMeter::new();

        // Feed enough samples to fill the filter buffer first
        // The filter has 12 taps, so we need at least 12 samples
        // before the output stabilizes
        for _ in 0..20 {
            meter.process(0.0);
        }

        // Now feed an alternating sequence that would create inter-sample peaks
        // At Nyquist frequency, samples alternate +/-
        for _ in 0..20 {
            meter.process(0.7);
            meter.process(-0.7);
        }

        let peak = meter.take_peak();
        // Inter-sample peak should be higher than the sample values
        // For alternating +/- 0.7 (Nyquist), the true peak can theoretically
        // be up to ~1.21× the sample amplitude due to reconstruction
        assert!(
            peak > 0.7,
            "inter-sample peak {peak:.4} should exceed sample amplitude 0.7"
        );
    }
}
