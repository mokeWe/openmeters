//! Spectrum analyser DSP scaffolding.

use super::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::dsp::spectrogram::{FrequencyScale, WindowKind};
use crate::util::audio::DEFAULT_SAMPLE_RATE;
use realfft::{RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex32;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

const LOG_FACTOR: f32 = 10.0 * core::f32::consts::LOG10_E;
const POWER_EPSILON: f32 = 1.0e-20;
const DB_FLOOR: f32 = -140.0;

pub const MIN_SPECTRUM_FFT_SIZE: usize = 128;
pub const DEFAULT_SPECTRUM_HOP_DIVISOR: usize = 4;
pub const MIN_SPECTRUM_EXP_FACTOR: f32 = 0.0;
pub const MAX_SPECTRUM_EXP_FACTOR: f32 = 0.95;
pub const MIN_SPECTRUM_PEAK_DECAY: f32 = 0.0;
pub const MAX_SPECTRUM_PEAK_DECAY: f32 = 60.0;
pub const DEFAULT_SPECTRUM_EXP_FACTOR: f32 = 0.5;
pub const DEFAULT_SPECTRUM_PEAK_DECAY: f32 = 12.0;

/// Output magnitude spectrum.
#[derive(Debug, Clone, Default)]
pub struct SpectrumSnapshot {
    pub frequency_bins: Vec<f32>,
    pub magnitudes_db: Vec<f32>,
    pub magnitudes_unweighted_db: Vec<f32>,
    pub peak_frequency_hz: Option<f32>,
}

/// Configuration for the spectrum analyser.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SpectrumConfig {
    pub sample_rate: f32,
    pub fft_size: usize,
    /// Hop size between successive FFT evaluations.
    pub hop_size: usize,
    pub window: WindowKind,
    pub averaging: AveragingMode,
    pub frequency_scale: FrequencyScale,
    pub reverse_frequency: bool,
    pub show_grid: bool,
    pub show_peak_label: bool,
}

impl Default for SpectrumConfig {
    fn default() -> Self {
        Self {
            sample_rate: DEFAULT_SAMPLE_RATE,
            fft_size: 2048,
            hop_size: 256,
            window: WindowKind::PlanckBessel {
                epsilon: 0.1,
                beta: 5.5,
            },
            averaging: AveragingMode::Exponential {
                factor: DEFAULT_SPECTRUM_EXP_FACTOR,
            },
            frequency_scale: FrequencyScale::Logarithmic,
            reverse_frequency: false,
            show_grid: true,
            show_peak_label: true,
        }
    }
}

impl SpectrumConfig {
    /// Ensures the configuration respects runtime invariants and sane defaults.
    pub fn normalize(&mut self) {
        if !self.sample_rate.is_finite() || self.sample_rate <= 0.0 {
            self.sample_rate = DEFAULT_SAMPLE_RATE;
        }

        self.fft_size = self.fft_size.max(MIN_SPECTRUM_FFT_SIZE);

        self.hop_size = if self.hop_size == 0 {
            (self.fft_size / DEFAULT_SPECTRUM_HOP_DIVISOR).max(1)
        } else {
            self.hop_size.min(self.fft_size).max(1)
        };

        self.averaging = self.averaging.normalized();
    }

    /// Returns a normalized copy of this configuration.
    pub fn normalized(mut self) -> Self {
        self.normalize();
        self
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum AveragingMode {
    None,
    Exponential { factor: f32 },
    PeakHold { decay_per_second: f32 },
}

impl AveragingMode {
    pub fn normalized(self) -> Self {
        match self {
            AveragingMode::None => AveragingMode::None,
            AveragingMode::Exponential { factor } => AveragingMode::Exponential {
                factor: Self::clamp_factor(factor),
            },
            AveragingMode::PeakHold { decay_per_second } => AveragingMode::PeakHold {
                decay_per_second: Self::clamp_decay(decay_per_second),
            },
        }
    }

    pub const fn default_exponential_factor() -> f32 {
        DEFAULT_SPECTRUM_EXP_FACTOR
    }

    pub const fn default_peak_decay() -> f32 {
        DEFAULT_SPECTRUM_PEAK_DECAY
    }

    pub fn clamp_factor(value: f32) -> f32 {
        let value = if value.is_finite() {
            value
        } else {
            MIN_SPECTRUM_EXP_FACTOR
        };
        value.clamp(MIN_SPECTRUM_EXP_FACTOR, MAX_SPECTRUM_EXP_FACTOR)
    }

    pub fn clamp_decay(value: f32) -> f32 {
        let value = if value.is_finite() {
            value
        } else {
            MIN_SPECTRUM_PEAK_DECAY
        };
        value.clamp(MIN_SPECTRUM_PEAK_DECAY, MAX_SPECTRUM_PEAK_DECAY)
    }
}

pub struct SpectrumProcessor {
    config: SpectrumConfig,
    snapshot: SpectrumSnapshot,
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    real_buffer: Vec<f32>,
    spectrum_buffer: Vec<Complex32>,
    scratch_buffer: Vec<Complex32>,
    bin_normalization: Vec<f32>,
    pcm_buffer: VecDeque<f32>,
    averaged_db: Vec<f32>,
    peak_hold_db: Vec<f32>,
    scratch_magnitudes: Vec<f32>,
    averaged_unweighted_db: Vec<f32>,
    peak_hold_unweighted_db: Vec<f32>,
    scratch_unweighted: Vec<f32>,
    a_weighting_db: Vec<f32>,
    last_update_at: Option<Instant>,
}

impl SpectrumProcessor {
    pub fn new(mut config: SpectrumConfig) -> Self {
        config.normalize();
        let fft_size = config.fft_size;
        let mut processor = Self {
            config,
            snapshot: SpectrumSnapshot::default(),
            fft: RealFftPlanner::<f32>::new().plan_fft_forward(fft_size),
            window: Vec::new(),
            real_buffer: Vec::new(),
            spectrum_buffer: Vec::new(),
            scratch_buffer: Vec::new(),
            bin_normalization: Vec::new(),
            pcm_buffer: VecDeque::new(),
            averaged_db: Vec::new(),
            peak_hold_db: Vec::new(),
            scratch_magnitudes: Vec::new(),
            averaged_unweighted_db: Vec::new(),
            peak_hold_unweighted_db: Vec::new(),
            scratch_unweighted: Vec::new(),
            a_weighting_db: Vec::new(),
            last_update_at: None,
        };
        processor.rebuild_fft();
        processor
    }

    pub fn config(&self) -> SpectrumConfig {
        self.config
    }

    fn rebuild_fft(&mut self) {
        self.config.normalize();
        let fft_size = self.config.fft_size;
        let mut planner = RealFftPlanner::<f32>::new();
        self.fft = planner.plan_fft_forward(fft_size);
        self.window = self.config.window.coefficients(fft_size);
        self.real_buffer.resize(fft_size, 0.0);
        self.spectrum_buffer = self.fft.make_output_vec();
        self.scratch_buffer = self.fft.make_scratch_vec();
        self.bin_normalization = compute_bin_normalization(&self.window);
        let bins = fft_size / 2 + 1;
        self.snapshot.frequency_bins = frequency_bins(self.config.sample_rate, fft_size);
        self.snapshot.magnitudes_db = vec![DB_FLOOR; bins];
        self.snapshot.magnitudes_unweighted_db = vec![DB_FLOOR; bins];
        self.snapshot.peak_frequency_hz = None;
        self.averaged_db = vec![DB_FLOOR; bins];
        self.peak_hold_db = vec![DB_FLOOR; bins];
        self.scratch_magnitudes = vec![DB_FLOOR; bins];
        self.averaged_unweighted_db = vec![DB_FLOOR; bins];
        self.peak_hold_unweighted_db = vec![DB_FLOOR; bins];
        self.scratch_unweighted = vec![DB_FLOOR; bins];
        self.a_weighting_db = self
            .snapshot
            .frequency_bins
            .iter()
            .map(|&f| a_weight(f))
            .collect();
        self.pcm_buffer.clear();
    }

    fn ensure_fft(&mut self) {
        if self.real_buffer.len() != self.config.fft_size {
            self.rebuild_fft();
        }
    }

    fn mixdown_into(&mut self, block: &AudioBlock<'_>) {
        if block.channels == 0 || block.samples.is_empty() {
            return;
        }

        match block.channels {
            1 => {
                for &sample in block.samples {
                    self.pcm_buffer.push_back(sample);
                }
            }
            2 => {
                let mut iter = block.samples.chunks_exact(2);
                for frame in iter.by_ref() {
                    self.pcm_buffer.push_back((frame[0] + frame[1]) * 0.5);
                }
                if let [last] = iter.remainder() {
                    self.pcm_buffer.push_back(*last);
                }
            }
            channels => {
                let inv = 1.0 / channels as f32;
                for frame in block.samples.chunks_exact(channels) {
                    let sum: f32 = frame.iter().copied().sum();
                    self.pcm_buffer.push_back(sum * inv);
                }
            }
        }
    }

    fn process_ready_windows(&mut self, timestamp: Instant) -> bool {
        let fft_size = self.config.fft_size;
        let hop = self.config.hop_size.max(1);
        let bins = fft_size / 2 + 1;
        let mut produced = false;

        while self.pcm_buffer.len() >= fft_size {
            for (sample, target) in self
                .pcm_buffer
                .iter()
                .take(fft_size)
                .zip(self.real_buffer.iter_mut())
            {
                *target = *sample;
            }

            remove_dc(&mut self.real_buffer);
            apply_window(&mut self.real_buffer, &self.window);

            self.fft
                .process_with_scratch(
                    &mut self.real_buffer,
                    &mut self.spectrum_buffer,
                    &mut self.scratch_buffer,
                )
                .expect("real FFT forward transform");

            if self.scratch_magnitudes.len() != bins {
                self.scratch_magnitudes.resize(bins, DB_FLOOR);
            }
            if self.scratch_unweighted.len() != bins {
                self.scratch_unweighted.resize(bins, DB_FLOOR);
            }

            for (idx, (complex, norm)) in self
                .spectrum_buffer
                .iter()
                .zip(&self.bin_normalization)
                .take(bins)
                .enumerate()
            {
                let power = (complex.norm_sqr() * *norm).max(POWER_EPSILON);
                let raw_magnitude = (power.ln() * LOG_FACTOR).max(DB_FLOOR);
                self.scratch_unweighted[idx] = raw_magnitude;
                let weight = self.a_weighting_db.get(idx).copied().unwrap_or_else(|| {
                    a_weight(*self.snapshot.frequency_bins.get(idx).unwrap_or(&0.0))
                });
                let weighted = (raw_magnitude + weight).max(DB_FLOOR);
                self.scratch_magnitudes[idx] = weighted;
            }

            let peak_index = averaging_update(
                &self.config.averaging,
                &mut self.averaged_db,
                &mut self.peak_hold_db,
                &mut self.snapshot.magnitudes_db,
                &self.scratch_magnitudes,
                self.last_update_at,
                timestamp,
            );

            averaging_update(
                &self.config.averaging,
                &mut self.averaged_unweighted_db,
                &mut self.peak_hold_unweighted_db,
                &mut self.snapshot.magnitudes_unweighted_db,
                &self.scratch_unweighted,
                self.last_update_at,
                timestamp,
            );

            self.snapshot.peak_frequency_hz =
                peak_index.and_then(|idx| self.snapshot.frequency_bins.get(idx).copied());
            self.last_update_at = Some(timestamp);

            for _ in 0..hop {
                self.pcm_buffer.pop_front();
            }

            produced = true;
        }

        produced
    }
}

impl fmt::Debug for SpectrumProcessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpectrumProcessor")
            .field("config", &self.config)
            .finish()
    }
}

impl AudioProcessor for SpectrumProcessor {
    type Output = SpectrumSnapshot;

    fn process_block(&mut self, block: &AudioBlock<'_>) -> ProcessorUpdate<Self::Output> {
        if block.frame_count() == 0 {
            return ProcessorUpdate::None;
        }

        if (block.sample_rate - self.config.sample_rate).abs() > f32::EPSILON {
            self.config.sample_rate = block.sample_rate;
            self.rebuild_fft();
        }

        self.ensure_fft();
        self.mixdown_into(block);

        if self.process_ready_windows(block.timestamp) {
            ProcessorUpdate::Snapshot(self.snapshot.clone())
        } else {
            ProcessorUpdate::None
        }
    }

    fn reset(&mut self) {
        self.snapshot = SpectrumSnapshot::default();
        self.pcm_buffer.clear();
        self.averaged_db.clear();
        self.peak_hold_db.clear();
        self.averaged_unweighted_db.clear();
        self.peak_hold_unweighted_db.clear();
        self.scratch_unweighted.clear();
        self.last_update_at = None;
    }
}

impl Reconfigurable<SpectrumConfig> for SpectrumProcessor {
    fn update_config(&mut self, config: SpectrumConfig) {
        self.config = config.normalized();
        self.rebuild_fft();
    }
}

fn averaging_update(
    mode: &AveragingMode,
    averaged_db: &mut Vec<f32>,
    peak_hold_db: &mut Vec<f32>,
    output: &mut Vec<f32>,
    input: &[f32],
    last_timestamp: Option<Instant>,
    current_timestamp: Instant,
) -> Option<usize> {
    let bins = input.len();
    if averaged_db.len() != bins {
        averaged_db.resize(bins, DB_FLOOR);
    }
    if peak_hold_db.len() != bins {
        peak_hold_db.resize(bins, DB_FLOOR);
    }

    if output.len() != bins {
        output.resize(bins, DB_FLOOR);
    }

    let dt = last_timestamp
        .map(|last| current_timestamp.saturating_duration_since(last))
        .unwrap_or_default();
    let dt_seconds = dt.as_secs_f32().max(0.0);
    let mut peak_index = None;
    let mut peak_value = DB_FLOOR;

    match mode {
        AveragingMode::None => {
            for (idx, value) in input.iter().enumerate() {
                let val = value.max(DB_FLOOR);
                output[idx] = val;
                if val > peak_value {
                    peak_value = val;
                    peak_index = Some(idx);
                }
            }
        }
        AveragingMode::Exponential { factor } => {
            let alpha = factor.clamp(0.0, 0.9999);
            for (idx, value) in input.iter().enumerate() {
                let previous = averaged_db[idx];
                let smoothed = if previous <= DB_FLOOR + f32::EPSILON {
                    *value
                } else {
                    previous * alpha + *value * (1.0 - alpha)
                };
                averaged_db[idx] = smoothed;
                let val = smoothed.max(DB_FLOOR);
                output[idx] = val;
                if val > peak_value {
                    peak_value = val;
                    peak_index = Some(idx);
                }
            }
        }
        AveragingMode::PeakHold { decay_per_second } => {
            let decay = (decay_per_second.max(0.0)) * dt_seconds;
            for (idx, value) in input.iter().enumerate() {
                let decayed = (peak_hold_db[idx] - decay).max(DB_FLOOR);
                let hold = decayed.max(*value);
                peak_hold_db[idx] = hold;
                output[idx] = hold;
                if hold > peak_value {
                    peak_value = hold;
                    peak_index = Some(idx);
                }
            }
        }
    }

    peak_index
}

fn apply_window(buffer: &mut [f32], window: &[f32]) {
    debug_assert_eq!(buffer.len(), window.len());
    for (sample, coeff) in buffer.iter_mut().zip(window.iter()) {
        *sample *= *coeff;
    }
}

fn remove_dc(buffer: &mut [f32]) {
    if buffer.is_empty() {
        return;
    }

    let mean = buffer.iter().sum::<f32>() / buffer.len() as f32;
    if mean.abs() <= f32::EPSILON {
        return;
    }

    for sample in buffer.iter_mut() {
        *sample -= mean;
    }
}

fn compute_bin_normalization(window: &[f32]) -> Vec<f32> {
    let fft_size = window.len();
    let bins = fft_size / 2 + 1;
    if bins == 0 {
        return Vec::new();
    }

    let window_sum: f32 = window.iter().sum();
    let inv_sum = if window_sum.abs() > f32::EPSILON {
        1.0 / window_sum
    } else if fft_size > 0 {
        1.0 / fft_size as f32
    } else {
        0.0
    };

    let dc_scale = inv_sum * inv_sum;
    let ac_scale = (2.0 * inv_sum) * (2.0 * inv_sum);
    let mut norms = vec![ac_scale; bins];
    norms[0] = dc_scale;
    if bins > 1 {
        norms[bins - 1] = dc_scale;
    }
    norms
}

fn a_weight(freq_hz: f32) -> f32 {
    const MIN_DB: f32 = -80.0;
    if freq_hz <= 0.0 {
        return MIN_DB;
    }

    // IEC 61672-1:2013 reference frequencies.
    const C1: f64 = 20.598_997 * 20.598_997;
    const C2: f64 = 107.652_65 * 107.652_65;
    const C3: f64 = 737.862_23 * 737.862_23;
    const C4: f64 = 12_194.217 * 12_194.217;

    let f = freq_hz as f64;
    let f2 = f * f;
    let numerator = C4 * f2 * f2;
    let denom = (f2 + C1) * ((f2 + C2) * (f2 + C3)).sqrt() * (f2 + C4);

    if denom <= 0.0 || numerator <= 0.0 {
        return MIN_DB;
    }

    let ra = numerator / denom;
    let db = 20.0 * ra.log10() + 2.0;
    db.max(MIN_DB as f64) as f32
}

fn frequency_bins(sample_rate: f32, fft_size: usize) -> Vec<f32> {
    if fft_size == 0 {
        return Vec::new();
    }

    let bins = fft_size / 2 + 1;
    let bin_hz = if sample_rate > 0.0 {
        sample_rate / fft_size as f32
    } else {
        0.0
    };
    (0..bins).map(|i| i as f32 * bin_hz).collect()
}

#[cfg(test)]
mod tests {
    use super::a_weight;

    #[test]
    fn a_weight_matches_iec_reference_points() {
        let reference_points: &[(f32, f32)] = &[
            // (frequency Hz, reference dB)
            (31.5, -39.4),
            (63.0, -26.2),
            (100.0, -19.1),
            (200.0, -10.9),
            (500.0, -3.2),
            (1000.0, 0.0),
            (2000.0, 1.2),
            (4000.0, 1.0),
            (8000.0, -1.1),
            (16000.0, -6.6),
        ];

        for &(freq, expected_db) in reference_points {
            let actual = a_weight(freq);
            let delta = (actual - expected_db).abs();
            assert!(
                delta <= 0.15,
                "A-weight mismatch at {freq} Hz: expected {expected_db} dB, got {actual} dB (delta={delta})"
            );
        }
    }
}
