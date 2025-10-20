//! Spectrogram DSP implementation built on a short-time Fourier transform.
//!

use super::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::util::audio::DEFAULT_SAMPLE_RATE;
use parking_lot::RwLock;
use realfft::{RealFftPlanner, RealToComplex};
use rustc_hash::FxHashMap;
use rustfft::num_complex::Complex32;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

const LOG_FACTOR: f32 = 10.0 * core::f32::consts::LOG10_E;
const POWER_EPSILON: f32 = 1.0e-18;
const DB_FLOOR: f32 = -120.0;

/// Configuration for spectrogram FFT analysis.
#[derive(Debug, Clone, Copy)]
pub struct SpectrogramConfig {
    pub sample_rate: f32,
    /// FFT size (must be a power of two for radix-2 implementations).
    pub fft_size: usize,
    /// Hop size between successive frames.
    pub hop_size: usize,
    /// Window selection controlling spectral leakage characteristics.
    pub window: WindowKind,
    /// Frequency scaling mode for display.
    pub frequency_scale: FrequencyScale,
    /// Maximum retained history columns.
    pub history_length: usize,
    /// Enable time-frequency reassignment for sharper spectral localization.
    pub use_reassignment: bool,
    /// Linear power floor in dBFS below which bins are skipped during reassignment.
    pub reassignment_power_floor_db: f32,
    /// Limit reassignment to the lowest N frequency bins (0 to disable limiting).
    pub reassignment_low_bin_limit: usize,
    /// Zero-padding factor applied before the FFT (1 = no padding).
    pub zero_padding_factor: usize,
    /// Enable synchrosqueezed accumulation on a log-frequency grid.
    pub use_synchrosqueezing: bool,
    /// Number of log-frequency bins used for synchrosqueezed accumulation.
    pub synchrosqueezing_bin_count: usize,
    /// Lowest frequency in Hz included in the synchrosqueezed grid.
    pub synchrosqueezing_min_hz: f32,
    /// Exponential smoothing factor applied per-bin in the time domain (0 disables).
    pub temporal_smoothing: f32,
    /// Highest frequency in Hz receiving full temporal smoothing (0 disables frequency gating).
    pub temporal_smoothing_max_hz: f32,
    /// Transition width in Hz used to fade temporal smoothing to zero above the max frequency.
    pub temporal_smoothing_blend_hz: f32,
    /// Averaging radius in bins for frequency-domain smoothing (0 disables).
    pub frequency_smoothing_radius: usize,
    /// Highest frequency in Hz receiving full frequency smoothing (0 disables gating).
    pub frequency_smoothing_max_hz: f32,
    /// Transition width in Hz used to fade frequency smoothing to zero above the max frequency.
    pub frequency_smoothing_blend_hz: f32,
}

impl Default for SpectrogramConfig {
    fn default() -> Self {
        Self {
            sample_rate: DEFAULT_SAMPLE_RATE,
            fft_size: 4096,
            hop_size: 512,
            // I'm not sure what the "best" alpha/beta value is for spectral analysis.
            // going for moderate sidelobe suppression without excessive main lobe widening.
            window: WindowKind::PlanckBessel {
                epsilon: 0.1,
                beta: 5.5,
            },
            frequency_scale: FrequencyScale::default(),
            history_length: 480,
            use_reassignment: true,
            reassignment_power_floor_db: -80.0,
            reassignment_low_bin_limit: 0,
            zero_padding_factor: 2,
            use_synchrosqueezing: true,
            synchrosqueezing_bin_count: 1024,
            synchrosqueezing_min_hz: 20.0,
            temporal_smoothing: 0.4,
            temporal_smoothing_max_hz: 900.0,
            temporal_smoothing_blend_hz: 400.0,
            frequency_smoothing_radius: 5,
            frequency_smoothing_max_hz: 2400.0,
            frequency_smoothing_blend_hz: 600.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrequencyScale {
    Linear,
    Logarithmic,
    Mel,
}

impl Default for FrequencyScale {
    fn default() -> Self {
        Self::Logarithmic
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WindowKind {
    Rectangular,
    Hann,
    Hamming,
    Blackman,
    PlanckBessel { epsilon: f32, beta: f32 },
}

impl WindowKind {
    pub(crate) fn coefficients(self, len: usize) -> Vec<f32> {
        if len <= 1 {
            return vec![1.0; len];
        }

        match self {
            WindowKind::Rectangular => vec![1.0; len],
            WindowKind::Hann => {
                let scale = core::f32::consts::TAU / len as f32;
                (0..len)
                    .map(|n| 0.5 * (1.0 - (n as f32 * scale).cos()))
                    .collect()
            }
            WindowKind::Hamming => {
                let scale = core::f32::consts::TAU / len as f32;
                (0..len)
                    .map(|n| 0.54 - 0.46 * (n as f32 * scale).cos())
                    .collect()
            }
            WindowKind::Blackman => {
                let scale = core::f32::consts::TAU / len as f32;
                (0..len)
                    .map(|n| {
                        let phase = n as f32 * scale;
                        0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos()
                    })
                    .collect()
            }
            WindowKind::PlanckBessel { epsilon, beta } => {
                planck_bessel_coefficients(len, epsilon, beta)
            }
        }
    }
}

impl PartialEq for WindowKind {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (WindowKind::Rectangular, WindowKind::Rectangular)
            | (WindowKind::Hann, WindowKind::Hann)
            | (WindowKind::Hamming, WindowKind::Hamming)
            | (WindowKind::Blackman, WindowKind::Blackman) => true,
            (
                WindowKind::PlanckBessel {
                    epsilon: ea,
                    beta: ba,
                },
                WindowKind::PlanckBessel {
                    epsilon: eb,
                    beta: bb,
                },
            ) => ea.to_bits() == eb.to_bits() && ba.to_bits() == bb.to_bits(),
            _ => false,
        }
    }
}

impl Eq for WindowKind {}

impl Hash for WindowKind {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            WindowKind::Rectangular => state.write_u8(0),
            WindowKind::Hann => state.write_u8(1),
            WindowKind::Hamming => state.write_u8(2),
            WindowKind::Blackman => state.write_u8(3),
            WindowKind::PlanckBessel { epsilon, beta } => {
                state.write_u8(4);
                state.write_u32(epsilon.to_bits());
                state.write_u32(beta.to_bits());
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct WindowKey {
    kind: WindowKind,
    len: usize,
}

struct WindowCache {
    entries: RwLock<FxHashMap<WindowKey, Arc<[f32]>>>,
}

impl WindowCache {
    fn global() -> &'static WindowCache {
        static INSTANCE: OnceLock<WindowCache> = OnceLock::new();
        INSTANCE.get_or_init(|| WindowCache {
            entries: RwLock::new(FxHashMap::default()),
        })
    }

    fn get(&self, kind: WindowKind, len: usize) -> Arc<[f32]> {
        if len == 0 {
            return Arc::from([]);
        }

        let key = WindowKey { kind, len };

        // Fast path: check if already cached
        if let Some(existing) = self.entries.read().get(&key) {
            return Arc::clone(existing);
        }

        // Slow path: compute and cache
        let mut entries = self.entries.write();
        Arc::clone(
            entries
                .entry(key)
                .or_insert_with(|| Arc::from(kind.coefficients(len))),
        )
    }
}

fn kaiser_coefficients(len: usize, beta: f32) -> Vec<f32> {
    if len <= 1 {
        return vec![1.0; len];
    }

    let beta = beta.max(0.0) as f64;
    let denom = modified_bessel_i0(beta);
    let span = (len - 1) as f64;

    (0..len)
        .map(|n| {
            let ratio = (2.0 * n as f64) / span - 1.0;
            let inside = (1.0 - ratio * ratio).max(0.0).sqrt();
            (modified_bessel_i0(beta * inside) / denom) as f32
        })
        .collect()
}

fn planck_bessel_coefficients(len: usize, epsilon: f32, beta: f32) -> Vec<f32> {
    if len <= 1 {
        return vec![1.0; len];
    }

    let epsilon = epsilon.clamp(1.0e-6, 0.5 - 1.0e-6);
    let planck = planck_taper_coefficients(len, epsilon);
    let kaiser = kaiser_coefficients(len, beta);
    planck.into_iter().zip(kaiser).map(|(p, k)| p * k).collect()
}

fn planck_taper_coefficients(len: usize, epsilon: f32) -> Vec<f32> {
    if len <= 1 {
        return vec![1.0; len];
    }

    let epsilon = epsilon.clamp(1.0e-6, 0.5 - 1.0e-6);
    let n_max = (len - 1) as f32;
    let half = n_max * 0.5;
    let taper_span = (epsilon * n_max).min(half);

    if taper_span <= 0.0 {
        return vec![1.0; len];
    }

    (0..len)
        .map(|i| {
            let mirrored = if i as f32 <= half {
                i as f32
            } else {
                n_max - i as f32
            };
            planck_taper_value(mirrored, taper_span)
        })
        .collect()
}

fn planck_taper_value(distance: f32, taper_span: f32) -> f32 {
    if distance <= 0.0 {
        0.0
    } else if distance >= taper_span {
        1.0
    } else {
        let denom = taper_span - distance;
        if denom <= f32::EPSILON {
            1.0
        } else {
            let exponent = taper_span / distance - taper_span / denom;
            1.0 / (exponent.exp() + 1.0)
        }
    }
}

fn modified_bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let y = (x / 3.75).powi(2);
        1.0 + y
            * (3.515_622_9
                + y * (3.089_942_4
                    + y * (1.206_749_2
                        + y * (0.265_973_2
                            + y * (0.036_076_8 + y * (0.004_581_3 + y * 0.000_324_11))))))
    } else {
        let y = 3.75 / ax;
        let poly = 0.398_942_28
            + y * (0.013_285_92
                + y * (0.002_253_19
                    + y * (-0.001_575_65
                        + y * (0.009_162_81
                            + y * (-0.020_577_06
                                + y * (0.026_355_37 + y * (-0.016_476_33 + y * 0.003_923_77)))))));
        poly * ax.exp() / ax.sqrt()
    }
}

fn modified_bessel_i1(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let y = x / 3.75;
        let y2 = y * y;
        x * (0.5
            + y2 * (0.878_905_94
                + y2 * (0.514_988_69
                    + y2 * (0.150_849_34
                        + y2 * (0.026_587_33 + y2 * (0.003_015_32 + y2 * 0.000_324_11))))))
    } else {
        let y = 3.75 / ax;
        let poly = 0.398_942_28
            + y * (-0.039_880_24
                + y * (-0.003_620_18
                    + y * (0.001_638_01
                        + y * (-0.010_315_55 + y * (0.022_829_67 - y * 0.028_953_12)))));
        let ans = poly * ax.exp() / ax.sqrt();
        if x < 0.0 { -ans } else { ans }
    }
}

/// Estimate the Kaiser beta parameter for a target stop-band attenuation in dB.
#[allow(dead_code)]
pub fn kaiser_beta_from_attenuation_db(atten_db: f32) -> f32 {
    match atten_db {
        a if a <= 21.0 => 0.0,
        a if a <= 50.0 => {
            let term = a - 21.0;
            0.5842 * term.powf(0.4) + 0.07886 * term
        }
        a => 0.1102 * (a - 8.7),
    }
}

/// Approximate minimum window length that satisfies attenuation and transition width specs.
#[allow(dead_code)]
pub fn kaiser_length_estimate(atten_db: f32, transition_width: f32) -> usize {
    if transition_width <= 0.0 || !transition_width.is_finite() {
        return 1;
    }
    let n = ((atten_db.max(0.0) - 8.0) / (2.285 * transition_width) + 1.0).ceil();
    if !n.is_finite() || n <= 1.0 {
        1
    } else {
        n as usize
    }
}

/// Reassigned sample containing high-resolution time-frequency localization.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct ReassignedSample {
    /// Time offset relative to the column timestamp, in seconds.
    pub time_offset_sec: f32,
    /// Reassigned instantaneous frequency in Hz.
    pub frequency_hz: f32,
    /// Log-power magnitude in dBFS.
    pub magnitude_db: f32,
}

/// One column of log-power magnitudes, with optional reassigned samples.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SpectrogramColumn {
    pub timestamp: Instant,
    pub magnitudes_db: Arc<[f32]>,
    pub reassigned: Option<Arc<[ReassignedSample]>>,
    pub synchro_magnitudes_db: Option<Arc<[f32]>>,
}

/// Incremental update emitted by the spectrogram processor.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SpectrogramUpdate {
    pub fft_size: usize,
    pub hop_size: usize,
    pub sample_rate: f32,
    pub frequency_scale: FrequencyScale,
    pub history_length: usize,
    pub reset: bool,
    pub reassignment_enabled: bool,
    pub synchro_bins_hz: Option<Arc<[f32]>>,
    pub new_columns: Vec<SpectrogramColumn>,
}

#[derive(Debug)]
struct SpectrogramHistory {
    slots: VecDeque<SpectrogramColumn>,
    capacity: usize,
}

impl SpectrogramHistory {
    fn new(capacity: usize) -> Self {
        Self {
            slots: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    fn set_capacity(&mut self, capacity: usize, evicted: &mut Vec<SpectrogramColumn>) {
        evicted.clear();
        if capacity == self.capacity {
            return;
        }

        self.capacity = capacity;
        if capacity == 0 {
            evicted.extend(self.slots.drain(..));
        } else {
            while self.slots.len() > capacity {
                if let Some(column) = self.slots.pop_front() {
                    evicted.push(column);
                }
            }
        }
    }

    fn clear_into(&mut self, out: &mut Vec<SpectrogramColumn>) {
        out.clear();
        out.extend(self.slots.drain(..));
    }

    fn push(&mut self, column: SpectrogramColumn) -> Option<SpectrogramColumn> {
        if self.capacity == 0 {
            return Some(column);
        }

        let evicted = if self.slots.len() == self.capacity {
            self.slots.pop_front()
        } else {
            None
        };
        self.slots.push_back(column);
        evicted
    }
}

#[derive(Default)]
struct SynchroState {
    enabled: bool,
    scale: FrequencyScale,
    bin_frequencies: Arc<[f32]>,
    power_buffer: Vec<f32>,
    magnitude_buffer: Vec<f32>,
    temporal_buffer: Vec<f32>,
    frequency_buffer: Vec<f32>,
    min_hz: f32,
    max_hz: f32,
    log_min: f64,
    log_range: f64,
    mel_min: f64,
    mel_range: f64,
}

impl SynchroState {
    fn new(config: &SpectrogramConfig, fft_size: usize, sample_rate: f32) -> Self {
        let mut state = Self::default();
        state.reconfigure(config, fft_size, sample_rate);
        state
    }

    fn reconfigure(&mut self, config: &SpectrogramConfig, fft_size: usize, sample_rate: f32) {
        *self = Self::default();

        if !config.use_synchrosqueezing
            || !config.use_reassignment
            || config.synchrosqueezing_bin_count == 0
            || sample_rate <= 0.0
            || fft_size == 0
        {
            return;
        }

        let scale = config.frequency_scale;
        let min_hz = config
            .synchrosqueezing_min_hz
            .max(1.0)
            .min(sample_rate * 0.5);
        let nyquist = (sample_rate * 0.5).max(min_hz * 1.001);
        let bin_count = config.synchrosqueezing_bin_count.max(2);

        let (log_min, log_range) = {
            let min = (min_hz as f64).ln();
            let max = (nyquist as f64).ln();
            (min, (max - min).max(1.0e-9))
        };

        let (mel_min, mel_range) = {
            let min = hz_to_mel(min_hz) as f64;
            let max = hz_to_mel(nyquist) as f64;
            (min, (max - min).max(1.0e-9))
        };

        let freqs = if bin_count == 1 {
            vec![min_hz]
        } else {
            let mut freqs = Vec::with_capacity(bin_count);
            for idx in 0..bin_count {
                let t = idx as f64 / (bin_count as f64 - 1.0);
                let freq = match scale {
                    FrequencyScale::Linear => {
                        (min_hz as f64 + (nyquist as f64 - min_hz as f64) * t) as f32
                    }
                    FrequencyScale::Logarithmic => ((log_min + log_range * t).exp()) as f32,
                    FrequencyScale::Mel => mel_to_hz((mel_min + mel_range * t) as f32),
                };
                freqs.push(freq);
            }
            freqs.reverse();
            freqs
        };

        self.enabled = true;
        self.scale = scale;
        self.bin_frequencies = Arc::from(freqs.into_boxed_slice());
        self.power_buffer = vec![0.0; bin_count];
        self.magnitude_buffer = vec![DB_FLOOR; bin_count];
        self.temporal_buffer = vec![DB_FLOOR; bin_count];
        self.frequency_buffer.clear();
        self.min_hz = min_hz;
        self.max_hz = nyquist;
        self.log_min = log_min;
        self.log_range = log_range;
        self.mel_min = mel_min;
        self.mel_range = mel_range;
    }

    fn is_active(&self) -> bool {
        self.enabled && !self.power_buffer.is_empty() && self.log_range > 0.0
    }

    fn bins_arc(&self) -> Option<Arc<[f32]>> {
        if self.enabled && !self.bin_frequencies.is_empty() {
            Some(Arc::clone(&self.bin_frequencies))
        } else {
            None
        }
    }

    fn reset_power(&mut self) {
        self.power_buffer.fill(0.0);
    }

    fn accumulate(&mut self, freq_hz: f64, display_power: f32) {
        if !self.is_active()
            || !freq_hz.is_finite()
            || freq_hz < self.min_hz as f64
            || freq_hz > self.max_hz as f64
        {
            return;
        }

        let normalized = match self.scale {
            FrequencyScale::Linear => {
                (freq_hz - self.min_hz as f64) / (self.max_hz - self.min_hz) as f64
            }
            FrequencyScale::Logarithmic => (freq_hz.ln() - self.log_min) / self.log_range,
            FrequencyScale::Mel => {
                (hz_to_mel(freq_hz as f32) as f64 - self.mel_min) / self.mel_range
            }
        }
        .clamp(0.0, 1.0);

        let bin_count = self.power_buffer.len();
        if bin_count == 1 {
            self.power_buffer[0] += display_power;
            return;
        }

        let position = (1.0 - normalized) * (bin_count as f64 - 1.0);
        let lower = position.floor() as usize;
        let upper = position.ceil() as usize;
        let frac = (position - lower as f64) as f32;

        self.power_buffer[lower] += display_power * (1.0 - frac);
        if upper != lower {
            self.power_buffer[upper] += display_power * frac;
        }
    }

    fn finalize_magnitudes(&mut self) {
        if !self.enabled {
            return;
        }
        self.magnitude_buffer
            .iter_mut()
            .zip(&self.power_buffer)
            .for_each(|(magnitude, &power)| {
                *magnitude = (power.max(POWER_EPSILON).ln() * LOG_FACTOR).max(DB_FLOOR);
            });
    }

    fn apply_temporal_smoothing(&mut self, smoothing: f32, max_hz: f32, blend_hz: f32) {
        if !self.enabled || self.magnitude_buffer.is_empty() {
            self.temporal_buffer.clear();
            return;
        }

        apply_temporal_smoothing_with_weights(
            &mut self.magnitude_buffer,
            &mut self.temporal_buffer,
            smoothing,
            |idx| smoothing_weight(self.bin_frequencies[idx], max_hz, blend_hz),
        );
    }

    fn magnitudes(&self) -> &[f32] {
        &self.magnitude_buffer
    }

    fn reset_temporal(&mut self) {
        self.temporal_buffer.clear();
        if self.enabled {
            self.temporal_buffer
                .resize(self.magnitude_buffer.len(), DB_FLOOR);
        }
    }

    fn reset_frequency(&mut self) {
        self.frequency_buffer.clear();
    }

    fn align_pool(&self, pool: &mut Vec<Arc<[f32]>>) {
        pool.retain(|buffer| buffer.len() == self.magnitude_buffer.len());
    }

    fn apply_frequency_smoothing(&mut self, radius: usize, max_hz: f32, blend_hz: f32) {
        if !self.enabled || self.magnitude_buffer.is_empty() {
            self.reset_frequency();
            return;
        }

        if radius == 0 {
            self.reset_frequency();
            return;
        }

        let bins = self.magnitude_buffer.len();
        let radius = radius.min(bins.saturating_sub(1));
        if radius == 0 {
            self.reset_frequency();
            return;
        }

        if self.frequency_buffer.len() != bins {
            self.frequency_buffer.resize(bins, 0.0);
        }

        for idx in 0..bins {
            let start = idx.saturating_sub(radius);
            let end = (idx + radius + 1).min(bins);
            let mut sum = 0.0;
            for value in &self.magnitude_buffer[start..end] {
                sum += *value;
            }
            self.frequency_buffer[idx] = sum / (end - start) as f32;
        }

        for (idx, value) in self.magnitude_buffer.iter_mut().enumerate() {
            let smoothed = self.frequency_buffer[idx];
            let weight = smoothing_weight(
                self.bin_frequencies.get(idx).copied().unwrap_or_default(),
                max_hz,
                blend_hz,
            );
            *value = smoothed * weight + *value * (1.0 - weight);
        }
    }
}

pub struct SpectrogramProcessor {
    stored_config: SpectrogramConfig,
    config: SpectrogramConfig,
    planner: RealFftPlanner<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
    window_size: usize,
    fft_size: usize,
    window: Arc<[f32]>,
    time_weight_window: Vec<f32>,
    derivative_window: Vec<f32>,
    real_buffer: Vec<f32>,
    time_weight_buffer: Vec<f32>,
    derivative_buffer: Vec<f32>,
    spectrum_buffer: Vec<Complex32>,
    time_spectrum_buffer: Vec<Complex32>,
    derivative_spectrum_buffer: Vec<Complex32>,
    scratch_buffer: Vec<Complex32>,
    magnitude_buffer: Vec<f32>,
    reassigned_power_buffer: Vec<f32>,
    synchro: SynchroState,
    bin_normalization: Vec<f32>,
    energy_normalization: Vec<f32>,
    pcm_buffer: VecDeque<f32>,
    buffer_start_index: u64,
    start_instant: Option<Instant>,
    accumulated_offset: Duration,
    history: SpectrogramHistory,
    magnitude_pool: Vec<Arc<[f32]>>,
    synchro_pool: Vec<Arc<[f32]>>,
    evicted_columns: Vec<SpectrogramColumn>,
    reassignment_power_floor_linear: f32,
    temporal_smoothing_buffer: Vec<f32>,
    frequency_scratch_buffer: Vec<f32>,
    pending_reset: bool,
}

impl SpectrogramProcessor {
    pub fn new(config: SpectrogramConfig) -> Self {
        let stored_config = config;
        let mut runtime_config = config;
        Self::normalize_config(&mut runtime_config);

        let window_size = runtime_config.fft_size;
        let zero_padding = runtime_config.zero_padding_factor.max(1);
        let fft_size = window_size.saturating_mul(zero_padding);
        assert!(fft_size > 0, "FFT size must be greater than zero");

        let history_len = runtime_config.history_length;
        let bins = fft_size / 2 + 1;

        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(fft_size);
        let window = WindowCache::global().get(runtime_config.window, window_size);
        let time_weight_window = compute_time_weight_window(window.as_ref());
        let derivative_window = compute_derivative_window(runtime_config.window, window.as_ref());

        let real_buffer = vec![0.0; fft_size];
        let time_weight_buffer = vec![0.0; fft_size];
        let derivative_buffer = vec![0.0; fft_size];
        let spectrum_buffer = fft.make_output_vec();
        let time_spectrum_buffer = fft.make_output_vec();
        let derivative_spectrum_buffer = fft.make_output_vec();
        let scratch_buffer = fft.make_scratch_vec();
        let magnitude_buffer = vec![0.0; bins];
        let reassigned_power_buffer = vec![0.0; bins];
        let bin_normalization = Self::compute_bin_normalization(window.as_ref(), fft_size);
        let energy_normalization = Self::compute_energy_normalization(window.as_ref(), fft_size);
        let pcm_buffer = VecDeque::with_capacity(window_size.saturating_mul(2));
        let history = SpectrogramHistory::new(history_len);
        let reassignment_power_floor_linear =
            power_db_to_linear(runtime_config.reassignment_power_floor_db);

        let synchro = SynchroState::new(&runtime_config, fft_size, runtime_config.sample_rate);
        let temporal_smoothing_buffer = vec![DB_FLOOR; bins];
        let frequency_scratch_buffer = vec![0.0; bins];

        Self {
            stored_config,
            config: runtime_config,
            planner,
            fft,
            window_size,
            fft_size,
            window,
            time_weight_window,
            derivative_window,
            real_buffer,
            time_weight_buffer,
            derivative_buffer,
            spectrum_buffer,
            time_spectrum_buffer,
            derivative_spectrum_buffer,
            scratch_buffer,
            magnitude_buffer,
            reassigned_power_buffer,
            synchro,
            bin_normalization,
            energy_normalization,
            pcm_buffer,
            buffer_start_index: 0,
            start_instant: None,
            accumulated_offset: Duration::default(),
            history,
            magnitude_pool: Vec::new(),
            synchro_pool: Vec::new(),
            evicted_columns: Vec::new(),
            reassignment_power_floor_linear,
            temporal_smoothing_buffer,
            frequency_scratch_buffer,
            pending_reset: true,
        }
    }

    fn normalize_config(config: &mut SpectrogramConfig) {
        if !config.use_reassignment {
            config.use_synchrosqueezing = false;
        }

        if !config.use_synchrosqueezing {
            config.temporal_smoothing = 0.0;
            config.temporal_smoothing_max_hz = 0.0;
            config.temporal_smoothing_blend_hz = 0.0;
            config.frequency_smoothing_radius = 0;
            config.frequency_smoothing_max_hz = 0.0;
            config.frequency_smoothing_blend_hz = 0.0;
        }
    }

    pub fn config(&self) -> SpectrogramConfig {
        self.stored_config
    }

    fn rebuild_fft(&mut self) {
        let window_size = self.config.fft_size;
        let zero_padding = self.config.zero_padding_factor.max(1);
        let fft_size = window_size.saturating_mul(zero_padding);
        assert!(fft_size > 0, "FFT size must be greater than zero");

        self.window_size = window_size;
        self.fft_size = fft_size;

        let bins = fft_size / 2 + 1;

        self.fft = self.planner.plan_fft_forward(fft_size);
        self.window = WindowCache::global().get(self.config.window, window_size);
        self.time_weight_window = compute_time_weight_window(self.window.as_ref());
        self.derivative_window =
            compute_derivative_window(self.config.window, self.window.as_ref());

        self.real_buffer.resize(fft_size, 0.0);
        self.time_weight_buffer.resize(fft_size, 0.0);
        self.derivative_buffer.resize(fft_size, 0.0);

        self.spectrum_buffer = self.fft.make_output_vec();
        self.time_spectrum_buffer = self.fft.make_output_vec();
        self.derivative_spectrum_buffer = self.fft.make_output_vec();
        self.scratch_buffer = self.fft.make_scratch_vec();

        self.magnitude_buffer.resize(bins, 0.0);
        self.reassigned_power_buffer.resize(bins, 0.0);
        self.bin_normalization = Self::compute_bin_normalization(self.window.as_ref(), fft_size);
        self.energy_normalization =
            Self::compute_energy_normalization(self.window.as_ref(), fft_size);

        self.temporal_smoothing_buffer.resize(bins, DB_FLOOR);
        self.frequency_scratch_buffer.resize(bins, 0.0);

        self.synchro
            .reconfigure(&self.config, fft_size, self.config.sample_rate);
        self.synchro.align_pool(&mut self.synchro_pool);
        self.synchro.reset_temporal();
        self.synchro.reset_power();
        self.synchro.reset_frequency();

        self.reassignment_power_floor_linear =
            power_db_to_linear(self.config.reassignment_power_floor_db);
        self.magnitude_pool.retain(|buffer| buffer.len() == bins);
        resize_vecdeque(&mut self.pcm_buffer, window_size.saturating_mul(2).max(1));

        // Clear history and recycle columns
        let mut evicted = std::mem::take(&mut self.evicted_columns);
        self.history.clear_into(&mut evicted);
        evicted
            .drain(..)
            .for_each(|column| self.recycle_column(column));

        self.history
            .set_capacity(self.config.history_length, &mut evicted);
        evicted
            .drain(..)
            .for_each(|column| self.recycle_column(column));
        self.evicted_columns = evicted;

        self.pcm_buffer.clear();
        self.buffer_start_index = 0;
        self.start_instant = None;
        self.accumulated_offset = Duration::default();
        self.pending_reset = true;
    }

    fn ensure_fft_capacity(&mut self) {
        let expected_window = self.config.fft_size;
        let expected_fft = expected_window.saturating_mul(self.config.zero_padding_factor.max(1));
        if self.window_size != expected_window
            || self.fft_size != expected_fft
            || self.real_buffer.len() != expected_fft
            || self.time_weight_window.len() != expected_window
            || self.derivative_window.len() != expected_window
            || self.spectrum_buffer.len() != expected_fft / 2 + 1
            || self.bin_normalization.len() != expected_fft / 2 + 1
            || self.energy_normalization.len() != expected_fft / 2 + 1
        {
            self.rebuild_fft();
        }
    }

    fn process_ready_windows(&mut self) -> Vec<SpectrogramColumn> {
        let mut new_columns = Vec::new();
        let window_size = self.window_size;
        let fft_size = self.fft_size;
        let hop = self.config.hop_size;
        if window_size == 0 || fft_size == 0 || hop == 0 {
            return new_columns;
        }

        let sample_rate = self.config.sample_rate;
        let hop_duration = if sample_rate > 0.0 {
            duration_from_samples(hop as u64, sample_rate)
        } else {
            Duration::default()
        };
        let bins = fft_size / 2 + 1;
        let reassignment_enabled = self.config.use_reassignment && sample_rate > f32::EPSILON;
        let reassignment_bin_limit = if self.config.reassignment_low_bin_limit == 0 {
            bins
        } else {
            self.config.reassignment_low_bin_limit.min(bins)
        };
        let synchro_active = reassignment_enabled && self.synchro.is_active();

        let Some(start_instant) = self.start_instant else {
            return new_columns;
        };
        let center_offset = duration_from_samples((window_size / 2) as u64, sample_rate);

        while self.pcm_buffer.len() >= window_size {
            {
                let (front, back) = self.pcm_buffer.as_slices();
                let window_input = &mut self.real_buffer[..window_size];
                if window_size <= front.len() {
                    window_input.copy_from_slice(&front[..window_size]);
                } else {
                    window_input[..front.len()].copy_from_slice(front);
                    let tail = window_size - front.len();
                    window_input[front.len()..window_size].copy_from_slice(&back[..tail]);
                }
                Self::remove_dc(window_input);

                if reassignment_enabled {
                    for (idx, &raw) in window_input.iter().enumerate() {
                        self.time_weight_buffer[idx] = raw * self.time_weight_window[idx];
                        self.derivative_buffer[idx] = raw * self.derivative_window[idx];
                    }
                    self.time_weight_buffer[window_size..fft_size].fill(0.0);
                    self.derivative_buffer[window_size..fft_size].fill(0.0);
                }

                Self::apply_window(window_input, self.window.as_ref());
            }

            if fft_size > window_size {
                self.real_buffer[window_size..fft_size].fill(0.0);
            }

            self.fft
                .process_with_scratch(
                    &mut self.real_buffer,
                    &mut self.spectrum_buffer,
                    &mut self.scratch_buffer,
                )
                .expect("real FFT forward transform");

            if reassignment_enabled {
                self.fft
                    .process_with_scratch(
                        &mut self.time_weight_buffer,
                        &mut self.time_spectrum_buffer,
                        &mut self.scratch_buffer,
                    )
                    .expect("time-weighted FFT");
                self.fft
                    .process_with_scratch(
                        &mut self.derivative_buffer,
                        &mut self.derivative_spectrum_buffer,
                        &mut self.scratch_buffer,
                    )
                    .expect("derivative-window FFT");
            }

            self.spectrum_buffer
                .iter()
                .zip(&self.bin_normalization)
                .zip(&mut self.magnitude_buffer)
                .for_each(|((complex, norm), target)| {
                    let power = (complex.norm_sqr() * *norm).max(POWER_EPSILON);
                    *target = (power.ln() * LOG_FACTOR).max(DB_FLOOR);
                });

            let timestamp = start_instant + self.accumulated_offset + center_offset;

            let reassigned = if reassignment_enabled {
                self.compute_reassigned_samples(
                    sample_rate,
                    fft_size,
                    reassignment_bin_limit,
                    synchro_active,
                )
            } else {
                None
            };

            self.apply_magnitude_post_processing(bins);
            if synchro_active {
                self.synchro.apply_frequency_smoothing(
                    self.config.frequency_smoothing_radius,
                    self.config.frequency_smoothing_max_hz,
                    self.config.frequency_smoothing_blend_hz,
                );
                self.synchro.apply_temporal_smoothing(
                    self.config.temporal_smoothing,
                    self.config.temporal_smoothing_max_hz,
                    self.config.temporal_smoothing_blend_hz,
                );
            }

            let magnitudes = Self::fill_arc(
                self.acquire_magnitude_storage(bins),
                &self.magnitude_buffer[..bins],
            );

            let synchro_magnitudes = if synchro_active {
                let len = self.synchro.magnitudes().len();
                if len == 0 {
                    None
                } else {
                    let storage = self.acquire_synchro_storage(len);
                    Some(Self::fill_arc(storage, self.synchro.magnitudes()))
                }
            } else {
                None
            };

            let history_column = SpectrogramColumn {
                timestamp,
                magnitudes_db: Arc::clone(&magnitudes),
                reassigned: reassigned.clone(),
                synchro_magnitudes_db: synchro_magnitudes.clone(),
            };
            if let Some(evicted) = self.history.push(history_column) {
                self.recycle_column(evicted);
            }

            new_columns.push(SpectrogramColumn {
                timestamp,
                magnitudes_db: magnitudes,
                reassigned,
                synchro_magnitudes_db: synchro_magnitudes,
            });

            let discard = hop.min(self.pcm_buffer.len());
            self.pcm_buffer.drain(..discard);
            self.buffer_start_index += hop as u64;
            self.accumulated_offset += hop_duration;
        }

        new_columns
    }

    fn compute_reassigned_samples(
        &mut self,
        sample_rate: f32,
        fft_size: usize,
        reassignment_bin_limit: usize,
        synchro_active: bool,
    ) -> Option<Arc<[ReassignedSample]>> {
        let mut samples = Vec::with_capacity(reassignment_bin_limit);
        let power_floor = self.reassignment_power_floor_linear;
        let sample_rate_f64 = sample_rate as f64;
        let fft_size_f64 = fft_size as f64;
        let nyquist = sample_rate_f64 * 0.5;
        let bin_hz = if fft_size > 0 {
            sample_rate / fft_size as f32
        } else {
            0.0
        };

        self.reassigned_power_buffer.fill(0.0);
        if synchro_active {
            self.synchro.reset_power();
        }

        for k in 0..reassignment_bin_limit {
            let base = self.spectrum_buffer[k];
            let power = base.norm_sqr();
            if power <= POWER_EPSILON {
                continue;
            }

            let display_power = power * self.bin_normalization[k];
            if display_power < power_floor {
                continue;
            }

            let energy_scale = self.energy_normalization[k];
            if energy_scale <= 0.0 {
                continue;
            }
            let energy_power = power * energy_scale;

            let base_conj = base.conj();
            let cross_time = self.time_spectrum_buffer[k] * base_conj;
            let cross_freq = self.derivative_spectrum_buffer[k] * base_conj;

            let denom = f64::from(power.max(POWER_EPSILON));

            let mut delta_n = -f64::from(cross_time.re) / denom;
            if !delta_n.is_finite() {
                continue;
            }
            delta_n = delta_n.clamp(-(self.window_size as f64), self.window_size as f64);

            let mut delta_omega = -f64::from(cross_freq.im) / denom;
            if !delta_omega.is_finite() {
                continue;
            }
            delta_omega = delta_omega.clamp(-std::f64::consts::PI, std::f64::consts::PI);

            if sample_rate_f64 <= f64::EPSILON {
                continue;
            }

            let freq_base = (k as f64) * sample_rate_f64 / fft_size_f64;
            let freq_offset = delta_omega * sample_rate_f64 / (2.0 * std::f64::consts::PI);
            let freq_hz = freq_base + freq_offset;
            if !(freq_hz.is_finite() && freq_hz >= 0.0 && freq_hz <= nyquist) {
                continue;
            }

            let magnitude_db = if display_power > 0.0 {
                (display_power.max(POWER_EPSILON).ln() * LOG_FACTOR).max(DB_FLOOR)
            } else {
                DB_FLOOR
            };

            if bin_hz > 0.0 {
                let max_index = (reassignment_bin_limit.saturating_sub(1)) as f32;
                let freq_position = (freq_hz as f32 / bin_hz).clamp(0.0, max_index);
                let lower = freq_position.floor() as usize;
                let upper = freq_position.ceil() as usize;
                let frac = freq_position - lower as f32;
                let lower_weight = 1.0 - frac;
                self.reassigned_power_buffer[lower] += energy_power * lower_weight;
                if upper != lower && upper < reassignment_bin_limit {
                    self.reassigned_power_buffer[upper] += energy_power * frac;
                }
            }

            if synchro_active {
                self.synchro.accumulate(freq_hz, display_power);
            }

            let time_offset_sec = (delta_n / sample_rate_f64) as f32;
            samples.push(ReassignedSample {
                time_offset_sec,
                frequency_hz: freq_hz as f32,
                magnitude_db,
            });
        }

        if reassignment_bin_limit > 0 {
            for idx in 0..reassignment_bin_limit {
                let energy_power = self.reassigned_power_buffer[idx];
                let energy_scale = self.energy_normalization[idx];
                let display_scale = self.bin_normalization[idx];
                let display_power = if energy_power > 0.0 && energy_scale > 0.0 {
                    energy_power * (display_scale / energy_scale.max(f32::EPSILON))
                } else {
                    0.0
                };
                let power = display_power.max(POWER_EPSILON);
                self.magnitude_buffer[idx] = (power.ln() * LOG_FACTOR).max(DB_FLOOR);
            }
        }

        if synchro_active {
            self.synchro.finalize_magnitudes();
        }

        if samples.is_empty() {
            None
        } else {
            Some(Arc::from(samples.into_boxed_slice()))
        }
    }

    fn apply_magnitude_post_processing(&mut self, bins: usize) {
        if bins == 0 {
            return;
        }

        let bin_hz = if self.config.sample_rate > 0.0 && self.fft_size > 0 {
            Some(self.config.sample_rate / self.fft_size as f32)
        } else {
            None
        };

        let max_hz = self.config.temporal_smoothing_max_hz;
        let blend_hz = self.config.temporal_smoothing_blend_hz;
        apply_temporal_smoothing_with_weights(
            &mut self.magnitude_buffer[..bins],
            &mut self.temporal_smoothing_buffer,
            self.config.temporal_smoothing,
            |idx| {
                bin_hz
                    .map(|hz| smoothing_weight(idx as f32 * hz, max_hz, blend_hz))
                    .unwrap_or(1.0)
            },
        );

        if self.config.frequency_smoothing_radius > 0 {
            let radius = self
                .config
                .frequency_smoothing_radius
                .min(bins.saturating_sub(1));
            if radius > 0 {
                if self.frequency_scratch_buffer.len() != bins {
                    self.frequency_scratch_buffer.resize(bins, 0.0);
                }

                for idx in 0..bins {
                    let start = idx.saturating_sub(radius);
                    let end = (idx + radius + 1).min(bins);
                    let mut sum = 0.0;
                    for value in &self.magnitude_buffer[start..end] {
                        sum += *value;
                    }
                    self.frequency_scratch_buffer[idx] = sum / (end - start) as f32;
                }

                let max_hz = self.config.frequency_smoothing_max_hz;
                let blend_hz = self.config.frequency_smoothing_blend_hz;

                for idx in 0..bins {
                    let original = self.magnitude_buffer[idx];
                    let smoothed = self.frequency_scratch_buffer[idx];
                    let weight = bin_hz
                        .map(|hz| smoothing_weight(idx as f32 * hz, max_hz, blend_hz))
                        .unwrap_or(1.0);
                    self.magnitude_buffer[idx] = smoothed * weight + original * (1.0 - weight);
                }
            }
        }
    }
}

impl AudioProcessor for SpectrogramProcessor {
    type Output = SpectrogramUpdate;

    fn process_block(&mut self, block: &AudioBlock<'_>) -> ProcessorUpdate<Self::Output> {
        if block.frame_count() == 0 || block.channels == 0 {
            return ProcessorUpdate::None;
        }

        if self.config.sample_rate <= 0.0 {
            self.config.sample_rate = block.sample_rate;
        } else if (self.config.sample_rate - block.sample_rate).abs() > f32::EPSILON {
            let duration_elapsed =
                duration_from_samples(self.buffer_start_index, self.config.sample_rate);
            let previous_start = self.start_instant;
            self.config.sample_rate = block.sample_rate;
            self.rebuild_fft();
            self.start_instant = previous_start;
            self.accumulated_offset = duration_elapsed;
        }

        if self.start_instant.is_none() {
            self.start_instant = Some(block.timestamp);
        }

        self.ensure_fft_capacity();

        self.mixdown_interleaved(block.samples, block.channels);

        let new_columns = self.process_ready_windows();
        if new_columns.is_empty() {
            ProcessorUpdate::None
        } else {
            let reset = std::mem::take(&mut self.pending_reset);
            ProcessorUpdate::Snapshot(SpectrogramUpdate {
                fft_size: self.fft_size,
                hop_size: self.config.hop_size,
                sample_rate: self.config.sample_rate,
                frequency_scale: self.config.frequency_scale,
                history_length: self.config.history_length,
                reset,
                reassignment_enabled: self.config.use_reassignment,
                synchro_bins_hz: self.synchro.bins_arc(),
                new_columns,
            })
        }
    }

    fn reset(&mut self) {
        let mut evicted = std::mem::take(&mut self.evicted_columns);
        self.history.clear_into(&mut evicted);
        evicted
            .drain(..)
            .for_each(|column| self.recycle_column(column));
        self.evicted_columns = evicted;

        resize_vecdeque(
            &mut self.pcm_buffer,
            self.window_size.saturating_mul(2).max(1),
        );
        self.pcm_buffer.clear();
        self.buffer_start_index = 0;
        self.start_instant = None;
        self.pending_reset = true;

        self.temporal_smoothing_buffer.fill(DB_FLOOR);
        self.frequency_scratch_buffer
            .resize(self.magnitude_buffer.len(), 0.0);

        self.synchro.reset_temporal();
        self.synchro.reset_power();
        self.synchro.reset_frequency();
    }
}

impl SpectrogramProcessor {
    #[inline]
    fn mixdown_interleaved(&mut self, samples: &[f32], channels: usize) {
        if channels == 1 {
            self.pcm_buffer.extend(samples);
            return;
        }

        let scale = 1.0 / channels as f32;
        let mut chunks = samples.chunks_exact(channels);
        for chunk in &mut chunks {
            self.pcm_buffer.push_back(chunk.iter().sum::<f32>() * scale);
        }
        let remainder = chunks.remainder();
        if !remainder.is_empty() {
            self.pcm_buffer
                .push_back(remainder.iter().sum::<f32>() * scale);
        }
    }

    #[inline(always)]
    fn apply_window(buffer: &mut [f32], window: &[f32]) {
        debug_assert_eq!(buffer.len(), window.len());
        buffer.iter_mut().zip(window).for_each(|(s, &w)| *s *= w);
    }

    #[inline]
    fn remove_dc(buffer: &mut [f32]) {
        if buffer.is_empty() {
            return;
        }
        let mean = buffer.iter().sum::<f32>() / buffer.len() as f32;
        if mean.abs() > f32::EPSILON {
            buffer.iter_mut().for_each(|s| *s -= mean);
        }
    }

    fn compute_bin_normalization(window: &[f32], fft_size: usize) -> Vec<f32> {
        let window_sum: f32 = window.iter().sum();
        let inv = 1.0 / window_sum;
        Self::compute_normalization(fft_size, window_sum, inv * inv, 4.0 * inv * inv)
    }

    fn compute_energy_normalization(window: &[f32], fft_size: usize) -> Vec<f32> {
        let energy: f32 = window.iter().map(|&c| c * c).sum();
        Self::compute_normalization(fft_size, energy, 1.0 / energy, 2.0 / energy)
    }

    fn compute_normalization(
        fft_size: usize,
        metric: f32,
        dc_scale: f32,
        interior_scale: f32,
    ) -> Vec<f32> {
        let bins = fft_size / 2 + 1;
        if bins == 0 || !metric.is_finite() || metric <= f32::EPSILON {
            return vec![0.0; bins];
        }

        let mut norms = vec![interior_scale; bins];
        norms[0] = dc_scale;
        if bins > 1 {
            norms[bins - 1] = dc_scale;
        }
        norms
    }

    fn acquire_magnitude_storage(&mut self, bins: usize) -> Arc<[f32]> {
        acquire_from_pool(&mut self.magnitude_pool, bins, 0.0)
    }

    fn acquire_synchro_storage(&mut self, bins: usize) -> Arc<[f32]> {
        acquire_from_pool(&mut self.synchro_pool, bins, DB_FLOOR)
    }

    fn fill_arc(mut storage: Arc<[f32]>, data: &[f32]) -> Arc<[f32]> {
        if storage.len() != data.len() {
            return Arc::<[f32]>::from(data.to_vec());
        }
        if let Some(buffer) = Arc::get_mut(&mut storage) {
            buffer.copy_from_slice(data);
            storage
        } else {
            Arc::<[f32]>::from(data.to_vec())
        }
    }

    fn recycle_column(&mut self, column: SpectrogramColumn) {
        if Arc::strong_count(&column.magnitudes_db) == 1 {
            self.magnitude_pool.push(column.magnitudes_db);
        }
        if let Some(synchro) = column.synchro_magnitudes_db
            && Arc::strong_count(&synchro) == 1
        {
            self.synchro_pool.push(synchro);
        }
    }

    fn clear_history(&mut self) {
        let mut evicted = std::mem::take(&mut self.evicted_columns);
        self.history.clear_into(&mut evicted);
        evicted
            .drain(..)
            .for_each(|column| self.recycle_column(column));
        self.evicted_columns = evicted;
        self.pending_reset = true;
    }
}

fn acquire_from_pool(pool: &mut Vec<Arc<[f32]>>, bins: usize, default: f32) -> Arc<[f32]> {
    if bins == 0 {
        return Arc::from([]);
    }
    pool.iter()
        .rposition(|buffer| buffer.len() == bins)
        .map(|idx| pool.swap_remove(idx))
        .unwrap_or_else(|| Arc::from(vec![default; bins]))
}

fn compute_time_weight_window(window: &[f32]) -> Vec<f32> {
    if window.is_empty() {
        return Vec::new();
    }
    let center = (window.len() - 1) as f32 * 0.5;
    window
        .iter()
        .enumerate()
        .map(|(idx, &c)| (idx as f32 - center) * c)
        .collect()
}

fn compute_derivative_window(kind: WindowKind, window: &[f32]) -> Vec<f32> {
    match kind {
        WindowKind::PlanckBessel { epsilon, beta } => {
            compute_planck_bessel_derivative(window, epsilon, beta)
        }
        _ => compute_numeric_derivative(window),
    }
}

fn compute_numeric_derivative(window: &[f32]) -> Vec<f32> {
    if window.len() <= 1 {
        return vec![0.0; window.len()];
    }

    let len = window.len();
    (0..len)
        .map(|i| {
            let prev = if i == 0 { window[0] } else { window[i - 1] };
            let next = window.get(i + 1).copied().unwrap_or(window[len - 1]);
            0.5 * (next - prev)
        })
        .collect()
}

fn compute_planck_bessel_derivative(window: &[f32], epsilon: f32, beta: f32) -> Vec<f32> {
    if window.len() <= 1 {
        return vec![0.0; window.len()];
    }

    let len = window.len();
    let epsilon = epsilon.clamp(1.0e-6, 0.5 - 1.0e-6);
    let n_max = (len - 1) as f32;
    let half = n_max * 0.5;
    let taper_span = (epsilon * n_max).min(half);

    if taper_span <= 0.0 {
        return compute_numeric_derivative(window);
    }

    let denom = modified_bessel_i0(beta as f64);
    let beta_f64 = beta.max(0.0) as f64;
    let span = (len - 1) as f64;

    (0..len)
        .map(|idx| {
            let position = idx as f32;
            let (mirrored, sign) = if position < half {
                (position, 1.0)
            } else if position > half {
                (n_max - position, -1.0)
            } else {
                (position, 0.0)
            };

            // Compute kaiser value and derivative inline
            let (kaiser_value, kaiser_derivative) = if beta_f64 == 0.0 {
                (1.0, 0.0)
            } else {
                let n = idx as f64;
                let ratio = (2.0 * n) / span - 1.0;
                let inside = (1.0 - ratio * ratio).max(0.0).sqrt();
                let argument = beta_f64 * inside;
                let value = (modified_bessel_i0(argument) / denom) as f32;
                let derivative = if inside <= f64::EPSILON {
                    0.0
                } else {
                    (modified_bessel_i1(argument) * beta_f64 * (-(2.0 * ratio) / (span * inside))
                        / denom) as f32
                };
                (value, derivative)
            };

            let planck_value =
                if beta.abs() > f32::EPSILON && kaiser_value.abs() > f32::MIN_POSITIVE {
                    window[idx] / kaiser_value
                } else {
                    planck_taper_value(mirrored, taper_span)
                };
            let planck_derivative = planck_taper_derivative(mirrored, taper_span) * sign;

            planck_derivative * kaiser_value + planck_value * kaiser_derivative
        })
        .collect()
}

fn planck_taper_derivative(distance: f32, taper_span: f32) -> f32 {
    if distance <= 0.0 || taper_span <= 0.0 || distance >= taper_span {
        return 0.0;
    }

    let s = taper_span as f64;
    let d = distance as f64;
    let denom = s - d;
    if denom <= f64::EPSILON {
        return 0.0;
    }

    let exponent = s / d - s / denom;
    let exp_e = exponent.exp();
    let logistic = 1.0 / (exp_e + 1.0);
    let gradient = s / (d * d) + s / (denom * denom);
    (logistic * (1.0 - logistic) * gradient) as f32
}

fn power_db_to_linear(db: f32) -> f32 {
    10.0f32.powf(db / 10.0).min(f32::MAX)
}

fn smoothing_weight(freq_hz: f32, max_hz: f32, blend_hz: f32) -> f32 {
    if max_hz <= 0.0 || freq_hz <= max_hz {
        1.0
    } else if blend_hz <= f32::EPSILON || freq_hz >= max_hz + blend_hz {
        0.0
    } else {
        (1.0 - (freq_hz - max_hz) / blend_hz).clamp(0.0, 1.0)
    }
}

fn apply_temporal_smoothing_with_weights(
    buffer: &mut [f32],
    history: &mut Vec<f32>,
    smoothing: f32,
    mut weight_for_bin: impl FnMut(usize) -> f32,
) {
    if buffer.is_empty() {
        history.clear();
        return;
    }

    let smoothing = smoothing.clamp(0.0, 0.999);
    if history.len() != buffer.len() {
        history.resize(buffer.len(), DB_FLOOR);
    }

    if smoothing <= 0.0 {
        history.copy_from_slice(buffer);
        return;
    }

    for (idx, value) in buffer.iter_mut().enumerate() {
        let alpha = (smoothing * weight_for_bin(idx)).clamp(0.0, 0.999);
        let prev = history[idx];
        let current = *value;
        let updated = if alpha > 0.0 && prev > DB_FLOOR {
            alpha * prev + (1.0 - alpha) * current
        } else {
            current
        };
        history[idx] = updated;
        *value = updated;
    }
}

pub fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

pub fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

impl Reconfigurable<SpectrogramConfig> for SpectrogramProcessor {
    fn update_config(&mut self, config: SpectrogramConfig) {
        let previous = self.config;

        // Persist the user-facing configuration for export and future edits.
        self.stored_config = config;

        // Apply the normalized configuration so helpers observe gated values.
        self.config = config;
        Self::normalize_config(&mut self.config);

        // Rebuild the FFT pipeline when structural parameters change. These
        // alterations force buffer and window reshaping, so preserving history
        // would yield inconsistent spectra anyway.
        let fft_related_changed = previous.fft_size != self.config.fft_size
            || previous.zero_padding_factor != self.config.zero_padding_factor
            || previous.window != self.config.window
            || (previous.sample_rate - self.config.sample_rate).abs() > f32::EPSILON;

        if fft_related_changed {
            self.rebuild_fft();
            return;
        }

        // Resize history without flushing existing columns so the UI can keep
        // rendering accumulated data while the user adjusts history length.
        if previous.history_length != self.config.history_length {
            let mut evicted = std::mem::take(&mut self.evicted_columns);
            self.history
                .set_capacity(self.config.history_length, &mut evicted);
            evicted
                .drain(..)
                .for_each(|column| self.recycle_column(column));
            self.evicted_columns = evicted;
        }

        // Update the reassignment floor to match the latest dB threshold.
        if previous.reassignment_power_floor_db != self.config.reassignment_power_floor_db {
            self.reassignment_power_floor_linear =
                power_db_to_linear(self.config.reassignment_power_floor_db);
        }

        // Refresh synchrosqueezing caches whenever the underlying grid changes.
        let synchro_changed = previous.use_synchrosqueezing != self.config.use_synchrosqueezing
            || previous.use_reassignment != self.config.use_reassignment
            || previous.synchrosqueezing_bin_count != self.config.synchrosqueezing_bin_count
            || (previous.synchrosqueezing_min_hz - self.config.synchrosqueezing_min_hz).abs()
                > f32::EPSILON
            || previous.frequency_scale != self.config.frequency_scale;

        if synchro_changed {
            self.synchro
                .reconfigure(&self.config, self.fft_size, self.config.sample_rate);
            self.synchro.align_pool(&mut self.synchro_pool);
            self.synchro.reset_temporal();
            self.synchro.reset_power();
            self.synchro.reset_frequency();
            self.clear_history();
        }

        // Reset temporal smoothing buffers when parameters change so that
        // legacy decay state does not bleed into the new response curve.
        let temporal_smoothing_changed = previous.temporal_smoothing
            != self.config.temporal_smoothing
            || previous.temporal_smoothing_max_hz != self.config.temporal_smoothing_max_hz
            || previous.temporal_smoothing_blend_hz != self.config.temporal_smoothing_blend_hz;

        if temporal_smoothing_changed {
            if self.temporal_smoothing_buffer.len() != self.magnitude_buffer.len() {
                self.temporal_smoothing_buffer
                    .resize(self.magnitude_buffer.len(), DB_FLOOR);
            }
            self.temporal_smoothing_buffer.fill(DB_FLOOR);
            self.synchro.reset_temporal();
        }

        let frequency_smoothing_changed = previous.frequency_smoothing_radius
            != self.config.frequency_smoothing_radius
            || previous.frequency_smoothing_max_hz != self.config.frequency_smoothing_max_hz
            || previous.frequency_smoothing_blend_hz != self.config.frequency_smoothing_blend_hz;

        if frequency_smoothing_changed {
            if self.frequency_scratch_buffer.len() != self.magnitude_buffer.len() {
                self.frequency_scratch_buffer
                    .resize(self.magnitude_buffer.len(), 0.0);
            } else {
                self.frequency_scratch_buffer.fill(0.0);
            }
            self.synchro.reset_frequency();
            self.clear_history();
        }
    }
}

fn duration_from_samples(sample_index: u64, sample_rate: f32) -> Duration {
    if sample_rate > 0.0 {
        Duration::from_secs_f64(sample_index as f64 / sample_rate as f64)
    } else {
        Duration::default()
    }
}

fn resize_vecdeque(buffer: &mut VecDeque<f32>, capacity: usize) {
    if capacity == 0 {
        buffer.clear();
        buffer.shrink_to_fit();
    } else if buffer.len() > capacity {
        buffer.drain(..buffer.len() - capacity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::{AudioBlock, ProcessorUpdate};
    use std::time::Instant;

    fn make_block(samples: Vec<f32>, channels: usize, sample_rate: f32) -> AudioBlock<'static> {
        AudioBlock::new(
            Box::leak(samples.into_boxed_slice()),
            channels,
            sample_rate,
            Instant::now(),
        )
    }

    fn test_config_base() -> SpectrogramConfig {
        SpectrogramConfig {
            temporal_smoothing: 0.0,
            temporal_smoothing_max_hz: 0.0,
            temporal_smoothing_blend_hz: 0.0,
            frequency_smoothing_radius: 0,
            frequency_smoothing_max_hz: 0.0,
            frequency_smoothing_blend_hz: 0.0,
            ..SpectrogramConfig::default()
        }
    }

    #[test]
    fn detects_sine_frequency_peak() {
        let config = SpectrogramConfig {
            fft_size: 1024,
            hop_size: 512,
            history_length: 8,
            sample_rate: DEFAULT_SAMPLE_RATE,
            window: WindowKind::Hann,
            zero_padding_factor: 1,
            ..test_config_base()
        };
        let mut processor = SpectrogramProcessor::new(config);

        let zero_padding = config.zero_padding_factor.max(1);
        let effective_fft_size = config.fft_size * zero_padding;
        let bin_hz = config.sample_rate / effective_fft_size as f32;
        let target_bin_unpadded = 200usize;
        let target_bin = target_bin_unpadded * zero_padding;
        let freq = target_bin as f32 * bin_hz;
        let frames = config.fft_size * 2;
        let mut samples = Vec::with_capacity(frames);
        for n in 0..frames {
            let t = n as f32 / config.sample_rate;
            samples.push((2.0 * core::f32::consts::PI * freq * t).sin());
        }

        let block = make_block(samples, 1, config.sample_rate);

        let result = processor.process_block(&block);
        let update = match result {
            ProcessorUpdate::Snapshot(update) => update,
            ProcessorUpdate::None => panic!("expected snapshot"),
        };

        assert!(!update.new_columns.is_empty());
        let last = update.new_columns.last().unwrap();
        let max_index = last
            .magnitudes_db
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        let peak_freq = max_index as f32 * bin_hz;
        assert!((peak_freq - freq).abs() < bin_hz * 1.5);

        assert_eq!(max_index, target_bin);

        let peak_db = last.magnitudes_db[max_index];
        assert!(
            peak_db > -1.5 && peak_db < 2.0,
            "expected near 0 dBFS peak, saw {peak_db}",
        );
    }

    #[test]
    fn history_respects_limit() {
        let config = SpectrogramConfig {
            history_length: 4,
            ..test_config_base()
        };
        let mut processor = SpectrogramProcessor::new(config);
        let frames = config.fft_size * 4;
        let samples = vec![0.0f32; frames];
        let block = make_block(samples, 1, config.sample_rate);
        let _ = processor.process_block(&block);
        assert!(processor.history.slots.len() <= config.history_length);
    }

    #[test]
    fn reassignment_emits_samples() {
        let config = SpectrogramConfig {
            fft_size: 2048,
            hop_size: 512,
            history_length: 4,
            sample_rate: DEFAULT_SAMPLE_RATE,
            window: WindowKind::Hann,
            use_reassignment: true,
            reassignment_low_bin_limit: 512,
            ..test_config_base()
        };
        let mut processor = SpectrogramProcessor::new(config);

        let frames = config.fft_size * 2;
        let freq_hz = 110.0f32;
        let mut samples = Vec::with_capacity(frames);
        for n in 0..frames {
            let t = n as f32 / config.sample_rate;
            samples.push((2.0 * core::f32::consts::PI * freq_hz * t).sin());
        }

        let block = make_block(samples, 1, config.sample_rate);
        let update = match processor.process_block(&block) {
            ProcessorUpdate::Snapshot(update) => update,
            ProcessorUpdate::None => panic!("expected reassignment snapshot"),
        };

        assert!(update.reassignment_enabled);
        let sample_count = update
            .new_columns
            .last()
            .and_then(|column| column.reassigned.as_ref())
            .map(|samples| samples.len())
            .unwrap_or(0);
        assert!(sample_count > 0);
    }

    #[test]
    fn reassignment_concentrates_energy_at_fundamental() {
        let config = SpectrogramConfig {
            fft_size: 4096,
            hop_size: 256,
            history_length: 4,
            sample_rate: DEFAULT_SAMPLE_RATE,
            window: WindowKind::Hann,
            use_reassignment: true,
            reassignment_low_bin_limit: 0,
            ..test_config_base()
        };
        let mut processor = SpectrogramProcessor::new(config);

        let frames = config.fft_size * 3;
        let freq_hz = 130.812_78_f32; // C3
        let mut samples = Vec::with_capacity(frames);
        for n in 0..frames {
            let t = n as f32 / config.sample_rate;
            samples.push((2.0 * core::f32::consts::PI * freq_hz * t).sin());
        }

        let block = make_block(samples, 1, config.sample_rate);
        let update = match processor.process_block(&block) {
            ProcessorUpdate::Snapshot(update) => update,
            ProcessorUpdate::None => panic!("expected reassignment snapshot"),
        };

        let samples = update
            .new_columns
            .last()
            .and_then(|column| column.reassigned.as_ref())
            .expect("reassignment samples present");

        let mut total_energy = 0.0f32;
        let mut off_harmonic_energy = 0.0f32;
        let mut contributions = Vec::new();
        for sample in samples.iter() {
            if !sample.frequency_hz.is_finite() {
                continue;
            }
            let power = if sample.magnitude_db <= DB_FLOOR {
                0.0
            } else {
                10.0f32.powf(sample.magnitude_db / 10.0)
            };
            if power <= 0.0 {
                continue;
            }

            total_energy += power;
            let ratio = sample.frequency_hz / freq_hz;
            let harmonic_distance = (ratio.round() - ratio).abs();
            if harmonic_distance > 0.05 {
                off_harmonic_energy += power;
            }
            contributions.push((sample.frequency_hz, power));
        }

        contributions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        assert!(total_energy > 0.0);
        let leakage_ratio = off_harmonic_energy / total_energy;
        assert!(
            leakage_ratio < 0.02,
            "excess harmonic leakage: {leakage_ratio}"
        );
    }

    #[test]
    fn reassignment_tracks_low_frequency_precisely() {
        let config = SpectrogramConfig {
            fft_size: 4096,
            hop_size: 512,
            history_length: 4,
            sample_rate: DEFAULT_SAMPLE_RATE,
            window: WindowKind::Hann,
            use_reassignment: true,
            reassignment_low_bin_limit: 256,
            ..test_config_base()
        };
        let mut processor = SpectrogramProcessor::new(config);

        let freq_hz = 58.5_f32;
        let frames = config.fft_size * 2;
        let mut samples = Vec::with_capacity(frames);
        for n in 0..frames {
            let t = n as f32 / config.sample_rate;
            samples.push((2.0 * core::f32::consts::PI * freq_hz * t).sin());
        }

        let block = make_block(samples, 1, config.sample_rate);
        let update = match processor.process_block(&block) {
            ProcessorUpdate::Snapshot(update) => update,
            ProcessorUpdate::None => panic!("expected reassignment snapshot"),
        };

        let zero_padding = config.zero_padding_factor.max(1);
        let effective_fft_size = config.fft_size * zero_padding;
        let bin_hz = config.sample_rate / effective_fft_size as f32;
        let column = update
            .new_columns
            .last()
            .expect("spectrogram produced at least one column");
        let reassigned = column
            .reassigned
            .as_ref()
            .expect("reassignment samples present");

        let peak_sample = reassigned
            .iter()
            .max_by(|a, b| a.magnitude_db.partial_cmp(&b.magnitude_db).unwrap())
            .copied()
            .expect("at least one reassigned sample");

        assert!((peak_sample.frequency_hz - freq_hz).abs() < 0.75);
        assert!(peak_sample.magnitude_db > -3.0);

        let approx_bin = (freq_hz / bin_hz).round() as usize;
        let peak_bin = column
            .magnitudes_db
            .iter()
            .take(config.reassignment_low_bin_limit)
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .expect("non-empty magnitude column");
        assert!((peak_bin as isize - approx_bin as isize).abs() <= 1);
        assert!(column.magnitudes_db[approx_bin] > -3.0);
    }

    #[test]
    fn frequency_smoothing_blurs_energy_across_bins() {
        let mut unsmoothed_cfg = test_config_base();
        unsmoothed_cfg.fft_size = 1024;
        unsmoothed_cfg.hop_size = 512;
        unsmoothed_cfg.history_length = 4;
        unsmoothed_cfg.zero_padding_factor = 1;
        unsmoothed_cfg.frequency_smoothing_radius = 0;
        unsmoothed_cfg.frequency_smoothing_max_hz = unsmoothed_cfg.sample_rate * 0.5;
        unsmoothed_cfg.frequency_smoothing_blend_hz = 0.0;
        unsmoothed_cfg.use_reassignment = false;
        unsmoothed_cfg.use_synchrosqueezing = false;

        let mut unsmoothed = SpectrogramProcessor::new(unsmoothed_cfg);

        let frames = unsmoothed_cfg.fft_size * 2;
        let freq_hz = 1200.0f32;
        let mut samples = Vec::with_capacity(frames);
        for n in 0..frames {
            let t = n as f32 / unsmoothed_cfg.sample_rate;
            samples.push((2.0 * core::f32::consts::PI * freq_hz * t).sin());
        }
        let block = make_block(samples, 1, unsmoothed_cfg.sample_rate);
        let unsmoothed_update = match unsmoothed.process_block(&block) {
            ProcessorUpdate::Snapshot(update) => update,
            ProcessorUpdate::None => panic!("expected spectrogram snapshot"),
        };
        let unsmoothed_column = unsmoothed_update
            .new_columns
            .last()
            .expect("spectrogram produced column");

        let mut smoothed_cfg = unsmoothed_cfg;
        smoothed_cfg.frequency_smoothing_radius = 12;
        let mut smoothed = SpectrogramProcessor::new(smoothed_cfg);
        let samples = (0..frames)
            .map(|n| {
                let t = n as f32 / smoothed_cfg.sample_rate;
                (2.0 * core::f32::consts::PI * freq_hz * t).sin()
            })
            .collect::<Vec<_>>();
        let block = make_block(samples, 1, smoothed_cfg.sample_rate);
        let smoothed_update = match smoothed.process_block(&block) {
            ProcessorUpdate::Snapshot(update) => update,
            ProcessorUpdate::None => panic!("expected spectrogram snapshot"),
        };
        let smoothed_column = smoothed_update
            .new_columns
            .last()
            .expect("spectrogram produced column");

        let (peak_idx, &unsmoothed_peak) = unsmoothed_column
            .magnitudes_db
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .expect("non-empty column");
        let smoothed_peak = smoothed_column.magnitudes_db[peak_idx];
        assert!(smoothed_peak < unsmoothed_peak - 6.0);

        let neighbor = (peak_idx + 6).min(smoothed_column.magnitudes_db.len() - 1);
        let unsmoothed_contrast = unsmoothed_peak - unsmoothed_column.magnitudes_db[neighbor];
        let smoothed_contrast = smoothed_peak - smoothed_column.magnitudes_db[neighbor];
        assert!(smoothed_contrast < unsmoothed_contrast - 6.0);
    }

    #[test]
    fn frequency_smoothing_softens_synchro_peak() {
        let mut base_cfg = test_config_base();
        base_cfg.fft_size = 2048;
        base_cfg.hop_size = 512;
        base_cfg.history_length = 4;
        base_cfg.zero_padding_factor = 1;
        base_cfg.reassignment_low_bin_limit = 0;
        base_cfg.synchrosqueezing_bin_count = 256;
        base_cfg.frequency_smoothing_max_hz = base_cfg.sample_rate * 0.5;
        base_cfg.frequency_smoothing_blend_hz = 0.0;

        let mut unsmoothed_cfg = base_cfg;
        unsmoothed_cfg.frequency_smoothing_radius = 0;

        let mut smoothed_cfg = base_cfg;
        smoothed_cfg.frequency_smoothing_radius = 10;
        let mut unsmoothed_processor = SpectrogramProcessor::new(unsmoothed_cfg);
        let mut smoothed_processor = SpectrogramProcessor::new(smoothed_cfg);

        let frames = base_cfg.fft_size * 3;
        let freq_hz = 1400.0f32;

        let unsmoothed_samples = (0..frames)
            .map(|n| {
                let t = n as f32 / unsmoothed_cfg.sample_rate;
                (2.0 * core::f32::consts::PI * freq_hz * t).sin()
            })
            .collect::<Vec<_>>();
        let unsmoothed_block = make_block(unsmoothed_samples, 1, unsmoothed_cfg.sample_rate);
        let unsmoothed_update = match unsmoothed_processor.process_block(&unsmoothed_block) {
            ProcessorUpdate::Snapshot(update) => update,
            ProcessorUpdate::None => panic!("expected synchrosqueezed snapshot"),
        };

        let smoothed_samples = (0..frames)
            .map(|n| {
                let t = n as f32 / smoothed_cfg.sample_rate;
                (2.0 * core::f32::consts::PI * freq_hz * t).sin()
            })
            .collect::<Vec<_>>();
        let smoothed_block = make_block(smoothed_samples, 1, smoothed_cfg.sample_rate);
        let smoothed_update = match smoothed_processor.process_block(&smoothed_block) {
            ProcessorUpdate::Snapshot(update) => update,
            ProcessorUpdate::None => panic!("expected synchrosqueezed snapshot"),
        };

        let bins = unsmoothed_update
            .synchro_bins_hz
            .as_ref()
            .expect("synchro bins available");
        let column_unsmoothed = unsmoothed_update
            .new_columns
            .last()
            .and_then(|column| column.synchro_magnitudes_db.as_ref())
            .expect("unsmoothed synchro column");
        let column_smoothed = smoothed_update
            .new_columns
            .last()
            .and_then(|column| column.synchro_magnitudes_db.as_ref())
            .expect("smoothed synchro column");

        assert_eq!(column_unsmoothed.len(), column_smoothed.len());
        assert_eq!(column_unsmoothed.len(), bins.len());

        let target_idx = bins
            .iter()
            .enumerate()
            .min_by(|a, b| {
                let da = (*a.1 - freq_hz).abs();
                let db = (*b.1 - freq_hz).abs();
                da.partial_cmp(&db).unwrap()
            })
            .map(|(idx, _)| idx)
            .expect("target frequency bin");

        let unsmoothed_peak = column_unsmoothed[target_idx];
        let smoothed_peak = column_smoothed[target_idx];
        assert!(smoothed_peak < unsmoothed_peak - 3.0);

        let neighbor_idx = if target_idx + 5 < column_unsmoothed.len() {
            target_idx + 5
        } else {
            target_idx.saturating_sub(5)
        };
        let unsmoothed_contrast = unsmoothed_peak - column_unsmoothed[neighbor_idx];
        let smoothed_contrast = smoothed_peak - column_smoothed[neighbor_idx];
        assert!(smoothed_contrast < unsmoothed_contrast - 3.0);
    }

    #[test]
    fn zero_padding_expands_fft_resolution() {
        let zero_padding_factor = 4;
        let config = SpectrogramConfig {
            fft_size: 1024,
            hop_size: 256,
            history_length: 4,
            sample_rate: DEFAULT_SAMPLE_RATE,
            window: WindowKind::Hann,
            zero_padding_factor,
            ..test_config_base()
        };
        let mut processor = SpectrogramProcessor::new(config);

        let freq_hz = 440.0f32;
        let frames = config.fft_size * 2;
        let mut samples = Vec::with_capacity(frames);
        for n in 0..frames {
            let t = n as f32 / config.sample_rate;
            samples.push((2.0 * core::f32::consts::PI * freq_hz * t).sin());
        }

        let block = make_block(samples, 1, config.sample_rate);
        let update = match processor.process_block(&block) {
            ProcessorUpdate::Snapshot(update) => update,
            ProcessorUpdate::None => panic!("expected spectrogram snapshot"),
        };

        assert_eq!(update.fft_size, config.fft_size * zero_padding_factor);

        let column = update
            .new_columns
            .last()
            .expect("spectrogram produced at least one column");
        assert_eq!(column.magnitudes_db.len(), update.fft_size / 2 + 1);

        let padded_bin_hz = config.sample_rate / update.fft_size as f32;
        let unpadded_bin_hz = config.sample_rate / config.fft_size as f32;
        assert!(padded_bin_hz < unpadded_bin_hz);
        let ratio = unpadded_bin_hz / padded_bin_hz;
        assert!((ratio - zero_padding_factor as f32).abs() < 1.0e-6);
    }

    #[test]
    fn mel_conversions_are_invertible() {
        let test_frequencies = [20.0, 100.0, 440.0, 1000.0, 4000.0, 10000.0];
        for &hz in &test_frequencies {
            let mel = hz_to_mel(hz);
            let reconstructed = mel_to_hz(mel);
            assert!(
                (hz - reconstructed).abs() < 0.01,
                "Failed roundtrip for {} Hz",
                hz
            );
        }
    }
}
