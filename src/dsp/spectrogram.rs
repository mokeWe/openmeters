//! spectrogram DSP. reassignment! (Auger-Flandrin 1995).
//! 2. K. Kodera, R. Gendrin & C. de Villedary, "Analysis of time-varying signals
//!    with small BT values", IEEE Trans. ASSP, vol. 26, no. 1, pp. 64-76, Feb 1978.
//!

use super::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::util::audio::{DB_FLOOR, DEFAULT_SAMPLE_RATE, copy_from_deque, db_to_power, power_to_db};
pub use crate::util::audio::{hz_to_mel, mel_to_hz};
use parking_lot::RwLock;
use realfft::{RealFftPlanner, RealToComplex};
use rustc_hash::FxHashMap;
use rustfft::num_complex::Complex32;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

const MAX_REASSIGNMENT_SAMPLES: usize = 8192;
pub const PLANCK_BESSEL_DEFAULT_EPSILON: f32 = 0.1;
pub const PLANCK_BESSEL_DEFAULT_BETA: f32 = 5.5;

/// Resize a VecDeque, trimming from the front if oversized.
#[inline]
fn resize_vecdeque<T: Default + Clone>(buffer: &mut VecDeque<T>, capacity: usize) {
    if capacity == 0 {
        buffer.clear();
        buffer.shrink_to_fit();
    } else if buffer.len() > capacity {
        buffer.drain(..buffer.len() - capacity);
    }
}

/// Configuration for spectrogram FFT analysis.
#[derive(Debug, Clone, Copy)]
pub struct SpectrogramConfig {
    pub sample_rate: f32,
    pub fft_size: usize,
    pub hop_size: usize,
    pub window: WindowKind,
    pub frequency_scale: FrequencyScale,
    pub history_length: usize,
    pub use_reassignment: bool,
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
    /// Maximum allowed frequency correction during reassignment, in Hz.
    /// Bins with corrections exceeding this limit are skipped as likely artifacts.
    /// Set to 0.0 to disable bounds checking (not recommended).
    /// Default: sample_rate / fft_size (one bin width).
    pub reassignment_max_correction_hz: f32,
}

impl Default for SpectrogramConfig {
    fn default() -> Self {
        Self {
            sample_rate: DEFAULT_SAMPLE_RATE,
            fft_size: 4096,
            hop_size: 256,
            // I'm not sure what the "best" alpha/beta value is for spectral analysis.
            // going for moderate sidelobe suppression without excessive main lobe widening.
            window: WindowKind::PlanckBessel {
                epsilon: PLANCK_BESSEL_DEFAULT_EPSILON,
                beta: PLANCK_BESSEL_DEFAULT_BETA,
            },
            frequency_scale: FrequencyScale::default(),
            history_length: 240,
            use_reassignment: true,
            reassignment_power_floor_db: -80.0,
            reassignment_low_bin_limit: 0,
            zero_padding_factor: 4,
            use_synchrosqueezing: true,
            synchrosqueezing_bin_count: 1024,
            synchrosqueezing_min_hz: 20.0,
            temporal_smoothing: 0.4,
            temporal_smoothing_max_hz: 900.0,
            temporal_smoothing_blend_hz: 400.0,
            frequency_smoothing_radius: 0,
            frequency_smoothing_max_hz: 2400.0,
            frequency_smoothing_blend_hz: 600.0,
            // Default to one bin width; computed dynamically if left at 0
            reassignment_max_correction_hz: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FrequencyScale {
    Linear,
    #[default]
    Logarithmic,
    Mel,
}

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
            WindowKind::Hann => Self::cosine_sum_window(len, &[0.5, -0.5]),
            WindowKind::Hamming => Self::cosine_sum_window(len, &[0.54, -0.46]),
            WindowKind::Blackman => Self::cosine_sum_window(len, &[0.42, -0.5, 0.08]),
            WindowKind::PlanckBessel { epsilon, beta } => {
                planck_bessel_coefficients(len, epsilon, beta)
            }
        }
    }

    fn cosine_sum_window(len: usize, coeffs: &[f32]) -> Vec<f32> {
        let denom = len.saturating_sub(1).max(1) as f32;
        let scale = core::f32::consts::TAU / denom;

        let mut window = vec![0.0; len];

        match coeffs {
            [a0, a1] => {
                let a0 = *a0;
                let a1 = *a1;
                for (n, value) in window.iter_mut().enumerate() {
                    let phase = (n as f32) * scale;
                    *value = a1.mul_add(phase.cos(), a0);
                }
            }
            [a0, a1, a2] => {
                let a0 = *a0;
                let a1 = *a1;
                let a2 = *a2;
                for (n, value) in window.iter_mut().enumerate() {
                    let phase = (n as f32) * scale;
                    let c1 = phase.cos();
                    // cos(2θ) = 2cos²(θ) - 1
                    let c2 = (2.0 * c1).mul_add(c1, -1.0);
                    *value = a2.mul_add(c2, a1.mul_add(c1, a0));
                }
            }
            _ => {
                for (n, value) in window.iter_mut().enumerate() {
                    let base_phase = (n as f32) * scale;
                    let mut sum = 0.0f32;
                    for (k, &a) in coeffs.iter().enumerate() {
                        let k_phase = base_phase * (k as f32);
                        sum = a.mul_add(k_phase.cos(), sum);
                    }
                    *value = sum;
                }
            }
        }
        window
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
        // Abramowitz & Stegun 9.8.4: asymptotic expansion for I₁(x)
        let y = 3.75 / ax;
        let poly = 0.398_942_28
            + y * (-0.039_880_24
                + y * (-0.003_620_18
                    + y * (0.001_638_01
                        + y * (-0.010_315_55
                            + y * (0.022_829_67
                                + y * (-0.028_953_12
                                    + y * (0.017_876_54 - y * 0.004_200_59)))))));
        let ans = poly * ax.exp() / ax.sqrt();
        if x < 0.0 { -ans } else { ans }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct ReassignedSample {
    pub frequency_hz: f32,
    pub group_delay_samples: f32,
    pub magnitude_db: f32,
}

/// Buffers for time-frequency reassignment (derivative and time-weighted windows).
struct ReassignmentBuffers {
    derivative_window: Vec<f32>,
    time_weighted_window: Vec<f32>,
    derivative_buffer: Vec<f32>,
    time_weighted_buffer: Vec<f32>,
    derivative_spectrum: Vec<Complex32>,
    time_weighted_spectrum: Vec<Complex32>,
    power_buffer: Vec<f32>,
    sample_cache: Vec<ReassignedSample>,
    power_floor_linear: f32,
}

impl ReassignmentBuffers {
    fn new(
        window_kind: WindowKind,
        window: &[f32],
        fft: &Arc<dyn RealToComplex<f32>>,
        fft_size: usize,
        power_floor_db: f32,
    ) -> Self {
        let bins = fft_size / 2 + 1;
        Self {
            derivative_window: compute_derivative_window(window_kind, window),
            time_weighted_window: compute_time_weighted_window(window),
            derivative_buffer: vec![0.0; fft_size],
            time_weighted_buffer: vec![0.0; fft_size],
            derivative_spectrum: fft.make_output_vec(),
            time_weighted_spectrum: fft.make_output_vec(),
            power_buffer: vec![0.0; bins],
            sample_cache: Vec::with_capacity(bins >> 4),
            power_floor_linear: db_to_power(power_floor_db),
        }
    }

    fn resize(
        &mut self,
        window_kind: WindowKind,
        window: &[f32],
        fft: &Arc<dyn RealToComplex<f32>>,
        fft_size: usize,
        power_floor_db: f32,
    ) {
        self.derivative_window = compute_derivative_window(window_kind, window);
        self.time_weighted_window = compute_time_weighted_window(window);
        self.derivative_buffer.resize(fft_size, 0.0);
        self.time_weighted_buffer.resize(fft_size, 0.0);
        self.derivative_spectrum = fft.make_output_vec();
        self.time_weighted_spectrum = fft.make_output_vec();
        self.power_buffer.resize(fft_size / 2 + 1, 0.0);
        self.power_floor_linear = db_to_power(power_floor_db);
    }
}

/// One column of log-power magnitudes, with optional reassigned samples.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SpectrogramColumn {
    pub timestamp: Instant,
    pub magnitudes_db: Arc<[f32]>,
    pub reassigned: Option<Arc<[ReassignedSample]>>,
    pub synchro_magnitudes_db: Option<Arc<[f32]>>,
}

/// Incremental update emitted by the spectrogram processor.
#[derive(Debug, Clone)]
pub struct SpectrogramUpdate {
    pub fft_size: usize,
    pub hop_size: usize,
    pub sample_rate: f32,
    pub frequency_scale: FrequencyScale,
    pub history_length: usize,
    pub reset: bool,
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
            let excess = self.slots.len().saturating_sub(capacity);
            evicted.extend(self.slots.drain(..excess));
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
    min_hz_f64: f64,
    max_hz_f64: f64,
    inv_log_range: f64,
    inv_mel_range: f64,
    inv_linear_range: f64,
    bin_count_minus_1: f32,
}

impl SynchroState {
    fn new(config: &SpectrogramConfig, fft_size: usize, sample_rate: f32) -> Self {
        let mut state = Self::default();
        state.reconfigure(config, fft_size, sample_rate);
        state
    }

    fn reconfigure(&mut self, config: &SpectrogramConfig, fft_size: usize, sample_rate: f32) {
        let enable = config.use_synchrosqueezing
            && config.use_reassignment
            && config.synchrosqueezing_bin_count != 0
            && sample_rate > 0.0
            && fft_size != 0;

        if !enable {
            self.enabled = false;
            self.bin_frequencies = Arc::from([]);
            self.power_buffer.clear();
            self.magnitude_buffer.clear();
            self.temporal_buffer.clear();
            self.frequency_buffer.clear();
            self.min_hz = 0.0;
            self.max_hz = 0.0;
            self.log_min = 0.0;
            self.log_range = 0.0;
            self.mel_min = 0.0;
            self.mel_range = 0.0;
            self.min_hz_f64 = 0.0;
            self.max_hz_f64 = 0.0;
            self.inv_log_range = 0.0;
            self.inv_mel_range = 0.0;
            self.inv_linear_range = 0.0;
            self.bin_count_minus_1 = 0.0;
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

        let freqs_changed = self.bin_frequencies.len() != bin_count
            || self.scale != scale
            || (self.min_hz - min_hz).abs() > f32::EPSILON
            || (self.max_hz - nyquist).abs() > f32::EPSILON;

        if freqs_changed {
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
                        FrequencyScale::Logarithmic => (log_min + log_range * t).exp() as f32,
                        FrequencyScale::Mel => mel_to_hz((mel_min + mel_range * t) as f32),
                    };
                    freqs.push(freq);
                }
                freqs.reverse();
                freqs
            };
            self.bin_frequencies = Arc::from(freqs.into_boxed_slice());
        }

        self.enabled = true;
        self.scale = scale;
        self.min_hz = min_hz;
        self.max_hz = nyquist;
        self.log_min = log_min;
        self.log_range = log_range;
        self.mel_min = mel_min;
        self.mel_range = mel_range;

        self.min_hz_f64 = min_hz as f64;
        self.max_hz_f64 = nyquist as f64;
        self.inv_log_range = if log_range > 1.0e-9 {
            1.0 / log_range
        } else {
            0.0
        };
        self.inv_mel_range = if mel_range > 1.0e-9 {
            1.0 / mel_range
        } else {
            0.0
        };
        self.inv_linear_range = if (nyquist - min_hz) > f32::EPSILON {
            1.0 / (nyquist as f64 - min_hz as f64)
        } else {
            0.0
        };
        self.bin_count_minus_1 = (bin_count - 1) as f32;

        self.power_buffer.resize(bin_count, 0.0);
        self.power_buffer.fill(0.0);
        self.magnitude_buffer.resize(bin_count, DB_FLOOR);
        self.magnitude_buffer.fill(DB_FLOOR);
        self.temporal_buffer.resize(bin_count, DB_FLOOR);
        self.temporal_buffer.fill(DB_FLOOR);
        self.frequency_buffer.clear();
    }

    fn is_active(&self) -> bool {
        self.enabled && !self.power_buffer.is_empty() && self.log_range > 0.0
    }

    fn bins_arc(&self) -> Option<&Arc<[f32]>> {
        self.enabled
            .then_some(&self.bin_frequencies)
            .filter(|b| !b.is_empty())
    }

    fn reset_power(&mut self) {
        self.power_buffer.fill(0.0);
    }

    #[inline(always)]
    fn accumulate(&mut self, freq_hz: f64, display_power: f32) {
        if display_power <= 0.0 {
            return;
        }

        if freq_hz < self.min_hz_f64 || freq_hz > self.max_hz_f64 {
            return;
        }

        let normalized = match self.scale {
            FrequencyScale::Linear => ((freq_hz - self.min_hz_f64) * self.inv_linear_range) as f32,
            FrequencyScale::Logarithmic => {
                ((freq_hz.ln() - self.log_min) * self.inv_log_range) as f32
            }
            FrequencyScale::Mel => {
                let mel = hz_to_mel(freq_hz as f32) as f64;
                ((mel - self.mel_min) * self.inv_mel_range) as f32
            }
        };

        let position = (1.0 - normalized) * self.bin_count_minus_1;
        let lower = position as usize;
        let frac = position - lower as f32;

        if let Some(power_lower) = self.power_buffer.get_mut(lower) {
            *power_lower += display_power * (1.0 - frac);

            let upper = lower + 1;
            if let Some(power_upper) = self.power_buffer.get_mut(upper) {
                *power_upper += display_power * frac;
            }
        }
    }

    #[inline]
    fn finalize_magnitudes(&mut self) {
        if !self.enabled {
            return;
        }

        let len = self.magnitude_buffer.len().min(self.power_buffer.len());

        for i in 0..len {
            self.magnitude_buffer[i] = power_to_db(self.power_buffer[i]);
        }
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

    fn reset_all(&mut self) {
        self.reset_power();
        self.reset_temporal();
        self.reset_frequency();
    }

    fn align_pool(&self, pool: &mut Vec<Arc<[f32]>>) {
        pool.retain(|buffer| buffer.len() == self.magnitude_buffer.len());
    }

    fn apply_frequency_smoothing(&mut self, radius: usize, max_hz: f32, blend_hz: f32) {
        if !self.enabled || self.magnitude_buffer.is_empty() || radius == 0 {
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

        let magnitude_slice = &self.magnitude_buffer;
        for idx in 0..bins {
            let start = idx.saturating_sub(radius);
            let end = (idx + radius + 1).min(bins);
            let inv_count = 1.0 / (end - start) as f32;

            let sum: f32 = magnitude_slice[start..end].iter().sum();
            self.frequency_buffer[idx] = sum * inv_count;
        }

        for (idx, value) in self.magnitude_buffer.iter_mut().enumerate() {
            let smoothed = self.frequency_buffer[idx];
            let weight = smoothing_weight(self.bin_frequencies[idx], max_hz, blend_hz);
            *value = smoothed * weight + *value * (1.0 - weight);
        }
    }
}

pub struct SpectrogramProcessor {
    config: SpectrogramConfig,
    planner: RealFftPlanner<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
    window_size: usize,
    fft_size: usize,
    window: Arc<[f32]>,
    real_buffer: Vec<f32>,
    spectrum_buffer: Vec<Complex32>,
    scratch_buffer: Vec<Complex32>,
    magnitude_buffer: Vec<f32>,
    reassignment: ReassignmentBuffers,
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
    temporal_smoothing_buffer: Vec<f32>,
    frequency_scratch_buffer: Vec<f32>,
    pending_reset: bool,
    output_columns_buffer: Vec<SpectrogramColumn>,
}

impl SpectrogramProcessor {
    pub fn new(config: SpectrogramConfig) -> Self {
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

        let reassignment = ReassignmentBuffers::new(
            runtime_config.window,
            window.as_ref(),
            &fft,
            fft_size,
            runtime_config.reassignment_power_floor_db,
        );

        let synchro = SynchroState::new(&runtime_config, fft_size, runtime_config.sample_rate);

        let spectrum_buffer = fft.make_output_vec();
        let scratch_buffer = fft.make_scratch_vec();
        let bin_normalization =
            crate::util::audio::compute_fft_bin_normalization(window.as_ref(), fft_size);
        let energy_normalization = Self::compute_energy_normalization(window.as_ref(), fft_size);

        Self {
            config: runtime_config,
            planner,
            fft,
            window_size,
            fft_size,
            window,
            real_buffer: vec![0.0; fft_size],
            spectrum_buffer,
            scratch_buffer,
            magnitude_buffer: vec![0.0; bins],
            reassignment,
            synchro,
            bin_normalization,
            energy_normalization,
            pcm_buffer: VecDeque::with_capacity(window_size.saturating_mul(2)),
            buffer_start_index: 0,
            start_instant: None,
            accumulated_offset: Duration::default(),
            history: SpectrogramHistory::new(history_len),
            magnitude_pool: Vec::new(),
            synchro_pool: Vec::new(),
            evicted_columns: Vec::new(),
            temporal_smoothing_buffer: vec![DB_FLOOR; bins],
            frequency_scratch_buffer: vec![0.0; bins],
            pending_reset: true,
            output_columns_buffer: Vec::with_capacity(8),
        }
    }

    fn normalize_config(config: &mut SpectrogramConfig) {
        if !config.use_reassignment {
            config.use_synchrosqueezing = false;
        }
    }

    pub fn config(&self) -> SpectrogramConfig {
        self.config
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

        self.real_buffer.resize(fft_size, 0.0);
        self.spectrum_buffer = self.fft.make_output_vec();
        self.scratch_buffer = self.fft.make_scratch_vec();
        self.magnitude_buffer.resize(bins, 0.0);

        self.reassignment.resize(
            self.config.window,
            self.window.as_ref(),
            &self.fft,
            fft_size,
            self.config.reassignment_power_floor_db,
        );

        self.bin_normalization =
            crate::util::audio::compute_fft_bin_normalization(self.window.as_ref(), fft_size);
        self.energy_normalization =
            Self::compute_energy_normalization(self.window.as_ref(), fft_size);

        self.temporal_smoothing_buffer.resize(bins, DB_FLOOR);
        self.frequency_scratch_buffer.resize(bins, 0.0);

        self.synchro
            .reconfigure(&self.config, fft_size, self.config.sample_rate);
        self.synchro.align_pool(&mut self.synchro_pool);
        self.synchro.reset_all();

        self.magnitude_pool.retain(|buffer| buffer.len() == bins);
        resize_vecdeque(&mut self.pcm_buffer, window_size.saturating_mul(2).max(1));

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
        self.accumulated_offset = Duration::default();
        self.pending_reset = true;
    }

    fn ensure_fft_capacity(&mut self) {
        let expected_window = self.config.fft_size;
        let expected_fft = expected_window.saturating_mul(self.config.zero_padding_factor.max(1));

        if self.window_size != expected_window
            || self.fft_size != expected_fft
            || self.real_buffer.len() != expected_fft
        {
            self.rebuild_fft();
            self.start_instant = None;
        }
    }

    fn process_ready_windows(&mut self) -> Vec<SpectrogramColumn> {
        self.output_columns_buffer.clear();

        let window_size = self.window_size;
        let fft_size = self.fft_size;
        let hop = self.config.hop_size;
        if window_size == 0 || fft_size == 0 || hop == 0 {
            return std::mem::take(&mut self.output_columns_buffer);
        }

        let sample_rate = self.config.sample_rate;
        let bins = fft_size / 2 + 1;
        let reassignment_enabled = self.config.use_reassignment && sample_rate > f32::EPSILON;
        let reassignment_bin_limit = if self.config.reassignment_low_bin_limit == 0 {
            bins
        } else {
            self.config.reassignment_low_bin_limit.min(bins)
        };
        let synchro_active = reassignment_enabled && self.synchro.is_active();

        let Some(start_instant) = self.start_instant else {
            return std::mem::take(&mut self.output_columns_buffer);
        };
        let center_offset = duration_from_samples((window_size / 2) as u64, sample_rate);

        let estimated_count = self.pcm_buffer.len() / hop;
        if estimated_count > 1 {
            self.output_columns_buffer.reserve(estimated_count);
        }

        while self.pcm_buffer.len() >= window_size {
            {
                let window_input = &mut self.real_buffer[..window_size];
                copy_from_deque(window_input, &self.pcm_buffer);
                crate::util::audio::remove_dc(window_input);
            }

            if reassignment_enabled {
                for i in 0..window_size {
                    self.reassignment.derivative_buffer[i] =
                        self.real_buffer[i] * self.reassignment.derivative_window[i];
                    self.reassignment.time_weighted_buffer[i] =
                        self.real_buffer[i] * self.reassignment.time_weighted_window[i];
                }
                self.reassignment.derivative_buffer[window_size..fft_size].fill(0.0);
                self.reassignment.time_weighted_buffer[window_size..fft_size].fill(0.0);
            }

            crate::util::audio::apply_window(
                &mut self.real_buffer[..window_size],
                self.window.as_ref(),
            );

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
                        &mut self.reassignment.derivative_buffer,
                        &mut self.reassignment.derivative_spectrum,
                        &mut self.scratch_buffer,
                    )
                    .expect("derivative-window FFT");

                self.fft
                    .process_with_scratch(
                        &mut self.reassignment.time_weighted_buffer,
                        &mut self.reassignment.time_weighted_spectrum,
                        &mut self.scratch_buffer,
                    )
                    .expect("time-weighted-window FFT");
            }

            for i in 0..bins {
                let complex = self.spectrum_buffer[i];
                let power =
                    (complex.re * complex.re + complex.im * complex.im) * self.bin_normalization[i];
                self.magnitude_buffer[i] = power_to_db(power);
            }

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

            self.output_columns_buffer.push(SpectrogramColumn {
                timestamp,
                magnitudes_db: magnitudes,
                reassigned,
                synchro_magnitudes_db: synchro_magnitudes,
            });

            let discard = hop.min(self.pcm_buffer.len());
            self.pcm_buffer.drain(..discard);
            let drained_samples = discard as u64;
            self.buffer_start_index += drained_samples;
            self.accumulated_offset += duration_from_samples(drained_samples, sample_rate);
        }

        std::mem::take(&mut self.output_columns_buffer)
    }

    fn compute_reassigned_samples(
        &mut self,
        sample_rate: f32,
        fft_size: usize,
        reassignment_bin_limit: usize,
        synchro_active: bool,
    ) -> Option<Arc<[ReassignedSample]>> {
        let power_floor = self.reassignment.power_floor_linear;
        let bin_hz = sample_rate / fft_size as f32;
        let nyquist = sample_rate * 0.5;
        // Scale derivative to Hz: delta_f = (f_s / 2*pi) * Im{X_Dh * X*} / |X|^2
        let inv_two_pi = sample_rate / (core::f32::consts::TAU);
        let max_index = (reassignment_bin_limit - 1) as f32;
        let inv_bin_hz = (fft_size as f32) / sample_rate;

        // Maximum allowed frequency correction in Hz (default to one bin width)
        let max_correction_hz = if self.config.reassignment_max_correction_hz > 0.0 {
            self.config.reassignment_max_correction_hz
        } else {
            bin_hz // Default: allow up to one bin of correction
        };

        let reassigned_power = &mut self.reassignment.power_buffer[..reassignment_bin_limit];
        reassigned_power.fill(0.0);

        if synchro_active {
            self.synchro.reset_power();
        }

        let spectrum = &self.spectrum_buffer[..reassignment_bin_limit];
        let derivative_spectrum = &self.reassignment.derivative_spectrum[..reassignment_bin_limit];
        let time_weighted_spectrum =
            &self.reassignment.time_weighted_spectrum[..reassignment_bin_limit];
        let bin_norm = &self.bin_normalization[..reassignment_bin_limit];
        let energy_scale = &self.energy_normalization[..reassignment_bin_limit];

        let samples = &mut self.reassignment.sample_cache;
        samples.clear();

        for (k, ((&base, &cross), ((&time_cross, &bin_norm_k), &energy_scale_k))) in spectrum
            .iter()
            .zip(derivative_spectrum.iter())
            .zip(
                time_weighted_spectrum
                    .iter()
                    .zip(bin_norm.iter())
                    .zip(energy_scale.iter()),
            )
            .enumerate()
        {
            let k_f32 = k as f32;

            let power = base.re * base.re + base.im * base.im;
            let display_power = power * bin_norm_k;

            if display_power < power_floor || energy_scale_k <= 0.0 {
                continue;
            }

            let inv_power = 1.0 / power;

            // Instantaneous frequency: delta_w = -Im{X_Dh * X*} / |X|^2
            let delta_omega = -(cross.im * base.re - cross.re * base.im) * inv_power;
            if !delta_omega.is_finite() {
                continue;
            }

            let freq_correction_hz = delta_omega * inv_two_pi;
            if freq_correction_hz.abs() > max_correction_hz {
                continue;
            }

            let freq_hz = k_f32.mul_add(bin_hz, freq_correction_hz);
            if freq_hz < 0.0 || freq_hz > nyquist {
                continue;
            }

            // Group delay: delta_t = Re{X_Th * X*} / |X|^2
            let delta_tau = (time_cross.re * base.re + time_cross.im * base.im) * inv_power;
            let group_delay_samples = if delta_tau.is_finite() {
                delta_tau
            } else {
                0.0
            };

            let freq_position = (freq_hz * inv_bin_hz).min(max_index);
            let lower = freq_position as usize;
            let frac = freq_position - lower as f32;
            let energy_power = power * energy_scale_k;

            let lower_contrib = energy_power * (1.0 - frac);
            reassigned_power[lower] += lower_contrib;

            let upper = lower + 1;
            if upper < reassignment_bin_limit {
                reassigned_power[upper] += energy_power * frac;
            }

            if synchro_active {
                self.synchro.accumulate(freq_hz as f64, display_power);
            }

            if samples.len() < MAX_REASSIGNMENT_SAMPLES {
                samples.push(ReassignedSample {
                    frequency_hz: freq_hz,
                    group_delay_samples,
                    magnitude_db: power_to_db(display_power),
                });
            }
        }

        for (((&energy_power, &energy_scale_k), &bin_norm_k), magnitude) in reassigned_power
            .iter()
            .zip(energy_scale.iter())
            .zip(bin_norm.iter())
            .zip(self.magnitude_buffer[..reassignment_bin_limit].iter_mut())
        {
            let normalized = if energy_scale_k > f32::EPSILON {
                energy_power * bin_norm_k / energy_scale_k
            } else {
                0.0
            };

            *magnitude = power_to_db(normalized);
        }

        if synchro_active {
            self.synchro.finalize_magnitudes();
        }

        (!samples.is_empty()).then(|| Arc::from(samples.as_slice()))
    }

    #[inline]
    fn apply_magnitude_post_processing(&mut self, bins: usize) {
        if bins == 0 {
            return;
        }

        let bin_hz = if self.config.sample_rate > 0.0 && self.fft_size > 0 {
            self.config.sample_rate / self.fft_size as f32
        } else {
            0.0
        };

        // Temporal smoothing
        let needs_temporal_weights = bin_hz > 0.0
            && self.config.temporal_smoothing > 0.0
            && self.config.temporal_smoothing_max_hz > 0.0
            && self.config.temporal_smoothing_blend_hz > 0.0;

        if needs_temporal_weights {
            let max_hz = self.config.temporal_smoothing_max_hz;
            let blend_hz = self.config.temporal_smoothing_blend_hz;
            apply_temporal_smoothing_with_weights(
                &mut self.magnitude_buffer[..bins],
                &mut self.temporal_smoothing_buffer,
                self.config.temporal_smoothing,
                |idx| smoothing_weight((idx as f32) * bin_hz, max_hz, blend_hz),
            );
        } else if self.config.temporal_smoothing > 0.0 {
            apply_temporal_smoothing_with_weights(
                &mut self.magnitude_buffer[..bins],
                &mut self.temporal_smoothing_buffer,
                self.config.temporal_smoothing,
                |_| 1.0,
            );
        }

        // Frequency smoothing
        let radius = self.config.frequency_smoothing_radius;
        if radius == 0 {
            return;
        }

        let clamped_radius = radius.min(bins.saturating_sub(1));
        if clamped_radius == 0 {
            return;
        }

        if self.frequency_scratch_buffer.len() != bins {
            self.frequency_scratch_buffer.resize(bins, 0.0);
        }

        let magnitude_slice = &self.magnitude_buffer[..bins];

        let start_0 = 0;
        let end_0 = clamped_radius.min(bins);
        let mut window_sum = magnitude_slice[start_0..end_0].iter().copied().sum::<f32>();
        let mut window_count = (end_0 - start_0) as f32;
        self.frequency_scratch_buffer[0] = window_sum / window_count;

        for idx in 1..bins {
            let new_start = idx.saturating_sub(clamped_radius);
            let new_end = (idx + clamped_radius + 1).min(bins);
            let old_start = (idx - 1).saturating_sub(clamped_radius);
            let old_end = (idx - 1 + clamped_radius + 1).min(bins);

            for &val in magnitude_slice.iter().take(new_start).skip(old_start) {
                window_sum -= val;
            }

            for &val in magnitude_slice.iter().take(new_end).skip(old_end) {
                window_sum += val;
            }

            window_count = (new_end - new_start) as f32;
            self.frequency_scratch_buffer[idx] = window_sum / window_count;
        }

        let needs_freq_weights = bin_hz > 0.0
            && self.config.frequency_smoothing_max_hz > 0.0
            && self.config.frequency_smoothing_blend_hz > f32::EPSILON;

        if needs_freq_weights {
            let max_hz = self.config.frequency_smoothing_max_hz;
            let blend_hz = self.config.frequency_smoothing_blend_hz;

            for idx in 0..bins {
                let freq = (idx as f32) * bin_hz;
                let weight = smoothing_weight(freq, max_hz, blend_hz);
                let original = self.magnitude_buffer[idx];
                let smoothed = self.frequency_scratch_buffer[idx];

                self.magnitude_buffer[idx] = weight.mul_add(smoothed - original, original);
            }
        } else {
            self.magnitude_buffer[..bins].copy_from_slice(&self.frequency_scratch_buffer[..bins]);
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
            self.config.sample_rate = block.sample_rate;
            self.rebuild_fft();
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
                synchro_bins_hz: self.synchro.bins_arc().cloned(),
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

        self.synchro.reset_all();
    }
}

impl SpectrogramProcessor {
    #[inline]
    fn mixdown_interleaved(&mut self, samples: &[f32], channels: usize) {
        crate::util::audio::mixdown_into_deque(&mut self.pcm_buffer, samples, channels);
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

    if let Some(last) = pool.last()
        && last.len() == bins
    {
        return pool.pop().unwrap();
    }

    let pos = pool.iter().rposition(|arc| arc.len() == bins);

    if let Some(idx) = pos {
        pool.swap_remove(idx)
    } else {
        Arc::from(vec![default; bins])
    }
}

fn compute_derivative_window(kind: WindowKind, window: &[f32]) -> Vec<f32> {
    match kind {
        WindowKind::PlanckBessel { epsilon, beta } => {
            compute_planck_bessel_derivative(window, epsilon, beta)
        }
        _ => compute_numeric_derivative(window),
    }
}

/// Time-weighted window h_T[n] = (n - center) * h[n] for group delay estimation.
fn compute_time_weighted_window(window: &[f32]) -> Vec<f32> {
    if window.is_empty() {
        return Vec::new();
    }

    let len = window.len();
    let center = (len - 1) as f32 * 0.5;

    (0..len)
        .map(|n| {
            let time_weight = n as f32 - center;
            time_weight * window[n]
        })
        .collect()
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

#[inline(always)]
fn smoothing_weight(freq_hz: f32, max_hz: f32, blend_hz: f32) -> f32 {
    if max_hz <= 0.0 || freq_hz <= max_hz {
        return 1.0;
    }

    let above_max = freq_hz - max_hz;

    if blend_hz <= f32::EPSILON || above_max >= blend_hz {
        return 0.0;
    }

    (-above_max).mul_add(1.0 / blend_hz, 1.0)
}

#[inline]
fn apply_temporal_smoothing_with_weights(
    buffer: &mut [f32],
    history: &mut Vec<f32>,
    smoothing: f32,
    mut weight_for_bin: impl FnMut(usize) -> f32,
) {
    let len = buffer.len();
    if len == 0 {
        history.clear();
        return;
    }

    let smoothing = smoothing.clamp(0.0, 0.999);

    if smoothing <= 0.0 {
        if history.len() != len {
            history.resize(len, DB_FLOOR);
        }
        history.copy_from_slice(buffer);
        return;
    }

    if history.len() != len {
        history.resize(len, DB_FLOOR);
    }

    for idx in 0..len {
        let current = buffer[idx];
        let prev = history[idx];

        if prev <= DB_FLOOR {
            history[idx] = current;
            buffer[idx] = current;
            continue;
        }

        let weight = weight_for_bin(idx);

        let result = if weight > 0.0 {
            let alpha = (smoothing * weight).min(0.999);
            (prev - current).mul_add(alpha, current)
        } else {
            current
        };

        history[idx] = result;
        buffer[idx] = result;
    }
}

impl Reconfigurable<SpectrogramConfig> for SpectrogramProcessor {
    fn update_config(&mut self, config: SpectrogramConfig) {
        let previous = self.config;

        self.config = config;
        Self::normalize_config(&mut self.config);

        let fft_related_changed = previous.fft_size != self.config.fft_size
            || previous.zero_padding_factor != self.config.zero_padding_factor
            || previous.window != self.config.window
            || (previous.sample_rate - self.config.sample_rate).abs() > f32::EPSILON;

        if fft_related_changed {
            self.rebuild_fft();
            self.start_instant = None;
            return;
        }

        if previous.history_length != self.config.history_length {
            let mut evicted = std::mem::take(&mut self.evicted_columns);
            self.history
                .set_capacity(self.config.history_length, &mut evicted);
            evicted
                .drain(..)
                .for_each(|column| self.recycle_column(column));
            self.evicted_columns = evicted;
        }

        if previous.reassignment_power_floor_db != self.config.reassignment_power_floor_db {
            self.reassignment.power_floor_linear =
                db_to_power(self.config.reassignment_power_floor_db);
        }

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
            self.synchro.reset_all();
            self.clear_history();
        }

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

    fn sine_samples(freq: f32, rate: f32, frames: usize) -> Vec<f32> {
        (0..frames)
            .map(|n| (core::f32::consts::TAU * freq * n as f32 / rate).sin())
            .collect()
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

    fn unwrap_update(result: ProcessorUpdate<SpectrogramUpdate>) -> SpectrogramUpdate {
        match result {
            ProcessorUpdate::Snapshot(u) => u,
            ProcessorUpdate::None => panic!("expected snapshot"),
        }
    }

    #[test]
    fn detects_sine_frequency_peak() {
        let config = SpectrogramConfig {
            fft_size: 1024,
            hop_size: 512,
            history_length: 8,
            window: WindowKind::Hann,
            zero_padding_factor: 1,
            ..test_config_base()
        };
        let mut processor = SpectrogramProcessor::new(config);
        let bin_hz = config.sample_rate / config.fft_size as f32;
        let target_bin = 200usize;
        let freq = target_bin as f32 * bin_hz;
        let block = make_block(
            sine_samples(freq, config.sample_rate, config.fft_size * 2),
            1,
            config.sample_rate,
        );
        let update = unwrap_update(processor.process_block(&block));

        let last = update.new_columns.last().unwrap();
        let (max_index, &peak_db) = last
            .magnitudes_db
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assert_eq!(max_index, target_bin);
        assert!(
            peak_db > -1.5 && peak_db < 2.0,
            "expected near 0 dBFS peak, saw {peak_db}"
        );
    }

    #[test]
    fn history_respects_limit() {
        let config = SpectrogramConfig {
            history_length: 4,
            ..test_config_base()
        };
        let mut processor = SpectrogramProcessor::new(config);
        let block = make_block(vec![0.0f32; config.fft_size * 4], 1, config.sample_rate);
        let _ = processor.process_block(&block);
        assert!(processor.history.slots.len() <= config.history_length);
    }

    #[test]
    fn reassignment_emits_samples() {
        let config = SpectrogramConfig {
            fft_size: 2048,
            hop_size: 512,
            history_length: 4,
            use_reassignment: true,
            reassignment_low_bin_limit: 512,
            ..test_config_base()
        };
        let mut processor = SpectrogramProcessor::new(config);
        let block = make_block(
            sine_samples(110.0, config.sample_rate, config.fft_size * 2),
            1,
            config.sample_rate,
        );
        let update = unwrap_update(processor.process_block(&block));
        let sample_count = update
            .new_columns
            .last()
            .and_then(|c| c.reassigned.as_ref())
            .map(|s| s.len())
            .unwrap_or(0);
        assert!(sample_count > 0);
    }

    #[test]
    fn reassignment_concentrates_energy_at_fundamental() {
        let config = SpectrogramConfig {
            fft_size: 4096,
            hop_size: 256,
            history_length: 4,
            use_reassignment: true,
            reassignment_low_bin_limit: 0,
            ..test_config_base()
        };
        let mut processor = SpectrogramProcessor::new(config);
        let freq_hz = 130.812_78_f32;
        let block = make_block(
            sine_samples(freq_hz, config.sample_rate, config.fft_size * 3),
            1,
            config.sample_rate,
        );
        let update = unwrap_update(processor.process_block(&block));
        let samples = update
            .new_columns
            .last()
            .and_then(|c| c.reassigned.as_ref())
            .expect("reassignment samples");

        let (total, off_harmonic): (f32, f32) = samples
            .iter()
            .filter(|s| s.frequency_hz.is_finite() && s.magnitude_db > DB_FLOOR)
            .map(|s| {
                let p = db_to_power(s.magnitude_db);
                let d = ((s.frequency_hz / freq_hz).round() - s.frequency_hz / freq_hz).abs();
                (p, if d > 0.05 { p } else { 0.0 })
            })
            .fold((0.0, 0.0), |(t, o), (p, op)| (t + p, o + op));
        assert!(total > 0.0 && off_harmonic / total < 0.02, "excess leakage");
    }

    #[test]
    fn reassignment_tracks_low_frequency_precisely() {
        let config = SpectrogramConfig {
            fft_size: 4096,
            hop_size: 512,
            history_length: 4,
            use_reassignment: true,
            reassignment_low_bin_limit: 256,
            ..test_config_base()
        };
        let mut processor = SpectrogramProcessor::new(config);
        let freq_hz = 58.5_f32;
        let block = make_block(
            sine_samples(freq_hz, config.sample_rate, config.fft_size * 2),
            1,
            config.sample_rate,
        );
        let update = unwrap_update(processor.process_block(&block));
        let column = update.new_columns.last().unwrap();
        let reassigned = column.reassigned.as_ref().unwrap();
        let peak_sample = reassigned
            .iter()
            .max_by(|a, b| a.magnitude_db.partial_cmp(&b.magnitude_db).unwrap())
            .unwrap();
        assert!(
            (peak_sample.frequency_hz - freq_hz).abs() < 0.75 && peak_sample.magnitude_db > -3.0
        );
    }

    #[test]
    fn frequency_smoothing_blurs_energy_across_bins() {
        let mut cfg = test_config_base();
        cfg.fft_size = 1024;
        cfg.hop_size = 512;
        cfg.history_length = 4;
        cfg.zero_padding_factor = 1;
        cfg.use_reassignment = false;
        cfg.use_synchrosqueezing = false;
        let freq_hz = 1200.0f32;
        let samples = sine_samples(freq_hz, cfg.sample_rate, cfg.fft_size * 2);

        let mut unsmoothed = SpectrogramProcessor::new(cfg);
        let unsm_col = unwrap_update(unsmoothed.process_block(&make_block(
            samples.clone(),
            1,
            cfg.sample_rate,
        )))
        .new_columns
        .pop()
        .unwrap();

        cfg.frequency_smoothing_radius = 12;
        cfg.frequency_smoothing_max_hz = cfg.sample_rate * 0.5;
        let mut smoothed = SpectrogramProcessor::new(cfg);
        let sm_col =
            unwrap_update(smoothed.process_block(&make_block(samples, 1, cfg.sample_rate)))
                .new_columns
                .pop()
                .unwrap();

        let (peak_idx, &unsm_peak) = unsm_col
            .magnitudes_db
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assert!(sm_col.magnitudes_db[peak_idx] < unsm_peak - 6.0);
    }

    #[test]
    fn frequency_smoothing_softens_synchro_peak() {
        let mut cfg = test_config_base();
        cfg.fft_size = 2048;
        cfg.hop_size = 512;
        cfg.history_length = 4;
        cfg.zero_padding_factor = 1;
        cfg.reassignment_low_bin_limit = 0;
        cfg.synchrosqueezing_bin_count = 256;
        cfg.frequency_smoothing_max_hz = cfg.sample_rate * 0.5;
        let freq_hz = 1400.0f32;
        let samples = sine_samples(freq_hz, cfg.sample_rate, cfg.fft_size * 3);

        let mut unsmoothed = SpectrogramProcessor::new(cfg);
        let unsm_update = unwrap_update(unsmoothed.process_block(&make_block(
            samples.clone(),
            1,
            cfg.sample_rate,
        )));

        cfg.frequency_smoothing_radius = 10;
        let mut smoothed = SpectrogramProcessor::new(cfg);
        let sm_update =
            unwrap_update(smoothed.process_block(&make_block(samples, 1, cfg.sample_rate)));

        let bins = unsm_update.synchro_bins_hz.as_ref().unwrap();
        let unsm_col = unsm_update
            .new_columns
            .last()
            .and_then(|c| c.synchro_magnitudes_db.as_ref())
            .unwrap();
        let sm_col = sm_update
            .new_columns
            .last()
            .and_then(|c| c.synchro_magnitudes_db.as_ref())
            .unwrap();
        let target_idx = bins
            .iter()
            .enumerate()
            .min_by(|a, b| {
                (a.1 - freq_hz)
                    .abs()
                    .partial_cmp(&(b.1 - freq_hz).abs())
                    .unwrap()
            })
            .unwrap()
            .0;
        assert!(sm_col[target_idx] < unsm_col[target_idx] - 3.0);
    }

    #[test]
    fn zero_padding_expands_fft_resolution() {
        let config = SpectrogramConfig {
            fft_size: 1024,
            hop_size: 256,
            history_length: 4,
            window: WindowKind::Hann,
            zero_padding_factor: 4,
            ..test_config_base()
        };
        let mut processor = SpectrogramProcessor::new(config);
        let block = make_block(
            sine_samples(440.0, config.sample_rate, config.fft_size * 2),
            1,
            config.sample_rate,
        );
        let update = unwrap_update(processor.process_block(&block));
        assert_eq!(update.fft_size, config.fft_size * 4);
        assert_eq!(
            update.new_columns.last().unwrap().magnitudes_db.len(),
            update.fft_size / 2 + 1
        );
    }

    #[test]
    fn mel_conversions_are_invertible() {
        for &hz in &[20.0, 100.0, 440.0, 1000.0, 4000.0, 10000.0] {
            assert!(
                (hz - mel_to_hz(hz_to_mel(hz))).abs() < 0.01,
                "Failed roundtrip for {} Hz",
                hz
            );
        }
    }

    #[test]
    fn reassignment_2d_with_group_delay() {
        let config = SpectrogramConfig {
            fft_size: 2048,
            hop_size: 512,
            history_length: 4,
            use_reassignment: true,
            reassignment_low_bin_limit: 0,
            zero_padding_factor: 1,
            ..test_config_base()
        };
        let mut processor = SpectrogramProcessor::new(config);

        let hann = WindowKind::Hann.coefficients(1024);
        let derivative = compute_derivative_window(WindowKind::Hann, &hann);
        assert!(derivative.iter().sum::<f32>().abs() < 0.01);
        let time_weighted = compute_time_weighted_window(&hann);
        assert!(time_weighted.iter().sum::<f32>().abs() < 0.1);
        let odd_tw = compute_time_weighted_window(&WindowKind::Hann.coefficients(1023));
        assert!(odd_tw[511].abs() < 1e-10);

        let bin_hz = config.sample_rate / config.fft_size as f32;
        let true_freq = 50.3 * bin_hz;
        let block = make_block(
            sine_samples(true_freq, config.sample_rate, config.fft_size * 2),
            1,
            config.sample_rate,
        );
        let update = unwrap_update(processor.process_block(&block));
        let peak = update
            .new_columns
            .last()
            .and_then(|c| c.reassigned.as_ref())
            .unwrap()
            .iter()
            .max_by(|a, b| a.magnitude_db.partial_cmp(&b.magnitude_db).unwrap())
            .unwrap();
        assert!((peak.frequency_hz - true_freq).abs() < 1.0);
        assert!(peak.group_delay_samples.abs() < config.fft_size as f32 * 0.1);
    }
}
