//! spectrogram DSP. reassignment! (Auger-Flandrin 1995).
//!
//! # References
//! 1. F. Auger and P. Flandrin, "Improving the readability of time-frequency and
//!    time-scale representations by the reassignment method", IEEE Trans. SP,
//!    vol. 43, no. 5, pp. 1068-1089, May 1995.
//!    Note: in our delta calculations the signs are inverted compared to the original
//!    paper, to match the more common convention.
//! 2. K. Kodera, R. Gendrin & C. de Villedary, "Analysis of time-varying signals
//!    with small BT values", IEEE Trans. ASSP, vol. 26, no. 1, pp. 64-76, Feb 1978.
//! 3. T. Oberlin, S. Meignen, V. Perrier, "Second-order synchrosqueezing transform
//!    or invertible reassignment? Towards ideal time-frequency representations",
//!    IEEE Trans. SP, vol. 63, no. 5, pp. 1335-1344, 2015.
//!    Note: we aren't implementing "true" SST, rather a simpler form of frequency
//!    correction based on second derivatives. *Our spectrogram is not invertible*
//!    by design.
//! 4. F. Auger et al., "Time-Frequency Reassignment and Synchrosqueezing: An
//!    Overview", IEEE Signal Processing Magazine, vol. 30, pp. 32-41, Nov 2013.

use super::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::util::audio::{
    DB_FLOOR, DEFAULT_SAMPLE_RATE, copy_from_deque, db_to_power, hz_to_mel, mel_to_hz, power_to_db,
};
use parking_lot::RwLock;
use realfft::{RealFftPlanner, RealToComplex};
use rustc_hash::FxHashMap;
use rustfft::num_complex::Complex32;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};
use wide::{CmpGe, CmpGt, CmpLe, CmpLt, f32x8};

const MAX_REASSIGNMENT_SAMPLES: usize = 8192;
pub const PLANCK_BESSEL_DEFAULT_EPSILON: f32 = 0.1;
pub const PLANCK_BESSEL_DEFAULT_BETA: f32 = 5.5;

/// SNR range (dB) over which confidence scales from 0 to 1.
const CONFIDENCE_SNR_RANGE_DB: f32 = 60.0;

/// Minimum confidence to prevent complete zeroing of valid bins.
const CONFIDENCE_FLOOR: f32 = 0.01;

/// Safety margin for max chirp rate; 2x accounts for estimation noise.
const CHIRP_SAFETY_MARGIN: f32 = 2.0;

const GAUSSIAN_KERNEL_3X3: [[f32; 3]; 3] = {
    const CENTER: f32 = 1.0; // exp(-0) = 1
    const EDGE: f32 = 0.324_652_5; // exp(-1) for d=1
    const CORNER: f32 = 0.105_399_2; // exp(-2) for d=sqrt(2)
    const SUM: f32 = CENTER + 4.0 * EDGE + 4.0 * CORNER;
    [
        [CORNER / SUM, EDGE / SUM, CORNER / SUM],
        [EDGE / SUM, CENTER / SUM, EDGE / SUM],
        [CORNER / SUM, EDGE / SUM, CORNER / SUM],
    ]
};

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
    pub reassignment_low_bin_limit: usize,
    pub zero_padding_factor: usize,
    pub display_bin_count: usize,
    pub display_min_hz: f32,
    pub reassignment_max_correction_hz: f32,
    pub reassignment_max_time_hops: f32,
}

impl Default for SpectrogramConfig {
    fn default() -> Self {
        Self {
            sample_rate: DEFAULT_SAMPLE_RATE,
            fft_size: 4096,
            hop_size: 256,
            window: WindowKind::Blackman,
            frequency_scale: FrequencyScale::default(),
            history_length: 240,
            use_reassignment: true,
            reassignment_power_floor_db: -80.0,
            reassignment_low_bin_limit: 0,
            zero_padding_factor: 4,
            display_bin_count: 1024,
            display_min_hz: 20.0,
            reassignment_max_correction_hz: 0.0,
            reassignment_max_time_hops: 2.0,
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

        if let Some(existing) = self.entries.read().get(&key) {
            return Arc::clone(existing);
        }

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
                        + y * (-0.010_315_55
                            + y * (0.022_829_67
                                + y * (-0.028_953_12 + y * (0.017_876_54 - y * 0.004_200_59)))))));
        let ans = poly * ax.exp() / ax.sqrt();
        if x < 0.0 { -ans } else { ans }
    }
}

/// Per-bin reassignment output for testing; UI uses the 2D grid magnitudes instead.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct ReassignedSample {
    pub frequency_hz: f32,
    pub group_delay_samples: f32,
    pub magnitude_db: f32,
}

struct ReassignmentBuffers {
    derivative_window: Vec<f32>,
    time_weighted_window: Vec<f32>,
    second_derivative_window: Vec<f32>,
    derivative_buffer: Vec<f32>,
    time_weighted_buffer: Vec<f32>,
    second_derivative_buffer: Vec<f32>,
    derivative_spectrum: Vec<Complex32>,
    time_weighted_spectrum: Vec<Complex32>,
    second_derivative_spectrum: Vec<Complex32>,
    sample_cache: Vec<ReassignedSample>,
    power_floor_linear: f32,
    /// Window time spread (sigma_t) in samples.
    sigma_t: f32,
    /// Precomputed 1/sigma_t for weight normalization.
    inv_sigma_t: f32,
    /// Max chirp = pi / sigma_t^2 in Hz/sample; beyond this estimates are unreliable.
    max_chirp_hz_per_sample: f32,
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
        let derivative_window = compute_derivative_window(window_kind, window);
        let second_derivative_window = compute_second_derivative_window(&derivative_window);

        // Window time spread from Gabor uncertainty; limits chirp estimation accuracy.
        let sigma_t = compute_window_sigma_t(window);
        let inv_sigma_t = 1.0 / sigma_t;
        // max_chirp = pi / sigma_t^2 (rad/sample^2), converted to Hz/sample.
        let max_chirp_hz_per_sample = 0.5 / (sigma_t * sigma_t);

        Self {
            derivative_window,
            time_weighted_window: compute_time_weighted_window(window),
            second_derivative_window,
            derivative_buffer: vec![0.0; fft_size],
            time_weighted_buffer: vec![0.0; fft_size],
            second_derivative_buffer: vec![0.0; fft_size],
            derivative_spectrum: fft.make_output_vec(),
            time_weighted_spectrum: fft.make_output_vec(),
            second_derivative_spectrum: fft.make_output_vec(),
            sample_cache: Vec::with_capacity(bins >> 4),
            power_floor_linear: db_to_power(power_floor_db),
            sigma_t,
            inv_sigma_t,
            max_chirp_hz_per_sample,
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
        self.second_derivative_window = compute_second_derivative_window(&self.derivative_window);
        self.derivative_buffer.resize(fft_size, 0.0);
        self.time_weighted_buffer.resize(fft_size, 0.0);
        self.second_derivative_buffer.resize(fft_size, 0.0);
        self.derivative_spectrum = fft.make_output_vec();
        self.time_weighted_spectrum = fft.make_output_vec();
        self.second_derivative_spectrum = fft.make_output_vec();
        self.power_floor_linear = db_to_power(power_floor_db);

        self.sigma_t = compute_window_sigma_t(window);
        self.inv_sigma_t = 1.0 / self.sigma_t;
        self.max_chirp_hz_per_sample = 0.5 / (self.sigma_t * self.sigma_t);
    }
}

#[derive(Clone, Copy, Default)]
struct FrequencyScaleParams {
    min_hz: f32,
    max_hz: f32,
    log_min: f32,
    inv_log_range: f32,
    mel_min: f32,
    inv_mel_range: f32,
    inv_linear_range: f32,
    bin_count_minus_1: f32,
}

impl FrequencyScaleParams {
    fn compute(min_hz: f32, max_hz: f32, bin_count: usize) -> Self {
        if bin_count == 0 || max_hz <= min_hz {
            return Self::default();
        }

        let (log_min, inv_log_range) = {
            let min = min_hz.ln();
            let max = max_hz.ln();
            let range = (max - min).max(1.0e-9);
            (min, 1.0 / range)
        };

        let (mel_min, inv_mel_range) = {
            let min = hz_to_mel(min_hz);
            let max = hz_to_mel(max_hz);
            let range = (max - min).max(1.0e-9);
            (min, 1.0 / range)
        };

        let inv_linear_range = if (max_hz - min_hz) > f32::EPSILON {
            1.0 / (max_hz - min_hz)
        } else {
            0.0
        };

        Self {
            min_hz,
            max_hz,
            log_min,
            inv_log_range,
            mel_min,
            inv_mel_range,
            inv_linear_range,
            bin_count_minus_1: (bin_count - 1) as f32,
        }
    }

    #[inline(always)]
    fn hz_to_bin_simd(&self, freq_hz: f32x8, scale: FrequencyScale) -> f32x8 {
        let normalized = match scale {
            FrequencyScale::Linear => (freq_hz - self.min_hz) * self.inv_linear_range,
            FrequencyScale::Logarithmic => (freq_hz.ln() - self.log_min) * self.inv_log_range,
            FrequencyScale::Mel => {
                let one = f32x8::splat(1.0);
                let seven_hundred = f32x8::splat(700.0);
                let two_five_nine_five = f32x8::splat(2595.0);
                let log10_e = f32x8::splat(core::f32::consts::LOG10_E);

                let val = one + freq_hz / seven_hundred;
                let mel = two_five_nine_five * (val.ln() * log10_e);
                (mel - self.mel_min) * self.inv_mel_range
            }
        };

        // No clamping here - caller is responsible for filtering out-of-range frequencies
        (f32x8::splat(1.0) - normalized) * self.bin_count_minus_1
    }
}

struct Reassignment2DGrid {
    display_bins: usize,
    max_time_hops: usize,
    hop_size: usize,
    power_grid: Vec<f32>,
    center_column: usize,
    column_count: usize,
    output_buffer: Vec<f32>,
    magnitude_buffer: Vec<f32>,
    scale: FrequencyScale,
    scale_params: FrequencyScaleParams,
    bin_frequencies_hz: Arc<[f32]>,
    min_hz: f32,
    max_hz: f32,
    enabled: bool,
}

impl Reassignment2DGrid {
    fn new(config: &SpectrogramConfig, sample_rate: f32) -> Self {
        let mut grid = Self {
            display_bins: 0,
            max_time_hops: 1,
            hop_size: config.hop_size,
            power_grid: Vec::new(),
            center_column: 1,
            column_count: 3,
            output_buffer: Vec::new(),
            magnitude_buffer: Vec::new(),
            scale: config.frequency_scale,
            scale_params: FrequencyScaleParams::default(),
            bin_frequencies_hz: Arc::from([]),
            min_hz: 0.0,
            max_hz: 0.0,
            enabled: false,
        };
        grid.reconfigure(config, sample_rate);
        grid
    }

    fn reconfigure(&mut self, config: &SpectrogramConfig, sample_rate: f32) {
        let enabled = config.use_reassignment && config.display_bin_count > 0 && sample_rate > 0.0;

        if !enabled {
            self.enabled = false;
            self.display_bins = 0;
            self.power_grid.clear();
            self.output_buffer.clear();
            self.magnitude_buffer.clear();
            self.bin_frequencies_hz = Arc::from([]);
            return;
        }

        let display_bins = config.display_bin_count.max(2);
        let max_hops = (config.reassignment_max_time_hops.ceil() as usize).max(1);
        let column_count = 2 * max_hops + 1;
        let hop_size = config.hop_size.max(1);

        let min_hz = config.display_min_hz.max(1.0).min(sample_rate * 0.5);
        let max_hz = (sample_rate * 0.5).max(min_hz * 1.001);
        let scale = config.frequency_scale;

        let freqs_changed = self.display_bins != display_bins
            || self.scale != scale
            || (self.min_hz - min_hz).abs() > f32::EPSILON
            || (self.max_hz - max_hz).abs() > f32::EPSILON;

        let size_changed = self.display_bins != display_bins || self.column_count != column_count;

        self.display_bins = display_bins;
        self.max_time_hops = max_hops;
        self.hop_size = hop_size;
        self.column_count = column_count;
        self.center_column = max_hops;
        self.scale = scale;
        self.min_hz = min_hz;
        self.max_hz = max_hz;
        self.enabled = true;

        if freqs_changed {
            self.scale_params = FrequencyScaleParams::compute(min_hz, max_hz, display_bins);
            self.bin_frequencies_hz =
                Self::compute_bin_frequencies(scale, display_bins, &self.scale_params);
        }

        if size_changed {
            let grid_size = display_bins * column_count;
            self.power_grid.resize(grid_size, 0.0);
            self.output_buffer.resize(display_bins, 0.0);
            self.magnitude_buffer.resize(display_bins, DB_FLOOR);
            self.reset();
        }
    }

    fn compute_bin_frequencies(
        scale: FrequencyScale,
        bin_count: usize,
        params: &FrequencyScaleParams,
    ) -> Arc<[f32]> {
        if bin_count == 0 {
            return Arc::from([]);
        }

        if bin_count == 1 {
            return Arc::from([params.min_hz]);
        }

        let log_range = 1.0 / params.inv_log_range;
        let mel_range = 1.0 / params.inv_mel_range;

        let mut freqs = Vec::with_capacity(bin_count);
        for idx in 0..bin_count {
            let t = idx as f32 / (bin_count as f32 - 1.0);
            let freq = match scale {
                FrequencyScale::Linear => params.min_hz + (params.max_hz - params.min_hz) * t,
                FrequencyScale::Logarithmic => (params.log_min + log_range * t).exp(),
                FrequencyScale::Mel => mel_to_hz(params.mel_min + mel_range * t),
            };
            freqs.push(freq);
        }
        freqs.reverse();
        Arc::from(freqs.into_boxed_slice())
    }

    fn reset(&mut self) {
        self.power_grid.fill(0.0);
        self.magnitude_buffer.fill(DB_FLOOR);
    }

    #[inline]
    fn is_enabled(&self) -> bool {
        self.enabled && self.display_bins > 0
    }

    fn bin_frequencies(&self) -> Option<&Arc<[f32]>> {
        self.is_enabled()
            .then_some(&self.bin_frequencies_hz)
            .filter(|b| !b.is_empty())
    }

    #[inline(always)]
    fn accumulate_simd(
        &mut self,
        freq_bin: f32x8,
        time_offset_samples: f32x8,
        power: f32x8,
        confidence: f32x8,
        mask: f32x8,
    ) {
        let masks: [f32; 8] = mask.to_array();

        let inv_hop = 1.0 / self.hop_size as f32;
        let max_offset = self.max_time_hops as f32;
        let center_col = self.center_column as f32;
        let display_bins = self.display_bins;
        let column_count = self.column_count;
        let w = display_bins;
        let grid = &mut self.power_grid;

        let freq_bins: [f32; 8] = freq_bin.to_array();
        let time_offsets: [f32; 8] = time_offset_samples.to_array();
        let powers: [f32; 8] = power.to_array();
        let confidences: [f32; 8] = confidence.to_array();

        // Pre-compute kernel weights (flattened for cache efficiency)
        const K00: f32 = GAUSSIAN_KERNEL_3X3[0][0];
        const K10: f32 = GAUSSIAN_KERNEL_3X3[1][0];
        const K20: f32 = GAUSSIAN_KERNEL_3X3[2][0];
        const K01: f32 = GAUSSIAN_KERNEL_3X3[0][1];
        const K11: f32 = GAUSSIAN_KERNEL_3X3[1][1];
        const K21: f32 = GAUSSIAN_KERNEL_3X3[2][1];
        const K02: f32 = GAUSSIAN_KERNEL_3X3[0][2];
        const K12: f32 = GAUSSIAN_KERNEL_3X3[1][2];
        const K22: f32 = GAUSSIAN_KERNEL_3X3[2][2];

        for i in 0..8 {
            if masks[i] == 0.0 {
                continue;
            }

            let fb = freq_bins[i];
            let val = powers[i] * confidences[i];

            if !(val > 0.0 && val.is_finite() && fb.is_finite()) {
                continue;
            }

            let time_hops = time_offsets[i] * inv_hop;
            let clamped = time_hops.clamp(-max_offset, max_offset);
            let tc_f = (clamped + center_col).round();
            let fc_f = fb.round();

            let fc = fc_f as i32;
            let tc = tc_f as i32;

            if fc >= 1
                && fc < (display_bins as i32 - 1)
                && tc >= 1
                && tc < (column_count as i32 - 1)
            {
                // Fast path: 3x3 kernel fully in bounds
                let base = (tc as usize) * w + (fc as usize);

                let v00 = val * K00;
                let v10 = val * K10;
                let v20 = val * K20;
                let v01 = val * K01;
                let v11 = val * K11;
                let v21 = val * K21;
                let v02 = val * K02;
                let v12 = val * K12;
                let v22 = val * K22;

                let r0 = base - w - 1;
                grid[r0] += v00;
                grid[r0 + 1] += v10;
                grid[r0 + 2] += v20;

                let r1 = base - 1;
                grid[r1] += v01;
                grid[r1 + 1] += v11;
                grid[r1 + 2] += v21;

                let r2 = base + w - 1;
                grid[r2] += v02;
                grid[r2 + 1] += v12;
                grid[r2 + 2] += v22;
            } else if fc >= 0 && fc < display_bins as i32 && tc >= 0 && tc < column_count as i32 {
                for ky in 0..3i32 {
                    let t = tc + ky - 1;
                    if t < 0 || t >= column_count as i32 {
                        continue;
                    }
                    let row_base = (t as usize) * w;

                    for kx in 0..3i32 {
                        let f = fc + kx - 1;
                        if f >= 0 && f < display_bins as i32 {
                            let idx = row_base + f as usize;
                            grid[idx] += val * GAUSSIAN_KERNEL_3X3[kx as usize][ky as usize];
                        }
                    }
                }
            }
        }
    }

    fn advance(&mut self, energy_scale: &[f32], bin_norm: &[f32]) {
        self.output_buffer
            .copy_from_slice(&self.power_grid[..self.display_bins]);
        self.power_grid.rotate_left(self.display_bins);
        let new_right_start = (self.column_count - 1) * self.display_bins;
        self.power_grid[new_right_start..].fill(0.0);

        let len = self.magnitude_buffer.len().min(self.output_buffer.len());
        for i in 0..len {
            let power_i = self.output_buffer[i];
            let energy_scale_i = energy_scale
                .get(i)
                .or(energy_scale.get(1))
                .copied()
                .unwrap_or(1.0);
            let bin_norm_i = bin_norm.get(i).or(bin_norm.get(1)).copied().unwrap_or(1.0);

            let normalized = if energy_scale_i > f32::EPSILON {
                power_i * bin_norm_i / energy_scale_i
            } else {
                0.0
            };
            self.magnitude_buffer[i] = power_to_db(normalized);
        }
    }

    fn magnitudes(&self) -> &[f32] {
        &self.magnitude_buffer
    }

    fn display_bin_count(&self) -> usize {
        self.display_bins
    }
}

/// Output column with magnitude data per FFT frame.
#[derive(Debug, Clone)]
pub struct SpectrogramColumn {
    pub magnitudes_db: Arc<[f32]>,
    /// Reassignment samples for testing/debugging; UI uses magnitudes_db.
    #[cfg_attr(not(test), allow(dead_code))]
    pub reassigned: Option<Arc<[ReassignedSample]>>,
}

#[derive(Debug, Clone)]
pub struct SpectrogramUpdate {
    pub fft_size: usize,
    pub hop_size: usize,
    pub sample_rate: f32,
    pub frequency_scale: FrequencyScale,
    pub history_length: usize,
    pub reset: bool,
    pub display_bins_hz: Option<Arc<[f32]>>,
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
    reassignment_2d: Reassignment2DGrid,
    bin_normalization: Vec<f32>,
    energy_normalization: Vec<f32>,
    pcm_buffer: VecDeque<f32>,
    history: SpectrogramHistory,
    magnitude_pool: Vec<Arc<[f32]>>,
    evicted_columns: Vec<SpectrogramColumn>,
    pending_reset: bool,
    output_columns_buffer: Vec<SpectrogramColumn>,
}

impl SpectrogramProcessor {
    pub fn new(config: SpectrogramConfig) -> Self {
        let window_size = config.fft_size;
        let zero_padding = config.zero_padding_factor.max(1);
        let fft_size = window_size.saturating_mul(zero_padding);
        assert!(fft_size > 0, "FFT size must be greater than zero");

        let history_len = config.history_length;
        let bins = fft_size / 2 + 1;

        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(fft_size);
        let window = WindowCache::global().get(config.window, window_size);

        let reassignment = ReassignmentBuffers::new(
            config.window,
            window.as_ref(),
            &fft,
            fft_size,
            config.reassignment_power_floor_db,
        );

        let reassignment_2d = Reassignment2DGrid::new(&config, config.sample_rate);

        let spectrum_buffer = fft.make_output_vec();
        let scratch_buffer = fft.make_scratch_vec();
        let bin_normalization =
            crate::util::audio::compute_fft_bin_normalization(window.as_ref(), fft_size);
        let energy_normalization = Self::compute_energy_normalization(window.as_ref(), fft_size);

        Self {
            config,
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
            reassignment_2d,
            bin_normalization,
            energy_normalization,
            pcm_buffer: VecDeque::with_capacity(window_size.saturating_mul(2)),
            history: SpectrogramHistory::new(history_len),
            magnitude_pool: Vec::new(),
            evicted_columns: Vec::new(),
            pending_reset: true,
            output_columns_buffer: Vec::with_capacity(8),
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

        self.reassignment_2d
            .reconfigure(&self.config, self.config.sample_rate);

        self.bin_normalization =
            crate::util::audio::compute_fft_bin_normalization(self.window.as_ref(), fft_size);
        self.energy_normalization =
            Self::compute_energy_normalization(self.window.as_ref(), fft_size);

        let output_bins = if self.reassignment_2d.is_enabled() {
            self.reassignment_2d.display_bin_count()
        } else {
            bins
        };
        self.magnitude_pool
            .retain(|buffer| buffer.len() == output_bins);
        self.pcm_buffer
            .truncate(window_size.saturating_mul(2).max(1));

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
        let fft_bins = fft_size / 2 + 1;
        let reassignment_enabled = self.config.use_reassignment
            && sample_rate > f32::EPSILON
            && self.reassignment_2d.is_enabled();
        let reassignment_bin_limit = if self.config.reassignment_low_bin_limit == 0 {
            fft_bins
        } else {
            self.config.reassignment_low_bin_limit.min(fft_bins)
        };

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

                for i in 0..window_size {
                    self.reassignment.second_derivative_buffer[i] =
                        self.real_buffer[i] * self.reassignment.second_derivative_window[i];
                }
                self.reassignment.second_derivative_buffer[window_size..fft_size].fill(0.0);
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

                self.fft
                    .process_with_scratch(
                        &mut self.reassignment.second_derivative_buffer,
                        &mut self.reassignment.second_derivative_spectrum,
                        &mut self.scratch_buffer,
                    )
                    .expect("second-derivative-window FFT");
            }

            let (magnitudes, reassigned) = if reassignment_enabled {
                let reassigned_samples =
                    self.compute_reassigned_samples(sample_rate, fft_size, reassignment_bin_limit);

                let display_bins = self.reassignment_2d.display_bin_count();
                let mags = Self::fill_arc(
                    self.acquire_magnitude_storage(display_bins),
                    self.reassignment_2d.magnitudes(),
                );
                (mags, reassigned_samples)
            } else {
                for i in 0..fft_bins {
                    let complex = self.spectrum_buffer[i];
                    let power = (complex.re * complex.re + complex.im * complex.im)
                        * self.bin_normalization[i];
                    self.magnitude_buffer[i] = power_to_db(power);
                }
                let mags = Self::fill_arc(
                    self.acquire_magnitude_storage(fft_bins),
                    &self.magnitude_buffer[..fft_bins],
                );
                (mags, None)
            };

            let history_column = SpectrogramColumn {
                magnitudes_db: Arc::clone(&magnitudes),
                reassigned: reassigned.clone(),
            };
            if let Some(evicted) = self.history.push(history_column) {
                self.recycle_column(evicted);
            }

            self.output_columns_buffer.push(SpectrogramColumn {
                magnitudes_db: magnitudes,
                reassigned,
            });

            let discard = hop.min(self.pcm_buffer.len());
            self.pcm_buffer.drain(..discard);
        }

        std::mem::take(&mut self.output_columns_buffer)
    }

    fn compute_reassigned_samples(
        &mut self,
        sample_rate: f32,
        fft_size: usize,
        reassignment_bin_limit: usize,
    ) -> Option<Arc<[ReassignedSample]>> {
        let power_floor = self.reassignment.power_floor_linear;
        let bin_hz = sample_rate / fft_size as f32;
        let inv_two_pi = sample_rate / core::f32::consts::TAU;

        let max_correction_hz = if self.config.reassignment_max_correction_hz > 0.0 {
            self.config.reassignment_max_correction_hz
        } else {
            bin_hz
        };

        let spectrum = &self.spectrum_buffer[..reassignment_bin_limit];
        let derivative_spectrum = &self.reassignment.derivative_spectrum[..reassignment_bin_limit];
        let time_weighted_spectrum =
            &self.reassignment.time_weighted_spectrum[..reassignment_bin_limit];
        let second_derivative_spectrum =
            &self.reassignment.second_derivative_spectrum[..reassignment_bin_limit];
        let bin_norm = &self.bin_normalization[..reassignment_bin_limit];
        let energy_scale = &self.energy_normalization[..reassignment_bin_limit];

        let samples = &mut self.reassignment.sample_cache;
        samples.clear();

        let v_power_floor = f32x8::splat(power_floor);
        let v_bin_hz = f32x8::splat(bin_hz);
        let v_inv_two_pi = f32x8::splat(inv_two_pi);
        let v_max_correction = f32x8::splat(max_correction_hz);
        let v_zero = f32x8::splat(0.0);
        let v_one = f32x8::splat(1.0);
        let v_db_scale = f32x8::splat(4.342_944_8);
        let v_snr_range = f32x8::splat(CONFIDENCE_SNR_RANGE_DB);
        let v_point_five = f32x8::splat(0.5);
        let v_confidence_floor = f32x8::splat(CONFIDENCE_FLOOR);

        // Chirp limit from uncertainty principle: |c| <= pi/sigma_t^2 in rad/sample^2.
        let v_max_chirp =
            f32x8::splat(self.reassignment.max_chirp_hz_per_sample * CHIRP_SAFETY_MARGIN);
        let v_inv_sigma_t = f32x8::splat(self.reassignment.inv_sigma_t);

        let v_display_min_hz = f32x8::splat(self.reassignment_2d.min_hz);
        let v_display_max_hz = f32x8::splat(self.reassignment_2d.max_hz);

        let chunks = reassignment_bin_limit.div_ceil(8);
        let limit_f = f32x8::splat(reassignment_bin_limit as f32);

        for i in 0..chunks {
            let offset = i * 8;
            let k_indices = f32x8::new([
                (offset) as f32,
                (offset + 1) as f32,
                (offset + 2) as f32,
                (offset + 3) as f32,
                (offset + 4) as f32,
                (offset + 5) as f32,
                (offset + 6) as f32,
                (offset + 7) as f32,
            ]);

            let mask_limit = k_indices.simd_lt(limit_f);

            let (base_re, base_im) = load_complex_simd_safe(&spectrum[offset..]);
            let (cross_re, cross_im) = load_complex_simd_safe(&derivative_spectrum[offset..]);
            let (time_re, time_im) = load_complex_simd_safe(&time_weighted_spectrum[offset..]);
            let (second_re, second_im) =
                load_complex_simd_safe(&second_derivative_spectrum[offset..]);

            let bin_norm_v = load_f32_simd_safe(&bin_norm[offset..]);
            let energy_scale_v = load_f32_simd_safe(&energy_scale[offset..]);

            let power = base_re * base_re + base_im * base_im;
            let display_power = power * bin_norm_v;

            let mask_power =
                display_power.simd_ge(v_power_floor) & energy_scale_v.simd_gt(v_zero) & mask_limit;

            if mask_power.none() {
                continue;
            }

            let inv_power = v_one / power.max(f32x8::splat(f32::MIN_POSITIVE));

            let cross_prod_freq = cross_im * base_re - cross_re * base_im;
            let delta_omega = -cross_prod_freq * inv_power;

            let mut freq_correction_hz = delta_omega * v_inv_two_pi;

            let second_cross_freq = second_im * base_re - second_re * base_im;
            let d_omega_dt = -second_cross_freq * inv_power;
            let chirp_rate = d_omega_dt * v_inv_two_pi;

            let mask_chirp = chirp_rate.abs().simd_lt(v_max_chirp);

            // Weight normalized by magnitude: exp(-|S_tg| / (|S_g| * sigma_t))
            let time_cross_mag = (time_re * time_re + time_im * time_im).sqrt();
            let base_mag = power.sqrt().max(f32x8::splat(f32::MIN_POSITIVE));
            let normalized_time = time_cross_mag * v_inv_sigma_t / base_mag;
            let weight = (-normalized_time).exp().min(v_one);
            let correction = chirp_rate * weight * v_point_five;

            freq_correction_hz =
                mask_chirp.blend(freq_correction_hz + correction, freq_correction_hz);

            let mask_correction = freq_correction_hz.abs().simd_le(v_max_correction);

            let freq_hz = k_indices.mul_add(v_bin_hz, freq_correction_hz);
            let mask_freq = freq_hz.simd_ge(v_display_min_hz) & freq_hz.simd_lt(v_display_max_hz);

            let final_mask = mask_power & mask_correction & mask_freq;

            if final_mask.none() {
                continue;
            }

            let cross_prod_time = time_re * base_re + time_im * base_im;
            let delta_tau = -cross_prod_time * inv_power;

            // Confidence combines two indicators:
            // 1. SNR: bins near noise floor contribute less (unreliable phase)
            // 2. Phase coherence: large corrections indicate poor separability
            let snr_factor =
                ((display_power.ln() * v_db_scale) - (v_power_floor.ln() * v_db_scale)).max(v_zero)
                    / v_snr_range;
            let snr_confidence = snr_factor.min(v_one);

            let correction_ratio = freq_correction_hz.abs() / v_bin_hz;
            let phase_coherence = v_one - correction_ratio.min(v_one);

            let confidence = (snr_confidence * phase_coherence).max(v_confidence_floor);
            let energy_power = power * energy_scale_v;

            let freq_bins = self
                .reassignment_2d
                .scale_params
                .hz_to_bin_simd(freq_hz, self.reassignment_2d.scale);

            self.reassignment_2d.accumulate_simd(
                freq_bins,
                delta_tau,
                energy_power,
                confidence,
                final_mask.blend(v_one, v_zero),
            );

            if samples.len() < MAX_REASSIGNMENT_SAMPLES {
                let freq_hz_arr = freq_hz.to_array();
                let group_delay_arr = delta_tau.to_array();
                let display_power_arr = display_power.to_array();
                let valid_flags: [f32; 8] = final_mask.blend(v_one, v_zero).to_array();

                for j in 0..8 {
                    if valid_flags[j] > 0.0 {
                        samples.push(ReassignedSample {
                            frequency_hz: freq_hz_arr[j],
                            group_delay_samples: group_delay_arr[j],
                            magnitude_db: power_to_db(display_power_arr[j]),
                        });
                    }
                }
            }
        }

        self.reassignment_2d
            .advance(&self.energy_normalization, &self.bin_normalization);

        (!samples.is_empty()).then(|| Arc::from(samples.as_slice()))
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
            self.config.sample_rate = block.sample_rate;
            self.rebuild_fft();
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
                display_bins_hz: self.reassignment_2d.bin_frequencies().cloned(),
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

        self.pcm_buffer.clear();
        self.pending_reset = true;

        self.reassignment_2d.reset();
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

/// Time spread (sigma_t) of window in samples: sqrt(sum(t^2*g^2) / sum(g^2)).
/// Determines max chirp rate: |c_max| = pi / sigma_t^2.
fn compute_window_sigma_t(window: &[f32]) -> f32 {
    if window.is_empty() {
        return 1.0;
    }

    let len = window.len();
    let center = (len - 1) as f32 * 0.5;

    let mut sum_t2_g2 = 0.0f64;
    let mut sum_g2 = 0.0f64;

    for (n, &g) in window.iter().enumerate() {
        let t = n as f32 - center;
        let g2 = (g * g) as f64;
        sum_t2_g2 += (t * t) as f64 * g2;
        sum_g2 += g2;
    }

    if sum_g2 < f64::EPSILON {
        return 1.0;
    }

    let sigma_t_squared = sum_t2_g2 / sum_g2;
    (sigma_t_squared.sqrt() as f32).max(1.0)
}

/// Second derivative via 3-point central differences.
fn compute_second_derivative_window(derivative_window: &[f32]) -> Vec<f32> {
    if derivative_window.len() <= 2 {
        return vec![0.0; derivative_window.len()];
    }

    let len = derivative_window.len();
    (0..len)
        .map(|i| {
            if i == 0 {
                let h0 = derivative_window[0];
                let h1 = derivative_window[1];
                let h2 = derivative_window.get(2).copied().unwrap_or(h1);
                h2 - 2.0 * h1 + h0
            } else if i == len - 1 {
                let h_n1 = derivative_window[len - 1];
                let h_n2 = derivative_window[len - 2];
                let h_n3 = derivative_window
                    .get(len.saturating_sub(3))
                    .copied()
                    .unwrap_or(h_n2);
                h_n1 - 2.0 * h_n2 + h_n3
            } else {
                let prev = derivative_window[i - 1];
                let curr = derivative_window[i];
                let next = derivative_window[i + 1];
                next - 2.0 * curr + prev
            }
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

/// Derivative of the Planck taper function.
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

impl Reconfigurable<SpectrogramConfig> for SpectrogramProcessor {
    fn update_config(&mut self, config: SpectrogramConfig) {
        let previous = self.config;

        self.config = config;

        let fft_related_changed = previous.fft_size != self.config.fft_size
            || previous.zero_padding_factor != self.config.zero_padding_factor
            || previous.window != self.config.window
            || (previous.sample_rate - self.config.sample_rate).abs() > f32::EPSILON;

        if fft_related_changed {
            self.rebuild_fft();
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

        let grid_changed = previous.use_reassignment != self.config.use_reassignment
            || previous.display_bin_count != self.config.display_bin_count
            || (previous.display_min_hz - self.config.display_min_hz).abs() > f32::EPSILON
            || previous.frequency_scale != self.config.frequency_scale
            || (previous.reassignment_max_time_hops - self.config.reassignment_max_time_hops).abs()
                > f32::EPSILON;

        if grid_changed {
            self.reassignment_2d
                .reconfigure(&self.config, self.config.sample_rate);
            let output_bins = if self.reassignment_2d.is_enabled() {
                self.reassignment_2d.display_bin_count()
            } else {
                self.fft_size / 2 + 1
            };
            self.magnitude_pool
                .retain(|buffer| buffer.len() == output_bins);
            self.clear_history();
        }
    }
}

#[inline(always)]
fn load_complex_simd(data: &[Complex32]) -> (f32x8, f32x8) {
    let re = f32x8::new([
        data[0].re, data[1].re, data[2].re, data[3].re, data[4].re, data[5].re, data[6].re,
        data[7].re,
    ]);
    let im = f32x8::new([
        data[0].im, data[1].im, data[2].im, data[3].im, data[4].im, data[5].im, data[6].im,
        data[7].im,
    ]);
    (re, im)
}

#[inline(always)]
fn load_complex_simd_safe(data: &[Complex32]) -> (f32x8, f32x8) {
    if data.len() >= 8 {
        load_complex_simd(data)
    } else {
        let mut re = [0.0; 8];
        let mut im = [0.0; 8];
        for (i, c) in data.iter().enumerate() {
            re[i] = c.re;
            im[i] = c.im;
        }
        (f32x8::new(re), f32x8::new(im))
    }
}

#[inline(always)]
fn load_f32_simd_safe(data: &[f32]) -> f32x8 {
    if data.len() >= 8 {
        f32x8::new(data[..8].try_into().unwrap())
    } else {
        let mut arr = [0.0; 8];
        arr[..data.len()].copy_from_slice(data);
        f32x8::new(arr)
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
        SpectrogramConfig::default()
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
            use_reassignment: false,
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

    #[test]
    fn window_sigma_t_matches_theoretical_ratios() {
        let fft_size = 4096;

        let hann = WindowKind::Hann.coefficients(fft_size);
        let sigma_t_hann = compute_window_sigma_t(&hann);
        let ratio_hann = sigma_t_hann / fft_size as f32;
        assert!(
            (ratio_hann - 0.1414).abs() < 0.01,
            "Hann sigma_t/N = {ratio_hann}, expected ~0.1414"
        );

        let blackman = WindowKind::Blackman.coefficients(fft_size);
        let sigma_t_blackman = compute_window_sigma_t(&blackman);
        let ratio_blackman = sigma_t_blackman / fft_size as f32;
        assert!(
            (ratio_blackman - 0.1188).abs() < 0.01,
            "Blackman sigma_t/N = {ratio_blackman}, expected ~0.1188"
        );

        let hann_2048 = WindowKind::Hann.coefficients(2048);
        let sigma_t_2048 = compute_window_sigma_t(&hann_2048);
        let ratio_2048 = sigma_t_2048 / 2048.0;
        assert!(
            (ratio_2048 - ratio_hann).abs() < 0.001,
            "sigma_t/N should be constant: {ratio_2048} vs {ratio_hann}"
        );
    }

    #[test]
    fn chirp_correction_tracks_linear_fm() {
        // Linear FM chirp: f(t) = f0 + c*t tests second-order correction.
        let sample_rate = 48_000.0;
        let config = SpectrogramConfig {
            fft_size: 2048,
            hop_size: 256,
            use_reassignment: true,
            zero_padding_factor: 2,
            window: WindowKind::Hann,
            ..test_config_base()
        };
        let mut processor = SpectrogramProcessor::new(config);

        let f0 = 1000.0;
        let chirp_rate = 2000.0; // Hz/second
        let duration_samples = config.fft_size * 3;

        let samples: Vec<f32> = (0..duration_samples)
            .map(|n| {
                let t = n as f32 / sample_rate;
                let phase = core::f32::consts::TAU * (f0 * t + 0.5 * chirp_rate * t * t);
                phase.sin()
            })
            .collect();

        let block = make_block(samples, 1, sample_rate);
        let update = unwrap_update(processor.process_block(&block));

        // Find peak in middle frame (after transient settles)
        let mid_frame = update.new_columns.len() / 2;
        let peak = update.new_columns[mid_frame]
            .reassigned
            .as_ref()
            .expect("reassignment enabled")
            .iter()
            .max_by(|a, b| a.magnitude_db.partial_cmp(&b.magnitude_db).unwrap())
            .unwrap();

        // Instantaneous frequency at window center
        let window_center_samples = (config.hop_size * mid_frame + config.fft_size / 2) as f32;
        let expected_freq = f0 + chirp_rate * (window_center_samples / sample_rate);

        // Tolerance: within 1% or 20 Hz, whichever is larger
        let tolerance = (expected_freq * 0.01).max(20.0);
        assert!(
            (peak.frequency_hz - expected_freq).abs() < tolerance,
            "Chirp tracking: got {:.1} Hz, expected {:.1} Hz (±{:.1})",
            peak.frequency_hz,
            expected_freq,
            tolerance
        );
    }
}
