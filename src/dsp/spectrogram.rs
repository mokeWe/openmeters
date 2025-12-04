//! spectrogram DSP. reassignment! (Auger-Flandrin 1995).
//!
//! # References
//! 1. F. Auger and P. Flandrin, "Improving the readability of time-frequency and
//!    time-scale representations by the reassignment method", IEEE Trans. SP,
//!    vol. 43, no. 5, pp. 1068-1089, May 1995.
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
    DB_FLOOR, DEFAULT_SAMPLE_RATE, copy_from_deque, db_to_power, power_to_db,
};
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

/// Precomputed Gaussian kernel for 3x3 splatting (sigma=0.45, radius=1).
/// Stored as [df][dt] for natural iteration order.
const GAUSSIAN_KERNEL_3X3: [[f32; 3]; 3] = {
    const CENTER: f32 = 1.0;
    const EDGE: f32 = 0.324_652_5;
    const CORNER: f32 = 0.105_399_2;
    const SUM: f32 = CENTER + 4.0 * EDGE + 4.0 * CORNER;
    [
        [CORNER / SUM, EDGE / SUM, CORNER / SUM], // df = -1
        [EDGE / SUM, CENTER / SUM, EDGE / SUM],   // df = 0
        [CORNER / SUM, EDGE / SUM, CORNER / SUM], // df = +1
    ]
};

#[inline]
fn resize_vecdeque<T: Default + Clone>(buffer: &mut VecDeque<T>, capacity: usize) {
    if capacity == 0 {
        buffer.clear();
        buffer.shrink_to_fit();
    } else if buffer.len() > capacity {
        buffer.drain(..buffer.len() - capacity);
    }
}

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

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct ReassignedSample {
    pub frequency_hz: f32,
    pub group_delay_samples: f32,
    pub magnitude_db: f32,
    pub confidence: f32,
    pub chirp_rate: f32,
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
    bin_omega: f32,
}

impl ReassignmentBuffers {
    fn new(
        window_kind: WindowKind,
        window: &[f32],
        fft: &Arc<dyn RealToComplex<f32>>,
        fft_size: usize,
        power_floor_db: f32,
        sample_rate: f32,
    ) -> Self {
        let bins = fft_size / 2 + 1;
        let derivative_window = compute_derivative_window(window_kind, window);
        let second_derivative_window = compute_second_derivative_window(&derivative_window);
        let bin_omega = if fft_size > 0 && sample_rate > 0.0 {
            core::f32::consts::TAU / fft_size as f32
        } else {
            0.0
        };
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
            bin_omega,
        }
    }

    fn resize(
        &mut self,
        window_kind: WindowKind,
        window: &[f32],
        fft: &Arc<dyn RealToComplex<f32>>,
        fft_size: usize,
        power_floor_db: f32,
        sample_rate: f32,
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
        self.bin_omega = if fft_size > 0 && sample_rate > 0.0 {
            core::f32::consts::TAU / fft_size as f32
        } else {
            0.0
        };
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
    fn compute(_scale: FrequencyScale, min_hz: f32, max_hz: f32, bin_count: usize) -> Self {
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
    fn hz_to_bin(&self, freq_hz: f32, scale: FrequencyScale) -> Option<f32> {
        if freq_hz < self.min_hz || freq_hz > self.max_hz {
            return None;
        }

        let normalized = match scale {
            FrequencyScale::Linear => (freq_hz - self.min_hz) * self.inv_linear_range,
            FrequencyScale::Logarithmic => (freq_hz.ln() - self.log_min) * self.inv_log_range,
            FrequencyScale::Mel => {
                let mel = hz_to_mel(freq_hz);
                (mel - self.mel_min) * self.inv_mel_range
            }
        };

        Some((1.0 - normalized) * self.bin_count_minus_1)
    }
}

struct Reassignment2DGrid {
    display_bins: usize,
    max_time_hops: usize,
    hop_size: usize,
    power_grid: Vec<f32>,
    weight_grid: Vec<f32>,
    center_column: usize,
    column_count: usize,
    frames_accumulated: usize,
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
            weight_grid: Vec::new(),
            center_column: 1,
            column_count: 3,
            frames_accumulated: 0,
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
            self.weight_grid.clear();
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
            self.scale_params = FrequencyScaleParams::compute(scale, min_hz, max_hz, display_bins);
            self.bin_frequencies_hz = Self::compute_bin_frequencies(
                scale,
                min_hz,
                max_hz,
                display_bins,
                self.scale_params.log_min,
                1.0 / self.scale_params.inv_log_range,
                self.scale_params.mel_min,
                1.0 / self.scale_params.inv_mel_range,
            );
        }

        if size_changed {
            let grid_size = display_bins * column_count;
            self.power_grid.resize(grid_size, 0.0);
            self.weight_grid.resize(grid_size, 0.0);
            self.output_buffer.resize(display_bins, 0.0);
            self.magnitude_buffer.resize(display_bins, DB_FLOOR);
            self.reset();
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_bin_frequencies(
        scale: FrequencyScale,
        min_hz: f32,
        max_hz: f32,
        bin_count: usize,
        log_min: f32,
        log_range: f32,
        mel_min: f32,
        mel_range: f32,
    ) -> Arc<[f32]> {
        if bin_count == 0 {
            return Arc::from([]);
        }

        if bin_count == 1 {
            return Arc::from([min_hz]);
        }

        let mut freqs = Vec::with_capacity(bin_count);
        for idx in 0..bin_count {
            let t = idx as f32 / (bin_count as f32 - 1.0);
            let freq = match scale {
                FrequencyScale::Linear => min_hz + (max_hz - min_hz) * t,
                FrequencyScale::Logarithmic => (log_min + log_range * t).exp(),
                FrequencyScale::Mel => mel_to_hz(mel_min + mel_range * t),
            };
            freqs.push(freq);
        }
        freqs.reverse();
        Arc::from(freqs.into_boxed_slice())
    }

    fn reset(&mut self) {
        self.power_grid.fill(0.0);
        self.weight_grid.fill(0.0);
        self.magnitude_buffer.fill(DB_FLOOR);
        self.frames_accumulated = 0;
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
    fn accumulate(&mut self, freq_hz: f32, time_offset_samples: f32, power: f32, confidence: f32) {
        if power <= 0.0 || !self.enabled {
            return;
        }

        let display_bins = self.display_bins;
        let column_count = self.column_count;

        if display_bins == 0 || column_count == 0 || self.hop_size == 0 {
            return;
        }

        let Some(freq_bin) = self.scale_params.hz_to_bin(freq_hz, self.scale) else {
            return;
        };

        let inv_hop = 1.0 / self.hop_size as f32;
        let time_offset_hops = time_offset_samples * inv_hop;
        let max_offset = self.max_time_hops as f32;
        let clamped_time = time_offset_hops.clamp(-max_offset, max_offset);
        let time_col = clamped_time + self.center_column as f32;

        let f_center_i = freq_bin.round() as i32;
        let t_center_i = time_col.round() as i32;

        let f_lo = (f_center_i - 1).max(0) as usize;
        let f_hi = (f_center_i + 1).min(display_bins as i32 - 1) as usize;
        let t_lo = (t_center_i - 1).max(0) as usize;
        let t_hi = (t_center_i + 1).min(column_count as i32 - 1) as usize;

        if f_lo > f_hi || t_lo > t_hi {
            return;
        }

        let kf_off = (f_lo as i32 - f_center_i + 1) as usize;
        let kt_off = (t_lo as i32 - t_center_i + 1) as usize;
        let weighted_power = power * confidence;

        let row_end = (t_hi + 1) * display_bins;
        let pg = &mut self.power_grid[..row_end];
        let wg = &mut self.weight_grid[..row_end];

        for dt in 0..=(t_hi - t_lo) {
            let t = t_lo + dt;
            let kt = kt_off + dt;
            let row_base = t * display_bins;

            for df in 0..=(f_hi - f_lo) {
                let f = f_lo + df;
                let kf = kf_off + df;
                let idx = row_base + f;
                let kw = GAUSSIAN_KERNEL_3X3[kf][kt];
                pg[idx] = kw.mul_add(weighted_power, pg[idx]);
                wg[idx] = kw.mul_add(confidence, wg[idx]);
            }
        }
    }

    fn advance(&mut self, energy_scale: &[f32], bin_norm: &[f32]) {
        self.frames_accumulated += 1;
        self.output_buffer
            .copy_from_slice(&self.power_grid[..self.display_bins]);
        self.power_grid.rotate_left(self.display_bins);
        self.weight_grid.rotate_left(self.display_bins);
        let new_right_start = (self.column_count - 1) * self.display_bins;
        self.power_grid[new_right_start..].fill(0.0);
        self.weight_grid[new_right_start..].fill(0.0);

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

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SpectrogramColumn {
    pub timestamp: Instant,
    pub magnitudes_db: Arc<[f32]>,
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
    buffer_start_index: u64,
    start_instant: Option<Instant>,
    accumulated_offset: Duration,
    history: SpectrogramHistory,
    magnitude_pool: Vec<Arc<[f32]>>,
    evicted_columns: Vec<SpectrogramColumn>,
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
            runtime_config.sample_rate,
        );

        let reassignment_2d = Reassignment2DGrid::new(&runtime_config, runtime_config.sample_rate);

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
            reassignment_2d,
            bin_normalization,
            energy_normalization,
            pcm_buffer: VecDeque::with_capacity(window_size.saturating_mul(2)),
            buffer_start_index: 0,
            start_instant: None,
            accumulated_offset: Duration::default(),
            history: SpectrogramHistory::new(history_len),
            magnitude_pool: Vec::new(),
            evicted_columns: Vec::new(),
            pending_reset: true,
            output_columns_buffer: Vec::with_capacity(8),
        }
    }

    fn normalize_config(_config: &mut SpectrogramConfig) {
        // All configuration is now self-consistent; no normalization needed.
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
            self.config.sample_rate,
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
        let fft_bins = fft_size / 2 + 1;
        let reassignment_enabled = self.config.use_reassignment
            && sample_rate > f32::EPSILON
            && self.reassignment_2d.is_enabled();
        let reassignment_bin_limit = if self.config.reassignment_low_bin_limit == 0 {
            fft_bins
        } else {
            self.config.reassignment_low_bin_limit.min(fft_bins)
        };

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

            let timestamp = start_instant + self.accumulated_offset + center_offset;

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
                timestamp,
                magnitudes_db: Arc::clone(&magnitudes),
                reassigned: reassigned.clone(),
            };
            if let Some(evicted) = self.history.push(history_column) {
                self.recycle_column(evicted);
            }

            self.output_columns_buffer.push(SpectrogramColumn {
                timestamp,
                magnitudes_db: magnitudes,
                reassigned,
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
    ) -> Option<Arc<[ReassignedSample]>> {
        let power_floor = self.reassignment.power_floor_linear;
        let bin_hz = sample_rate / fft_size as f32;
        let nyquist = sample_rate * 0.5;
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

            let cross_product_freq = cross.im * base.re - cross.re * base.im;
            let delta_omega = -cross_product_freq * inv_power;

            if !delta_omega.is_finite() {
                continue;
            }

            let mut chirp_rate = 0.0f32;
            let mut freq_correction_hz = delta_omega * inv_two_pi;

            {
                let second = second_derivative_spectrum[k];
                let second_cross_freq = second.im * base.re - second.re * base.im;
                let d_omega_dt = -second_cross_freq * inv_power;

                if d_omega_dt.is_finite() {
                    chirp_rate = d_omega_dt * inv_two_pi;

                    let max_chirp_correction = 0.25 * bin_hz;
                    if chirp_rate.abs() < max_chirp_correction * 10.0 {
                        let time_cross_mag =
                            (time_cross.re * time_cross.re + time_cross.im * time_cross.im).sqrt();
                        let weight = (-time_cross_mag * 0.01).exp().min(1.0);
                        freq_correction_hz += chirp_rate * weight * 0.5;
                    }
                }
            }

            if freq_correction_hz.abs() > max_correction_hz {
                continue;
            }

            let freq_hz = k_f32.mul_add(bin_hz, freq_correction_hz);
            if freq_hz < 0.0 || freq_hz > nyquist {
                continue;
            }

            let cross_product_time = time_cross.re * base.re + time_cross.im * base.im;
            let delta_tau = -cross_product_time * inv_power;
            let group_delay_samples = if delta_tau.is_finite() {
                delta_tau
            } else {
                0.0
            };

            let snr_factor =
                (power_to_db(display_power) - power_to_db(power_floor)).max(0.0) / 60.0;
            let snr_confidence = snr_factor.min(1.0);

            let correction_ratio = freq_correction_hz.abs() / bin_hz;
            let phase_coherence = 1.0 - correction_ratio.min(1.0);

            let confidence = (snr_confidence * phase_coherence).max(0.01);

            let energy_power = power * energy_scale_k;

            self.reassignment_2d
                .accumulate(freq_hz, group_delay_samples, energy_power, confidence);

            if samples.len() < MAX_REASSIGNMENT_SAMPLES {
                samples.push(ReassignedSample {
                    frequency_hz: freq_hz,
                    group_delay_samples,
                    magnitude_db: power_to_db(display_power),
                    confidence,
                    chirp_rate,
                });
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

        resize_vecdeque(
            &mut self.pcm_buffer,
            self.window_size.saturating_mul(2).max(1),
        );
        self.pcm_buffer.clear();
        self.buffer_start_index = 0;
        self.start_instant = None;
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
    let gradient = s / (d * d) - s / (denom * denom);
    (logistic * (1.0 - logistic) * gradient) as f32
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
        // Test without reassignment to verify basic FFT functionality
        let config = SpectrogramConfig {
            fft_size: 1024,
            hop_size: 512,
            history_length: 8,
            window: WindowKind::Hann,
            zero_padding_factor: 1,
            use_reassignment: false, // Disable reassignment for this test
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
    fn zero_padding_expands_fft_resolution() {
        // Test without reassignment to verify FFT sizing
        let config = SpectrogramConfig {
            fft_size: 1024,
            hop_size: 256,
            history_length: 4,
            window: WindowKind::Hann,
            zero_padding_factor: 4,
            use_reassignment: false, // Disable reassignment for this test
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

    #[test]
    fn reassignment_2d_grid_accumulates_energy() {
        // Test that the 2D grid properly accumulates and redistributes energy
        let config = SpectrogramConfig {
            fft_size: 1024,
            hop_size: 256,
            history_length: 8,
            use_reassignment: true,
            reassignment_max_time_hops: 2.0,
            reassignment_low_bin_limit: 0,
            zero_padding_factor: 1,
            ..test_config_base()
        };
        let mut processor = SpectrogramProcessor::new(config);

        // Generate a steady sine wave - with true 2D reassignment, energy
        // should still be concentrated at the correct frequency
        let true_freq = 1000.0;
        let block = make_block(
            sine_samples(true_freq, config.sample_rate, config.fft_size * 6),
            1,
            config.sample_rate,
        );

        let update = unwrap_update(processor.process_block(&block));

        // Should have multiple columns from the sliding 2D grid
        assert!(
            !update.new_columns.is_empty(),
            "expected output columns with 2D reassignment"
        );

        // The peak frequency should still be accurately detected
        if let Some(last) = update.new_columns.last()
            && let Some(ref samples) = last.reassigned
        {
            let peak = samples
                .iter()
                .max_by(|a, b| a.magnitude_db.partial_cmp(&b.magnitude_db).unwrap())
                .unwrap();

            // Frequency detection should be within 5 Hz
            let freq_error = (peak.frequency_hz - true_freq).abs();
            assert!(
                freq_error < 5.0,
                "expected peak near {true_freq} Hz, got {} Hz (error: {} Hz)",
                peak.frequency_hz,
                freq_error
            );
        }
    }
}
