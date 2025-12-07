use super::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::util::audio::DEFAULT_SAMPLE_RATE;
use realfft::{RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;

const PITCH_MIN_HZ: f32 = 20.0;
const PITCH_MAX_HZ: f32 = 8000.0;
const PITCH_THRESHOLD: f32 = 0.15;
const FFT_AUTOCORR_THRESHOLD: usize = 512;

#[inline]
fn parabolic_refine(y_prev: f32, y_curr: f32, y_next: f32, tau: usize) -> f32 {
    let denom = y_prev - 2.0 * y_curr + y_next;
    if denom.abs() < f32::EPSILON {
        return tau as f32;
    }
    let delta = 0.5 * (y_prev - y_next) / denom;
    (tau as f32 + delta.clamp(-1.0, 1.0)).max(1.0)
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TriggerMode {
    FreeRun,
    Stable { num_cycles: usize },
}

impl Default for TriggerMode {
    fn default() -> Self {
        Self::Stable { num_cycles: 1 }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct OscilloscopeConfig {
    pub sample_rate: f32,
    pub segment_duration: f32,
    pub trigger_mode: TriggerMode,
}

impl Default for OscilloscopeConfig {
    fn default() -> Self {
        Self {
            sample_rate: DEFAULT_SAMPLE_RATE,
            segment_duration: 0.02,
            trigger_mode: TriggerMode::default(),
        }
    }
}

#[derive(Clone)]
struct PitchDetector {
    diff: Vec<f32>,
    cmean: Vec<f32>,
    fft_size: usize,
    fft_forward: Option<Arc<dyn RealToComplex<f32>>>,
    fft_inverse: Option<Arc<dyn realfft::ComplexToReal<f32>>>,
    fft_input: Vec<f32>,
    fft_spectrum: Vec<Complex<f32>>,
    fft_output: Vec<f32>,
    fft_scratch: Vec<Complex<f32>>,
}

impl std::fmt::Debug for PitchDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PitchDetector")
            .field("diff", &self.diff.len())
            .field("cmean", &self.cmean.len())
            .field("fft_size", &self.fft_size)
            .field("has_fft", &self.fft_forward.is_some())
            .finish()
    }
}

impl PitchDetector {
    fn new() -> Self {
        Self {
            diff: Vec::new(),
            cmean: Vec::new(),
            fft_size: 0,
            fft_forward: None,
            fft_inverse: None,
            fft_input: Vec::new(),
            fft_spectrum: Vec::new(),
            fft_output: Vec::new(),
            fft_scratch: Vec::new(),
        }
    }

    fn rebuild_fft(&mut self, size: usize) {
        if self.fft_size == size && self.fft_forward.is_some() {
            return;
        }
        self.fft_size = size;
        let mut planner = RealFftPlanner::new();
        let forward = planner.plan_fft_forward(size);
        let inverse = planner.plan_fft_inverse(size);
        self.fft_input = forward.make_input_vec();
        self.fft_spectrum = forward.make_output_vec();
        self.fft_output = inverse.make_output_vec();
        self.fft_scratch = forward.make_scratch_vec();
        let inv_scratch = inverse.make_scratch_vec();
        if inv_scratch.len() > self.fft_scratch.len() {
            self.fft_scratch = inv_scratch;
        }
        self.fft_forward = Some(forward);
        self.fft_inverse = Some(inverse);
    }

    fn detect_pitch(&mut self, samples: &[f32], rate: f32) -> Option<f32> {
        if samples.is_empty() {
            return None;
        }

        let min_period = (rate / PITCH_MAX_HZ).max(2.0) as usize;
        let max_period = (rate / PITCH_MIN_HZ).min(samples.len() as f32 / 2.0) as usize;

        if max_period <= min_period || samples.len() < max_period * 2 {
            return None;
        }

        self.diff.resize(max_period, 0.0);
        self.cmean.resize(max_period, 0.0);

        if samples.len() >= FFT_AUTOCORR_THRESHOLD {
            if !self.compute_diff_fft(samples, max_period) {
                self.compute_diff_direct(samples, max_period);
            }
        } else {
            self.compute_diff_direct(samples, max_period);
        }

        self.cmean[0] = 1.0;
        let mut sum = 0.0;
        for tau in 1..max_period {
            sum += self.diff[tau];
            self.cmean[tau] = if sum > f32::EPSILON {
                self.diff[tau] * tau as f32 / sum
            } else {
                1.0
            };
        }

        for tau in min_period..max_period - 1 {
            if self.cmean[tau] < PITCH_THRESHOLD && self.cmean[tau] < self.cmean[tau + 1] {
                let refined_tau = if tau > 0 && tau + 1 < max_period {
                    parabolic_refine(
                        self.cmean[tau - 1],
                        self.cmean[tau],
                        self.cmean[tau + 1],
                        tau,
                    )
                } else {
                    tau as f32
                };
                return Some(rate / refined_tau);
            }
        }

        let mut best_tau = min_period;
        let mut best_val = f32::MAX;
        for tau in min_period..max_period {
            if self.cmean[tau] < best_val {
                best_val = self.cmean[tau];
                best_tau = tau;
            }
        }

        if best_val < 0.5 {
            let refined_tau = if best_tau > 0 && best_tau + 1 < max_period {
                parabolic_refine(
                    self.cmean[best_tau - 1],
                    self.cmean[best_tau],
                    self.cmean[best_tau + 1],
                    best_tau,
                )
            } else {
                best_tau as f32
            };
            Some(rate / refined_tau)
        } else {
            None
        }
    }

    fn compute_diff_fft(&mut self, samples: &[f32], max_period: usize) -> bool {
        let fft_size = (samples.len() * 2).next_power_of_two();
        self.rebuild_fft(fft_size);

        let Some(ref forward) = self.fft_forward else {
            return false;
        };
        let Some(ref inverse) = self.fft_inverse else {
            return false;
        };

        self.fft_input[..samples.len()].copy_from_slice(samples);
        self.fft_input[samples.len()..].fill(0.0);

        if forward
            .process_with_scratch(
                &mut self.fft_input,
                &mut self.fft_spectrum,
                &mut self.fft_scratch,
            )
            .is_err()
        {
            return false;
        }

        for c in &mut self.fft_spectrum {
            *c = Complex::new(c.norm_sqr(), 0.0);
        }

        if inverse
            .process_with_scratch(
                &mut self.fft_spectrum,
                &mut self.fft_output,
                &mut self.fft_scratch,
            )
            .is_err()
        {
            return false;
        }

        let norm = 1.0 / fft_size as f32;
        let acf_0 = self.fft_output[0] * norm;

        for tau in 0..max_period {
            let acf_tau = self.fft_output[tau] * norm;
            self.diff[tau] = 2.0 * (acf_0 - acf_tau);
        }
        true
    }

    fn compute_diff_direct(&mut self, samples: &[f32], max_period: usize) {
        let len = samples.len() - max_period;
        if len == 0 {
            return;
        }

        for tau in 0..max_period {
            let mut sum = 0.0_f32;
            for i in 0..len {
                let delta = samples[i] - samples[i + tau];
                sum += delta * delta;
            }
            self.diff[tau] = sum;
        }
    }
}

#[derive(Debug, Clone, Default)]
struct TriggerScratch {
    sin: Vec<f32>,
    cos: Vec<f32>,
    psin: Vec<f32>,
    pcos: Vec<f32>,
}

impl TriggerScratch {
    fn clear(&mut self) {
        self.sin.clear();
        self.cos.clear();
        self.psin.clear();
        self.pcos.clear();
    }

    fn prepare(&mut self, data: &[f32], period: usize) {
        let len = data.len();

        self.sin.clear();
        self.cos.clear();
        self.psin.clear();
        self.pcos.clear();

        self.sin.resize(len + 1, 0.0);
        self.cos.resize(len + 1, 0.0);
        self.psin.resize(len + 1, 0.0);
        self.pcos.resize(len + 1, 0.0);

        let step = std::f32::consts::TAU / period as f32;
        let (s_step, c_step) = step.sin_cos();

        let mut s = 0.0_f32;
        let mut c = 1.0_f32;

        self.psin[0] = s;
        self.pcos[0] = c;

        for (i, &sample) in data.iter().take(len).enumerate() {
            self.sin[i + 1] = self.sin[i] + sample * s;
            self.cos[i + 1] = self.cos[i] + sample * c;

            let mut ns = s * c_step + c * s_step;
            let mut nc = c * c_step - s * s_step;

            if (i & 127) == 127 {
                let mag = (ns * ns + nc * nc).sqrt();
                if mag > f32::EPSILON {
                    ns /= mag;
                    nc /= mag;
                } else {
                    ns = 0.0;
                    nc = 1.0;
                }
            }

            s = ns;
            c = nc;

            self.psin[i + 1] = s;
            self.pcos[i + 1] = c;
        }
    }

    #[inline]
    fn correlation(&self, offset: usize, length: usize) -> f32 {
        debug_assert!(offset + length < self.sin.len());
        debug_assert!(offset < self.pcos.len());

        let ss = self.sin[offset + length] - self.sin[offset];
        let sc = self.cos[offset + length] - self.cos[offset];
        self.pcos[offset] * ss - self.psin[offset] * sc
    }
}

#[inline]
fn find_trigger(
    period: usize,
    cycles: usize,
    available: usize,
    mono: &[f32],
    scratch: &mut TriggerScratch,
) -> (usize, usize) {
    let cycles = cycles.max(1);
    let len = period.saturating_mul(cycles);
    let guard = period.saturating_mul(2);
    let window = len.saturating_add(guard);
    let start = available.saturating_sub(window);

    let data = &mono[start..];
    if period == 0 || len == 0 {
        return (0, available);
    }

    if data.len() <= len {
        return (len, start);
    }

    scratch.prepare(data, period);

    let range = data.len() - len;
    let stride = (period / 4).max(1);

    let mut best = f32::NEG_INFINITY;
    let mut pos = 0;

    for i in (0..=range).step_by(stride) {
        let corr = scratch.correlation(i, len);
        if corr > best {
            best = corr;
            pos = i;
        }
    }

    let refine_start = pos.saturating_sub(stride);
    let refine_end = (pos + stride).min(range);

    for i in refine_start..=refine_end {
        if i == pos || i % stride == 0 {
            continue;
        }
        let corr = scratch.correlation(i, len);
        if corr > best {
            best = corr;
            pos = i;
        }
    }

    (len, start + pos)
}

#[derive(Debug, Clone, Default)]
pub struct OscilloscopeSnapshot {
    pub channels: usize,
    pub samples: Vec<f32>,
    pub samples_per_channel: usize,
}

#[derive(Debug, Clone)]
pub struct OscilloscopeProcessor {
    config: OscilloscopeConfig,
    snapshot: OscilloscopeSnapshot,
    history: VecDeque<f32>,
    pitch_detector: PitchDetector,
    last_pitch: Option<f32>,
    mono_buffer: Vec<f32>,
    trigger_scratch: TriggerScratch,
}

impl OscilloscopeProcessor {
    pub fn new(config: OscilloscopeConfig) -> Self {
        Self {
            config,
            snapshot: OscilloscopeSnapshot::default(),
            history: VecDeque::new(),
            pitch_detector: PitchDetector::new(),
            last_pitch: None,
            mono_buffer: Vec::new(),
            trigger_scratch: TriggerScratch::default(),
        }
    }

    pub fn config(&self) -> OscilloscopeConfig {
        self.config
    }
}

impl AudioProcessor for OscilloscopeProcessor {
    type Output = OscilloscopeSnapshot;

    fn process_block(&mut self, block: &AudioBlock<'_>) -> ProcessorUpdate<Self::Output> {
        let ch = block.channels.max(1);
        if block.frame_count() == 0 {
            return ProcessorUpdate::None;
        }

        let base = (self.config.sample_rate * self.config.segment_duration)
            .round()
            .max(1.0) as usize;
        let detect = (self.config.sample_rate * 0.1) as usize;
        let capacity = detect.max(base) * ch;

        if !self.history.is_empty() && !self.history.len().is_multiple_of(ch) {
            self.history.clear();
            self.last_pitch = None;
        }

        if block.samples.len() >= capacity {
            self.history.clear();
            self.history
                .extend(&block.samples[block.samples.len() - capacity..]);
        } else {
            let overflow = self.history.len() + block.samples.len();
            if overflow > capacity {
                let remove = ((overflow - capacity).div_ceil(ch) * ch).min(self.history.len());
                self.history.drain(..remove);
            }
            self.history.extend(block.samples);
        }

        let avail = self.history.len() / ch;

        let (frames, start) = match self.config.trigger_mode {
            TriggerMode::FreeRun => {
                let frames = base.min(avail);
                if frames == 0 {
                    return ProcessorUpdate::None;
                }
                (frames, avail.saturating_sub(frames))
            }
            TriggerMode::Stable { num_cycles } => {
                if avail < base {
                    return ProcessorUpdate::None;
                }

                {
                    let data = self.history.make_contiguous();
                    self.mono_buffer.clear();
                    self.mono_buffer.reserve(avail);

                    if ch == 1 {
                        self.mono_buffer.extend_from_slice(&data[..avail]);
                    } else {
                        let scale = 1.0 / ch as f32;
                        for i in 0..avail {
                            let idx = i * ch;
                            let mut sum = 0.0_f32;
                            for c in 0..ch {
                                sum += data[idx + c];
                            }
                            self.mono_buffer.push(sum * scale);
                        }
                    }
                }

                let freq = self
                    .pitch_detector
                    .detect_pitch(&self.mono_buffer, self.config.sample_rate)
                    .or(self.last_pitch);

                if let Some(f) = freq {
                    self.last_pitch = Some(f);
                    let period = (self.config.sample_rate / f).max(1.0) as usize;
                    find_trigger(
                        period,
                        num_cycles,
                        avail,
                        &self.mono_buffer,
                        &mut self.trigger_scratch,
                    )
                } else {
                    (base, avail.saturating_sub(base))
                }
            }
        };

        const TARGET: usize = 4096;
        let target = TARGET.clamp(1, frames);
        let extract_start = start * ch;
        let data = self.history.make_contiguous();
        let extract_len = (frames * ch).min(data.len() - extract_start);

        self.snapshot.samples.clear();

        downsample_interleaved(
            &mut self.snapshot.samples,
            &data[extract_start..extract_start + extract_len],
            frames.min(extract_len / ch),
            ch,
            target,
        );

        self.snapshot.channels = ch;
        self.snapshot.samples_per_channel = target;

        ProcessorUpdate::Snapshot(self.snapshot.clone())
    }

    fn reset(&mut self) {
        self.snapshot = OscilloscopeSnapshot::default();
        self.history.clear();
        self.last_pitch = None;
        self.mono_buffer.clear();
        self.trigger_scratch.clear();
    }
}

impl Reconfigurable<OscilloscopeConfig> for OscilloscopeProcessor {
    fn update_config(&mut self, config: OscilloscopeConfig) {
        self.config = config;
        self.reset();
    }
}

fn downsample_interleaved(
    output: &mut Vec<f32>,
    data: &[f32],
    frames: usize,
    ch: usize,
    target: usize,
) {
    if frames == 0 || ch == 0 || target == 0 {
        return;
    }

    let step = frames as f32 / target as f32;

    for c in 0..ch {
        for i in 0..target {
            let pos = i as f32 * step;
            let idx = pos as usize;
            let frac = pos - idx as f32;

            let sample = if frac > f32::EPSILON && idx + 1 < frames {
                let curr = data[idx * ch + c];
                let next = data[(idx + 1) * ch + c];
                crate::util::audio::lerp(curr, next, frac)
            } else {
                data[idx * ch + c]
            };

            output.push(sample);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::{AudioBlock, ProcessorUpdate};
    use std::time::Instant;

    fn make_block(samples: &[f32], channels: usize, sample_rate: f32) -> AudioBlock<'_> {
        AudioBlock::new(samples, channels, sample_rate, Instant::now())
    }

    fn sine_samples(freq: f32, rate: f32, frames: usize) -> Vec<f32> {
        (0..frames)
            .map(|n| (std::f32::consts::TAU * freq * n as f32 / rate).sin())
            .collect()
    }

    #[test]
    fn produces_downsampled_snapshot_when_buffer_ready() {
        let config = OscilloscopeConfig {
            segment_duration: 0.01,
            trigger_mode: TriggerMode::FreeRun,
            ..Default::default()
        };
        let mut processor = OscilloscopeProcessor::new(config);
        let frames = (config.sample_rate * config.segment_duration).round() as usize;
        let samples: Vec<f32> = (0..frames)
            .flat_map(|f| {
                let s = (f as f32 / frames as f32 * std::f32::consts::TAU).sin();
                [s, -s]
            })
            .collect();

        if let ProcessorUpdate::Snapshot(s) =
            processor.process_block(&make_block(&samples, 2, DEFAULT_SAMPLE_RATE))
        {
            assert_eq!(s.channels, 2);
            assert!(s.samples_per_channel > 0 && s.samples_per_channel <= 4096);
            assert_eq!(s.samples.len(), s.samples_per_channel * 2);
        } else {
            panic!("expected snapshot");
        }
    }

    #[test]
    fn pitch_detection() {
        let mut detector = PitchDetector::new();
        let rate = 48_000.0;

        for freq in [41.0, 110.0, 440.0, 1000.0, 4000.0] {
            let samples = sine_samples(freq, rate, (rate * 0.1) as usize);
            let detected = detector
                .detect_pitch(&samples, rate)
                .unwrap_or_else(|| panic!("Failed to detect {}Hz", freq));
            let error = (detected - freq).abs() / freq;
            assert!(
                error < 0.05,
                "Detected {}Hz, expected {}Hz (error {:.1}%)",
                detected,
                freq,
                error * 100.0
            );
        }
    }

    #[test]
    fn parabolic_interpolation() {
        let y = |x: f32| (x - 5.3_f32).powi(2);
        let refined = parabolic_refine(y(4.0), y(5.0), y(6.0), 5);
        assert!((refined - 5.3).abs() < 0.01);
    }
}
