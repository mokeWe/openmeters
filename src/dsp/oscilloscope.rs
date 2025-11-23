//! Oscilloscope/triggered waveform DSP implementation.

use super::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::util::audio::{DEFAULT_SAMPLE_RATE, interpolate_linear};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

const PITCH_MIN_HZ: f32 = 10.0;
const PITCH_MAX_HZ: f32 = 8000.0;
const PITCH_THRESHOLD: f32 = 0.15;
const PITCH_DOWNSAMPLE_RATE: f32 = 12_000.0;

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

#[derive(Debug, Clone)]
struct PitchDetector {
    difference: Vec<f32>,
    cumulative_mean: Vec<f32>,
    downsample_buffer: Vec<f32>,
}

impl PitchDetector {
    fn new() -> Self {
        Self {
            difference: Vec::new(),
            cumulative_mean: Vec::new(),
            downsample_buffer: Vec::new(),
        }
    }

    fn detect_pitch(&mut self, samples: &[f32], sample_rate: f32) -> Option<f32> {
        if samples.is_empty() {
            return None;
        }

        let downsampled = if sample_rate > PITCH_DOWNSAMPLE_RATE * 1.5 {
            self.downsample(samples, sample_rate, PITCH_DOWNSAMPLE_RATE);
            &self.downsample_buffer
        } else {
            samples
        };

        let working_rate = if sample_rate > PITCH_DOWNSAMPLE_RATE * 1.5 {
            PITCH_DOWNSAMPLE_RATE
        } else {
            sample_rate
        };

        let min_period = (working_rate / PITCH_MAX_HZ).max(2.0) as usize;
        let max_period = (working_rate / PITCH_MIN_HZ).min(downsampled.len() as f32 / 2.0) as usize;

        if max_period <= min_period || downsampled.len() < max_period * 2 {
            return None;
        }

        self.difference.resize(max_period, 0.0);
        self.cumulative_mean.resize(max_period, 0.0);

        let stride = ((max_period - min_period) / 64).max(1);
        let len = downsampled.len() - max_period;

        for tau in (0..max_period).step_by(stride) {
            let mut sum = 0.0_f32;
            for i in (0..len).step_by(4) {
                let remaining = len - i;
                if remaining >= 4 {
                    let d0 = unsafe {
                        *downsampled.get_unchecked(i) - *downsampled.get_unchecked(i + tau)
                    };
                    let d1 = unsafe {
                        *downsampled.get_unchecked(i + 1) - *downsampled.get_unchecked(i + 1 + tau)
                    };
                    let d2 = unsafe {
                        *downsampled.get_unchecked(i + 2) - *downsampled.get_unchecked(i + 2 + tau)
                    };
                    let d3 = unsafe {
                        *downsampled.get_unchecked(i + 3) - *downsampled.get_unchecked(i + 3 + tau)
                    };
                    sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
                } else {
                    for j in 0..remaining {
                        let delta = downsampled[i + j] - downsampled[i + j + tau];
                        sum += delta * delta;
                    }
                }
            }
            self.difference[tau] = sum;
        }

        for tau in 0..max_period {
            if tau % stride != 0 {
                let prev = (tau / stride) * stride;
                let next = prev + stride;
                if next < max_period {
                    let t = (tau - prev) as f32 / stride as f32;
                    self.difference[tau] =
                        self.difference[prev] * (1.0 - t) + self.difference[next] * t;
                } else {
                    self.difference[tau] = self.difference[prev];
                }
            }
        }

        self.cumulative_mean[0] = 1.0;
        let mut running_sum = 0.0;
        for tau in 1..max_period {
            running_sum += self.difference[tau];
            self.cumulative_mean[tau] = if running_sum > f32::EPSILON {
                self.difference[tau] * tau as f32 / running_sum
            } else {
                1.0
            };
        }

        for tau in min_period..max_period {
            if self.cumulative_mean[tau] < PITCH_THRESHOLD
                && tau + 1 < max_period
                && self.cumulative_mean[tau] < self.cumulative_mean[tau + 1]
            {
                let confidence = 1.0 - self.cumulative_mean[tau];
                if confidence > PITCH_THRESHOLD {
                    return Some(working_rate / tau as f32);
                }
            }
        }

        None
    }

    fn downsample(&mut self, samples: &[f32], original_rate: f32, target_rate: f32) {
        let ratio = original_rate / target_rate;
        let output_len = (samples.len() as f32 / ratio) as usize;

        self.downsample_buffer.clear();
        self.downsample_buffer.reserve(output_len);

        for i in 0..output_len {
            let pos = i as f32 * ratio;
            self.downsample_buffer
                .push(interpolate_linear(samples, pos));
        }
    }
}

#[inline]
fn find_best_trigger_offset(mono: &[f32], sine_table: &[f32], num_cycles: usize) -> usize {
    let period = sine_table.len();
    if period == 0 {
        return 0;
    }

    let cycles = num_cycles.max(1);
    let eval_length = period.saturating_mul(cycles);
    if eval_length == 0 || mono.len() < eval_length {
        return 0;
    }

    let search_range = mono.len() - eval_length;
    let stride = (period / 4).max(1);

    let mut best_correlation = f32::NEG_INFINITY;
    let mut best_offset = 0;

    for offset in (0..=search_range).step_by(stride) {
        let signal = &mono[offset..offset + eval_length];
        let correlation = compute_correlation_chunked(signal, sine_table);
        if correlation > best_correlation {
            best_correlation = correlation;
            best_offset = offset;
        }
    }

    let refine_start = best_offset.saturating_sub(stride);
    let refine_end = (best_offset + stride).min(search_range);

    for offset in refine_start..=refine_end {
        if offset == best_offset || offset % stride == 0 {
            continue;
        }
        let signal = &mono[offset..offset + eval_length];
        let correlation = compute_correlation_chunked(signal, sine_table);
        if correlation > best_correlation {
            best_correlation = correlation;
            best_offset = offset;
        }
    }

    best_offset
}

#[inline]
fn compute_trigger_region(
    period: usize,
    num_cycles: usize,
    available_frames: usize,
    mono: &[f32],
    sine_table: &[f32],
) -> (usize, usize) {
    if period == 0 || sine_table.is_empty() {
        return (0, available_frames);
    }

    let cycles = num_cycles.max(1);
    let cycle_frames = period.saturating_mul(cycles);
    let guard_frames = period.saturating_mul(2);
    let search_window = cycle_frames.saturating_add(guard_frames);
    let search_start = available_frames.saturating_sub(search_window);

    let trigger_slice = &mono[search_start..];
    let trigger_offset = find_best_trigger_offset(trigger_slice, sine_table, cycles);

    (cycle_frames, search_start + trigger_offset)
}

#[inline(always)]
fn compute_correlation_chunked(signal: &[f32], sine_table: &[f32]) -> f32 {
    let period = sine_table.len();
    if period == 0 || signal.is_empty() {
        return 0.0;
    }

    let mut correlation = 0.0_f32;
    let mut chunks = signal.chunks_exact(period);

    for chunk in &mut chunks {
        let mut i = 0;
        while i + 3 < period {
            correlation += chunk[i] * sine_table[i]
                + chunk[i + 1] * sine_table[i + 1]
                + chunk[i + 2] * sine_table[i + 2]
                + chunk[i + 3] * sine_table[i + 3];
            i += 4;
        }
        while i < period {
            correlation += chunk[i] * sine_table[i];
            i += 1;
        }
    }

    let remainder = chunks.remainder();
    let mut i = 0;
    while i + 3 < remainder.len() {
        correlation += remainder[i] * sine_table[i]
            + remainder[i + 1] * sine_table[i + 1]
            + remainder[i + 2] * sine_table[i + 2]
            + remainder[i + 3] * sine_table[i + 3];
        i += 4;
    }
    while i < remainder.len() {
        correlation += remainder[i] * sine_table[i];
        i += 1;
    }

    correlation
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
    sine_table: Vec<f32>,
    sine_table_period: usize,
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
            sine_table: Vec::new(),
            sine_table_period: 0,
        }
    }

    pub fn config(&self) -> OscilloscopeConfig {
        self.config
    }

    fn ensure_sine_table(&mut self, period: usize) {
        if self.sine_table_period == period {
            return;
        }

        if period == 0 {
            self.sine_table.clear();
            self.sine_table_period = 0;
            return;
        }

        self.sine_table.resize(period, 0.0);

        let angle_step = std::f32::consts::TAU / period as f32;
        let (sin_step, cos_step) = angle_step.sin_cos();
        let mut sin_value = 0.0_f32;
        let mut cos_value = 1.0_f32;
        let mut refresh_counter = 0_usize;

        for value in &mut self.sine_table {
            *value = sin_value;

            let next_sin = sin_value * cos_step + cos_value * sin_step;
            let next_cos = cos_value * cos_step - sin_value * sin_step;

            sin_value = next_sin;
            cos_value = next_cos;

            refresh_counter += 1;
            if refresh_counter == 128 {
                let magnitude = (sin_value * sin_value + cos_value * cos_value).sqrt();
                if magnitude > f32::EPSILON {
                    sin_value /= magnitude;
                    cos_value /= magnitude;
                } else {
                    sin_value = 0.0;
                    cos_value = 1.0;
                }
                refresh_counter = 0;
            }
        }

        self.sine_table_period = period;
    }
}

impl AudioProcessor for OscilloscopeProcessor {
    type Output = OscilloscopeSnapshot;

    fn process_block(&mut self, block: &AudioBlock<'_>) -> ProcessorUpdate<Self::Output> {
        let channels = block.channels.max(1);
        if block.frame_count() == 0 {
            return ProcessorUpdate::None;
        }

        let base_frames = (self.config.sample_rate * self.config.segment_duration)
            .round()
            .max(1.0) as usize;
        let detection_frames = (self.config.sample_rate * 0.1) as usize;
        let capacity = detection_frames.max(base_frames) * channels;

        if !self.history.is_empty() && !self.history.len().is_multiple_of(channels) {
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
                let remove =
                    ((overflow - capacity).div_ceil(channels) * channels).min(self.history.len());
                self.history.drain(..remove);
            }
            self.history.extend(block.samples);
        }

        let available_frames = self.history.len() / channels;

        let (frames, start_frame) = match self.config.trigger_mode {
            TriggerMode::FreeRun => {
                let frames = base_frames.min(available_frames);
                if frames == 0 {
                    return ProcessorUpdate::None;
                }
                (frames, available_frames.saturating_sub(frames))
            }
            TriggerMode::Stable { num_cycles } => {
                if available_frames < base_frames {
                    return ProcessorUpdate::None;
                }

                {
                    let data = self.history.make_contiguous();
                    self.mono_buffer.clear();
                    self.mono_buffer.reserve(available_frames);

                    if channels == 1 {
                        self.mono_buffer
                            .extend_from_slice(&data[..available_frames]);
                    } else {
                        let scale = 1.0 / channels as f32;
                        for frame in 0..available_frames {
                            let idx = frame * channels;
                            let mut sum = 0.0_f32;
                            for ch in 0..channels {
                                sum += data[idx + ch];
                            }
                            self.mono_buffer.push(sum * scale);
                        }
                    }
                }

                let frequency = self
                    .pitch_detector
                    .detect_pitch(&self.mono_buffer, self.config.sample_rate)
                    .or(self.last_pitch);

                if let Some(freq) = frequency {
                    self.last_pitch = Some(freq);
                    let period = (self.config.sample_rate / freq).max(1.0) as usize;
                    self.ensure_sine_table(period);
                    let sine_table = self.sine_table.as_slice();
                    compute_trigger_region(
                        period,
                        num_cycles,
                        available_frames,
                        &self.mono_buffer,
                        sine_table,
                    )
                } else {
                    (base_frames, available_frames.saturating_sub(base_frames))
                }
            }
        };

        const TARGET_SAMPLES: usize = 4096;
        let target = TARGET_SAMPLES.clamp(1, frames);
        let extract_start = start_frame * channels;
        let data = self.history.make_contiguous();
        let extract_len = (frames * channels).min(data.len() - extract_start);

        self.snapshot.samples.clear();

        downsample_interleaved(
            &mut self.snapshot.samples,
            &data[extract_start..extract_start + extract_len],
            frames.min(extract_len / channels),
            channels,
            target,
        );

        self.snapshot.channels = channels;
        self.snapshot.samples_per_channel = target;

        ProcessorUpdate::Snapshot(self.snapshot.clone())
    }

    fn reset(&mut self) {
        self.snapshot = OscilloscopeSnapshot::default();
        self.history.clear();
        self.last_pitch = None;
        self.mono_buffer.clear();
        self.sine_table.clear();
        self.sine_table_period = 0;
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
    channels: usize,
    target: usize,
) {
    if frames == 0 || channels == 0 || target == 0 {
        return;
    }

    let step = frames as f32 / target as f32;

    for channel in 0..channels {
        for index in 0..target {
            let pos = index as f32 * step;
            let idx = pos as usize;
            let frac = pos - idx as f32;

            let sample = if frac > f32::EPSILON && idx + 1 < frames {
                let curr = data[idx * channels + channel];
                let next = data[(idx + 1) * channels + channel];
                crate::util::audio::lerp(curr, next, frac)
            } else {
                data[idx * channels + channel]
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

    #[test]
    fn produces_downsampled_snapshot_when_buffer_ready() {
        let config = OscilloscopeConfig {
            segment_duration: 0.01,
            trigger_mode: TriggerMode::FreeRun,
            ..Default::default()
        };
        let mut processor = OscilloscopeProcessor::new(config);

        let frames = (config.sample_rate * config.segment_duration)
            .round()
            .max(1.0) as usize;
        let mut samples = Vec::with_capacity(frames * 2);
        for frame in 0..frames {
            let t = frame as f32 / frames as f32;
            let sample = (t * std::f32::consts::TAU).sin();
            samples.push(sample);
            samples.push(-sample);
        }

        match processor.process_block(&make_block(&samples, 2, DEFAULT_SAMPLE_RATE)) {
            ProcessorUpdate::Snapshot(snapshot) => {
                assert_eq!(snapshot.channels, 2);
                assert!(snapshot.samples_per_channel > 0);
                assert!(snapshot.samples_per_channel <= 4096);
                assert_eq!(snapshot.samples.len(), snapshot.samples_per_channel * 2);
            }
            ProcessorUpdate::None => panic!("expected snapshot"),
        }
    }

    #[test]
    fn stable_mode_detects_pitch() {
        let mut processor = OscilloscopeProcessor::new(OscilloscopeConfig {
            segment_duration: 0.1,
            trigger_mode: TriggerMode::Stable { num_cycles: 2 },
            sample_rate: DEFAULT_SAMPLE_RATE,
        });

        let frequency = 440.0;
        let duration = 0.15;
        let total_samples = (DEFAULT_SAMPLE_RATE * duration) as usize;
        let mut samples = Vec::with_capacity(total_samples * 2);

        for i in 0..total_samples {
            let t = i as f32 / DEFAULT_SAMPLE_RATE;
            let sample = (t * frequency * std::f32::consts::TAU).sin();
            samples.push(sample);
            samples.push(sample);
        }

        match processor.process_block(&make_block(&samples, 2, DEFAULT_SAMPLE_RATE)) {
            ProcessorUpdate::Snapshot(snapshot) => {
                assert_eq!(snapshot.channels, 2);
                assert!(snapshot.samples_per_channel > 0);
                assert!(!snapshot.samples.is_empty());
            }
            ProcessorUpdate::None => panic!("expected snapshot"),
        }
    }
}
