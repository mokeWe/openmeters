//! Oscilloscope/triggered waveform DSP implementation.

use super::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::util::audio::{DEFAULT_SAMPLE_RATE, interpolate_linear};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use wide::f32x4;

const PITCH_MIN_HZ: f32 = 10.0;
const PITCH_MAX_HZ: f32 = 10000.0;
// lower = more sensitive. 0.15 was chosen practically arbitrarily but works.
const PITCH_THRESHOLD: f32 = 0.15;
// Could be lowered, but at the cost of accuracy. Here for future use maybe.
const PITCH_DOWNSAMPLE_RATE: f32 = 48_000.0;

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
    diff: Vec<f32>,
    cmean: Vec<f32>,
    downsampled: Vec<f32>,
    energy: Vec<f32>,
    left_simd: Vec<f32x4>,
    left_tail: [f32; 4],
    left_tail_len: usize,
}

impl PitchDetector {
    fn new() -> Self {
        Self {
            diff: Vec::new(),
            cmean: Vec::new(),
            downsampled: Vec::new(),
            energy: Vec::new(),
            left_simd: Vec::new(),
            left_tail: [0.0; 4],
            left_tail_len: 0,
        }
    }

    fn detect_pitch(&mut self, samples: &[f32], rate: f32) -> Option<f32> {
        if samples.is_empty() {
            return None;
        }

        let data = if rate > PITCH_DOWNSAMPLE_RATE * 1.5 {
            self.downsample(samples, rate, PITCH_DOWNSAMPLE_RATE);
            &self.downsampled
        } else {
            samples
        };

        let work_rate = if rate > PITCH_DOWNSAMPLE_RATE * 1.5 {
            PITCH_DOWNSAMPLE_RATE
        } else {
            rate
        };

        let min_period = (work_rate / PITCH_MAX_HZ).max(2.0) as usize;
        let max_period = (work_rate / PITCH_MIN_HZ).min(data.len() as f32 / 2.0) as usize;

        if max_period <= min_period || data.len() < max_period * 2 {
            return None;
        }

        self.diff.resize(max_period, 0.0);
        self.cmean.resize(max_period, 0.0);
        self.energy.resize(data.len() + 1, 0.0);

        let len = data.len() - max_period;
        let precise_limit = 256.min(max_period);

        let mut acc = 0.0_f32;
        self.energy[0] = 0.0;
        for (i, &s) in data.iter().enumerate() {
            acc = s.mul_add(s, acc);
            self.energy[i + 1] = acc;
        }

        let left_energy = self.energy[len];
        let left = &data[..len];

        if left.is_empty() {
            return None;
        }

        self.left_simd.clear();
        self.left_tail.fill(0.0);
        let (chunks, tail) = left.as_chunks::<4>();
        self.left_simd.reserve(chunks.len());
        for chunk in chunks {
            self.left_simd.push(f32x4::new(*chunk));
        }
        self.left_tail_len = tail.len();
        if self.left_tail_len > 0 {
            self.left_tail[..self.left_tail_len].copy_from_slice(tail);
        }

        // 1. Precise region (stride = 1)
        for tau in 0..precise_limit {
            let right = &data[tau..tau + len];
            let base = self.energy[tau];
            let right_energy = self.energy[tau + len] - base;
            self.diff[tau] = Self::diff_at_tau(
                &self.left_simd,
                &self.left_tail[..self.left_tail_len],
                right,
                left_energy,
                right_energy,
            );
        }

        // 2. Strided region
        let stride = ((max_period - precise_limit) / 64).max(1);
        for tau in (precise_limit..max_period).step_by(stride) {
            let right = &data[tau..tau + len];
            let base = self.energy[tau];
            let right_energy = self.energy[tau + len] - base;
            self.diff[tau] = Self::diff_at_tau(
                &self.left_simd,
                &self.left_tail[..self.left_tail_len],
                right,
                left_energy,
                right_energy,
            );
        }

        // 3. Interpolate strided region
        for tau in precise_limit..max_period {
            if (tau - precise_limit) % stride != 0 {
                let prev = precise_limit + ((tau - precise_limit) / stride) * stride;
                let next = prev + stride;
                if next < max_period {
                    let t = (tau - prev) as f32 / stride as f32;
                    self.diff[tau] = self.diff[prev] * (1.0 - t) + self.diff[next] * t;
                } else {
                    self.diff[tau] = self.diff[prev];
                }
            }
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

        for tau in min_period..max_period {
            if self.cmean[tau] < PITCH_THRESHOLD
                && tau + 1 < max_period
                && self.cmean[tau] < self.cmean[tau + 1]
            {
                let confidence = 1.0 - self.cmean[tau];
                if confidence > PITCH_THRESHOLD {
                    return Some(work_rate / tau as f32);
                }
            }
        }

        None
    }

    #[inline(always)]
    fn diff_at_tau(
        left_chunks: &[f32x4],
        left_tail: &[f32],
        right: &[f32],
        left_energy: f32,
        right_energy: f32,
    ) -> f32 {
        let expected_len = left_chunks.len() * 4 + left_tail.len();
        debug_assert_eq!(expected_len, right.len());

        if expected_len == 0 {
            return 0.0;
        }

        let dot = simd_dot_product(left_chunks, left_tail, right);
        let diff = left_energy + right_energy - 2.0 * dot;
        diff.max(0.0)
    }

    fn downsample(&mut self, samples: &[f32], src_rate: f32, dst_rate: f32) {
        let ratio = src_rate / dst_rate;
        let len = (samples.len() as f32 / ratio) as usize;

        self.downsampled.clear();
        self.downsampled.reserve(len);

        for i in 0..len {
            self.downsampled
                .push(interpolate_linear(samples, i as f32 * ratio));
        }
    }
}

#[inline(always)]
fn simd_dot_product(left_chunks: &[f32x4], left_tail: &[f32], right: &[f32]) -> f32 {
    let (right_chunks, right_tail) = right.as_chunks::<4>();
    debug_assert_eq!(right_chunks.len(), left_chunks.len());
    debug_assert_eq!(right_tail.len(), left_tail.len());

    let mut left_iter = left_chunks.chunks_exact(4);
    let mut right_iter = right_chunks.chunks_exact(4);

    let mut acc0 = f32x4::splat(0.0);
    let mut acc1 = f32x4::splat(0.0);
    let mut acc2 = f32x4::splat(0.0);
    let mut acc3 = f32x4::splat(0.0);

    for (left_block, right_block) in left_iter.by_ref().zip(right_iter.by_ref()) {
        acc0 = left_block[0].mul_add(f32x4::new(right_block[0]), acc0);
        acc1 = left_block[1].mul_add(f32x4::new(right_block[1]), acc1);
        acc2 = left_block[2].mul_add(f32x4::new(right_block[2]), acc2);
        acc3 = left_block[3].mul_add(f32x4::new(right_block[3]), acc3);
    }

    let mut acc = (acc0 + acc1) + (acc2 + acc3);

    for (left_chunk, right_chunk) in left_iter.remainder().iter().zip(right_iter.remainder()) {
        acc = left_chunk.mul_add(f32x4::new(*right_chunk), acc);
    }

    let dot = acc.reduce_add();

    let tail_dot = match left_tail.len() {
        0 => 0.0,
        1 => left_tail[0] * right_tail[0],
        2 => left_tail[0].mul_add(right_tail[0], left_tail[1] * right_tail[1]),
        3 => {
            let partial = left_tail[0].mul_add(right_tail[0], left_tail[1] * right_tail[1]);
            left_tail[2].mul_add(right_tail[2], partial)
        }
        _ => {
            let mut scalar = 0.0;
            for (l, r) in left_tail.iter().zip(right_tail.iter()) {
                scalar = l.mul_add(*r, scalar);
            }
            scalar
        }
    };

    dot + tail_dot
}

// Stores cached sine/cosine prefix sums so trigger correlation reuses a single pass.
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
            self.history.clear(); // will drop on channel count change. sort of an edge case
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

    #[test]
    fn high_frequency_pitch_detection() {
        let mut detector = PitchDetector::new();
        let sample_rate = 48_000.0;
        // yeah this is a lofty goal. pitch detection is fucking expensive.
        // still made it work
        let frequency = 12_000.0;
        let duration = 0.05;
        let total_samples = (sample_rate * duration) as usize;
        let mut samples = Vec::with_capacity(total_samples);

        for i in 0..total_samples {
            let t = i as f32 / sample_rate;
            let sample = (t * frequency * std::f32::consts::TAU).sin();
            samples.push(sample);
        }

        let detected = detector.detect_pitch(&samples, sample_rate);
        assert!(detected.is_some(), "Failed to detect 12kHz pitch");
        let detected_freq = detected.unwrap();
        assert!(
            (detected_freq - frequency).abs() < frequency * 0.05,
            "Detected frequency {} is too far from expected {}",
            detected_freq,
            frequency
        );
    }
}
