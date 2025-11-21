//! Oscilloscope/triggered waveform DSP implementation.

use super::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::util::audio::DEFAULT_SAMPLE_RATE;
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy)]
pub struct OscilloscopeConfig {
    pub sample_rate: f32,
    pub segment_duration: f32,
    pub trigger_rising: bool,
    pub target_sample_count: usize,
    pub hysteresis: f32,
    pub min_slope: f32,
}

impl Default for OscilloscopeConfig {
    fn default() -> Self {
        Self {
            sample_rate: DEFAULT_SAMPLE_RATE,
            segment_duration: 0.02,
            trigger_rising: true,
            target_sample_count: 4_096,
            hysteresis: 0.02,
            min_slope: 0.01,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct Prefilter {
    prev1: f32,
    prev2: f32,
}

impl Prefilter {
    fn process(&mut self, sample: f32) -> f32 {
        let filtered = self.prev2 * 0.25 + self.prev1 * 0.5 + sample * 0.25;
        self.prev2 = self.prev1;
        self.prev1 = sample;
        filtered
    }

    fn reset(&mut self, sample: f32) {
        self.prev1 = sample;
        self.prev2 = sample;
    }
}

#[derive(Debug, Clone)]
struct Trigger {
    prefilter: Prefilter,
}

impl Trigger {
    fn new() -> Self {
        Self {
            prefilter: Prefilter::default(),
        }
    }

    fn reset(&mut self) {
        self.prefilter = Prefilter::default();
    }

    fn find_trigger(
        &mut self,
        data: &[f32],
        frames: usize,
        channels: usize,
        config: &OscilloscopeConfig,
    ) -> Option<(usize, f32)> {
        if frames < 2 || channels == 0 {
            return None;
        }

        let upper = config.hysteresis;
        let lower = -config.hysteresis;
        let min_slope = config.min_slope;
        let rising = config.trigger_rising;

        let mono_sum: Vec<f32> = (0..frames)
            .map(|frame| {
                let sum: f32 = (0..channels).map(|ch| data[frame * channels + ch]).sum();
                sum / channels as f32
            })
            .collect();

        self.prefilter.reset(mono_sum[0]);
        let filtered: Vec<f32> = mono_sum
            .iter()
            .map(|&s| self.prefilter.process(s))
            .collect();

        let mut armed = false;
        let mut prev = filtered[0];

        for (frame, &curr) in filtered.iter().enumerate().skip(1) {
            armed = armed || if rising { curr < lower } else { curr > upper };

            if armed && (curr - prev).abs() >= min_slope {
                let crossed = if rising {
                    prev < 0.0 && curr >= 0.0
                } else {
                    prev > 0.0 && curr <= 0.0
                };

                if crossed {
                    let t = (prev / (prev - curr)).clamp(0.0, 1.0);
                    return Some((frame, t));
                }
            }

            prev = curr;
        }

        None
    }
}

#[derive(Debug, Clone)]
pub struct OscilloscopeSnapshot {
    pub channels: usize,
    pub samples: Vec<f32>,
    pub samples_per_channel: usize,
}

impl Default for OscilloscopeSnapshot {
    fn default() -> Self {
        Self {
            channels: 2,
            samples: Vec::new(),
            samples_per_channel: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OscilloscopeProcessor {
    config: OscilloscopeConfig,
    snapshot: OscilloscopeSnapshot,
    history: VecDeque<f32>,
    history_channels: usize,
    trigger: Trigger,
    freerun_position: usize,
}

impl OscilloscopeProcessor {
    pub fn new(config: OscilloscopeConfig) -> Self {
        Self {
            config,
            snapshot: OscilloscopeSnapshot::default(),
            history: VecDeque::new(),
            history_channels: 0,
            trigger: Trigger::new(),
            freerun_position: 0,
        }
    }

    pub fn config(&self) -> OscilloscopeConfig {
        self.config
    }

    pub fn snapshot(&self) -> &OscilloscopeSnapshot {
        &self.snapshot
    }

    fn segment_frames(&self) -> usize {
        (self.config.sample_rate * self.config.segment_duration)
            .round()
            .max(1.0) as usize
    }
}

impl AudioProcessor for OscilloscopeProcessor {
    type Output = OscilloscopeSnapshot;

    fn process_block(&mut self, block: &AudioBlock<'_>) -> ProcessorUpdate<Self::Output> {
        let channels = block.channels.max(1);
        if block.frame_count() == 0 {
            return ProcessorUpdate::None;
        }

        if self.history_channels != channels {
            self.history.clear();
            self.history_channels = channels;
            self.trigger.reset();
        }

        let frames = self.segment_frames();
        let capacity = frames * channels;

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

        if self.history.len() < capacity {
            return ProcessorUpdate::None;
        }

        let data = self.history.make_contiguous();
        let target = self.config.target_sample_count.clamp(1, frames);

        self.snapshot.samples.clear();
        self.snapshot.samples.reserve(target * channels);

        let (trigger_frame, sub_sample) = self
            .trigger
            .find_trigger(data, frames, channels, &self.config)
            .map(|result| {
                self.freerun_position = 0;
                result
            })
            .unwrap_or_else(|| {
                let advance = (target / 10).max(1);
                self.freerun_position = (self.freerun_position + advance) % frames;
                (self.freerun_position, 0.0)
            });

        downsample_interleaved(
            &mut self.snapshot.samples,
            data,
            frames,
            channels,
            target,
            trigger_frame,
            sub_sample,
        );

        self.snapshot.channels = channels;
        self.snapshot.samples_per_channel = target;

        ProcessorUpdate::Snapshot(self.snapshot.clone())
    }

    fn reset(&mut self) {
        self.snapshot = OscilloscopeSnapshot::default();
        self.history.clear();
        self.history_channels = 0;
        self.trigger.reset();
        self.freerun_position = 0;
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
    start_frame: usize,
    sub_sample_offset: f32,
) {
    if frames == 0 || channels == 0 || target == 0 {
        return;
    }

    let base = start_frame % frames;
    let step = frames as f32 / target as f32;

    for channel in 0..channels {
        for index in 0..target {
            let frame_pos = base as f32 + index as f32 * step + sub_sample_offset;
            let frame_int = frame_pos as usize % frames;
            let frac = frame_pos.fract();

            let sample = if frac > f32::EPSILON && frame_int + 1 < frames {
                let curr = data[frame_int * channels + channel];
                let next = data[(frame_int + 1) * channels + channel];
                curr + (next - curr) * frac
            } else {
                data[frame_int * channels + channel]
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
        let mut processor = OscilloscopeProcessor::new(OscilloscopeConfig {
            segment_duration: 0.01,
            target_sample_count: 64,
            ..Default::default()
        });

        let frames = processor.segment_frames();
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
                assert_eq!(snapshot.samples_per_channel, 64);
                assert_eq!(snapshot.samples.len(), 128);
            }
            ProcessorUpdate::None => panic!("expected snapshot"),
        }
    }

    #[test]
    fn trigger_rotation_aligns_snapshot() {
        let mut processor = OscilloscopeProcessor::new(OscilloscopeConfig {
            segment_duration: 0.01,
            target_sample_count: 32,
            ..Default::default()
        });

        let frames = processor.segment_frames();
        let mut samples = Vec::with_capacity(frames * 2);
        for frame in 0..frames {
            let value = if frame < frames / 2 { -1.0 } else { 1.0 };
            samples.push(value);
            samples.push(value);
        }

        let snapshot = match processor.process_block(&make_block(&samples, 2, DEFAULT_SAMPLE_RATE))
        {
            ProcessorUpdate::Snapshot(snapshot) => snapshot,
            ProcessorUpdate::None => panic!("expected snapshot"),
        };

        let first_left = snapshot.samples[0];
        assert!(first_left >= -1.0);
        assert!(first_left <= 1.0);
    }
}
