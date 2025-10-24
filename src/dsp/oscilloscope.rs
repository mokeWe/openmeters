//! Oscilloscope/triggered waveform DSP implementation.

use super::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::util::audio::DEFAULT_SAMPLE_RATE;
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy)]
pub struct OscilloscopeConfig {
    pub sample_rate: f32,
    pub segment_duration: f32,
    pub trigger_level: f32,
    pub trigger_rising: bool,
    pub target_sample_count: usize,
}

impl Default for OscilloscopeConfig {
    fn default() -> Self {
        Self {
            sample_rate: DEFAULT_SAMPLE_RATE,
            segment_duration: 0.02,
            trigger_level: 0.0,
            trigger_rising: true,
            target_sample_count: 1_024,
        }
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
}

impl OscilloscopeProcessor {
    pub fn new(config: OscilloscopeConfig) -> Self {
        Self {
            config,
            snapshot: OscilloscopeSnapshot::default(),
            history: VecDeque::new(),
            history_channels: 0,
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
        }

        let frames = self.segment_frames();
        let capacity = frames * channels;

        if block.samples.len() >= capacity {
            self.history.clear();
            self.history.extend(
                block.samples[block.samples.len() - capacity..]
                    .iter()
                    .copied(),
            );
        } else {
            let overflow = self.history.len() + block.samples.len();
            if overflow > capacity {
                let remove =
                    (overflow - capacity).div_ceil(channels) * channels.min(self.history.len());
                self.history.drain(..remove);
            }
            self.history.extend(block.samples.iter().copied());
        }

        if self.history.len() < capacity {
            return ProcessorUpdate::None;
        }

        let data = self.history.make_contiguous();
        let target = self.config.target_sample_count.clamp(1, frames);

        self.snapshot.samples.clear();
        self.snapshot.samples.reserve(target * channels);

        let trigger = find_trigger(
            data,
            frames,
            channels,
            self.config.trigger_level,
            self.config.trigger_rising,
        );
        downsample_interleaved(
            &mut self.snapshot.samples,
            data,
            frames,
            channels,
            target,
            trigger,
        );

        self.snapshot.channels = channels;
        self.snapshot.samples_per_channel = target;

        ProcessorUpdate::Snapshot(self.snapshot.clone())
    }

    fn reset(&mut self) {
        self.snapshot = OscilloscopeSnapshot::default();
        self.history.clear();
        self.history_channels = 0;
    }
}

impl Reconfigurable<OscilloscopeConfig> for OscilloscopeProcessor {
    fn update_config(&mut self, config: OscilloscopeConfig) {
        self.config = config;
        self.reset();
    }
}

fn find_trigger(data: &[f32], frames: usize, channels: usize, level: f32, rising: bool) -> usize {
    if frames < 2 || channels == 0 {
        return 0;
    }

    let mut prev = data[0];
    for frame in 1..frames {
        let current = data[frame * channels];
        let crossed = if rising {
            prev < level && current >= level
        } else {
            prev > level && current <= level
        };
        if crossed {
            return frame;
        }
        prev = current;
    }

    frames / 2
}

fn downsample_interleaved(
    output: &mut Vec<f32>,
    data: &[f32],
    frames: usize,
    channels: usize,
    target: usize,
    start_frame: usize,
) {
    if frames == 0 || channels == 0 || target == 0 {
        return;
    }

    let base = start_frame % frames;
    for channel in 0..channels {
        for index in 0..target {
            let frame = (base + index * frames / target) % frames;
            output.push(data[frame * channels + channel]);
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
            trigger_level: 0.0,
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
            trigger_level: 0.5,
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
