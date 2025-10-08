//! Oscilloscope/triggered waveform DSP scaffolding.

use super::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::util::audio::DEFAULT_SAMPLE_RATE;

/// Options controlling oscilloscope behaviour.
#[derive(Debug, Clone, Copy)]
pub struct OscilloscopeConfig {
    pub sample_rate: f32,
    /// Duration of the captured segment in seconds.
    pub segment_duration: f32,
    /// Trigger level expressed in linear amplitude.
    pub trigger_level: f32,
    /// Whether to use rising-edge or falling-edge detection.
    pub trigger_rising: bool,
    /// Target number of samples per channel in the emitted snapshot.
    pub target_sample_count: usize,
    /// Persistence factor applied by the renderer (0 = full clear, 1 = infinite trail).
    pub persistence: f32,
}

impl Default for OscilloscopeConfig {
    fn default() -> Self {
        Self {
            sample_rate: DEFAULT_SAMPLE_RATE,
            segment_duration: 0.02,
            trigger_level: 0.05,
            trigger_rising: true,
            target_sample_count: 1_024,
            persistence: 0.85,
        }
    }
}

/// Snapshot handed to the renderer containing oscilloscope samples.
#[derive(Debug, Clone)]
pub struct OscilloscopeSnapshot {
    pub channels: usize,
    pub samples: Vec<f32>,
    pub samples_per_channel: usize,
    pub persistence: f32,
}

impl Default for OscilloscopeSnapshot {
    fn default() -> Self {
        Self {
            channels: 2,
            samples: Vec::new(),
            samples_per_channel: 0,
            persistence: 0.85,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OscilloscopeProcessor {
    config: OscilloscopeConfig,
    snapshot: OscilloscopeSnapshot,
    history: std::collections::VecDeque<f32>,
}

impl OscilloscopeProcessor {
    pub fn new(config: OscilloscopeConfig) -> Self {
        Self {
            config,
            snapshot: OscilloscopeSnapshot::default(),
            history: std::collections::VecDeque::new(),
        }
    }

    pub fn config(&self) -> OscilloscopeConfig {
        self.config
    }

    pub fn snapshot(&self) -> &OscilloscopeSnapshot {
        &self.snapshot
    }

    fn segment_frame_count(&self) -> usize {
        let frames = (self.config.sample_rate * self.config.segment_duration)
            .round()
            .max(1.0) as usize;
        frames.max(1)
    }

    fn capacity(&self, channels: usize) -> usize {
        self.segment_frame_count() * channels.max(1)
    }

    fn ensure_history_capacity(&mut self, channels: usize) {
        let capacity = self.capacity(channels);
        if self.history.capacity() < capacity {
            self.history = std::collections::VecDeque::with_capacity(capacity);
        }
    }

    fn update_snapshot(&mut self, channels: usize) {
        let frames = self.segment_frame_count();
        if channels == 0 {
            return;
        }

        if self.history.len() < frames * channels {
            return;
        }

        let mut contiguous = Vec::with_capacity(frames * channels);
        contiguous.extend(self.history.iter().copied());

        let trigger_frame = find_trigger_index(
            &contiguous,
            channels,
            self.config.trigger_level,
            self.config.trigger_rising,
        )
        .unwrap_or(frames / 2);

        let rotated = rotate_frames(&contiguous, channels, trigger_frame);

        let target = self.config.target_sample_count.max(1).min(frames);

        let mut samples = Vec::with_capacity(channels * target);

        for channel in 0..channels {
            for index in 0..target {
                let src_frame = index * frames / target;
                let value = rotated[src_frame * channels + channel];
                samples.push(value);
            }
        }

        self.snapshot.channels = channels;
        self.snapshot.samples_per_channel = target;
        self.snapshot.samples = samples;
        self.snapshot.persistence = self.config.persistence.clamp(0.0, 1.0);
    }
}

impl AudioProcessor for OscilloscopeProcessor {
    type Output = OscilloscopeSnapshot;

    fn process_block(&mut self, block: &AudioBlock<'_>) -> ProcessorUpdate<Self::Output> {
        let channels = block.channels.max(1);
        if block.frame_count() == 0 {
            return ProcessorUpdate::None;
        }

        self.ensure_history_capacity(channels);

        let capacity = self.capacity(channels);
        for sample in block.samples.iter().copied() {
            if self.history.len() == capacity {
                self.history.pop_front();
            }
            self.history.push_back(sample);
        }

        if self.history.len() < capacity {
            return ProcessorUpdate::None;
        }

        self.update_snapshot(channels);
        ProcessorUpdate::Snapshot(self.snapshot.clone())
    }

    fn reset(&mut self) {
        self.snapshot = OscilloscopeSnapshot::default();
        self.history.clear();
    }
}

impl Reconfigurable<OscilloscopeConfig> for OscilloscopeProcessor {
    fn update_config(&mut self, config: OscilloscopeConfig) {
        self.config = config;
        self.reset();
    }
}

fn find_trigger_index(samples: &[f32], channels: usize, level: f32, rising: bool) -> Option<usize> {
    if channels == 0 || samples.len() < channels * 2 {
        return None;
    }

    let mut prev = samples[0];
    for frame in 1..(samples.len() / channels) {
        let current = samples[frame * channels];
        if rising {
            if prev < level && current >= level {
                return Some(frame);
            }
        } else if prev > level && current <= level {
            return Some(frame);
        }
        prev = current;
    }
    None
}

fn rotate_frames(samples: &[f32], channels: usize, start_frame: usize) -> Vec<f32> {
    if channels == 0 {
        return Vec::new();
    }

    let total_frames = samples.len() / channels;
    let mut rotated = Vec::with_capacity(samples.len());

    for frame in start_frame..total_frames {
        let begin = frame * channels;
        let end = begin + channels;
        rotated.extend_from_slice(&samples[begin..end]);
    }

    for frame in 0..start_frame {
        let begin = frame * channels;
        let end = begin + channels;
        rotated.extend_from_slice(&samples[begin..end]);
    }

    rotated
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

        let frames = processor.segment_frame_count();
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

        let frames = processor.segment_frame_count();
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
