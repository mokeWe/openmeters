//! Stereometer (vector scope & correlation meter) DSP implementation.

use super::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::util::audio::DEFAULT_SAMPLE_RATE;
use std::collections::VecDeque;

pub type Correlation = f32;

/// Configuration controlling the stereometer response.
#[derive(Debug, Clone, Copy)]
pub struct StereometerConfig {
    pub sample_rate: f32,
    pub segment_duration: f32,
    pub target_sample_count: usize,
}

impl Default for StereometerConfig {
    fn default() -> Self {
        Self {
            sample_rate: DEFAULT_SAMPLE_RATE,
            segment_duration: 0.02,
            target_sample_count: 1_024,
        }
    }
}

/// Snapshot containing the latest stereometer data.
#[derive(Debug, Clone, Default)]
pub struct StereometerSnapshot {
    pub xy_points: Vec<(f32, f32)>,
    pub correlation: Correlation,
}

#[derive(Debug, Clone)]
pub struct StereometerProcessor {
    config: StereometerConfig,
    snapshot: StereometerSnapshot,
    history: VecDeque<f32>,
    history_channels: usize,
}

impl StereometerProcessor {
    pub fn new(config: StereometerConfig) -> Self {
        Self {
            config,
            snapshot: StereometerSnapshot::default(),
            history: VecDeque::new(),
            history_channels: 0,
        }
    }

    pub fn config(&self) -> StereometerConfig {
        self.config
    }

    pub fn snapshot(&self) -> &StereometerSnapshot {
        &self.snapshot
    }

    fn segment_frames(&self) -> usize {
        (self.config.sample_rate * self.config.segment_duration)
            .round()
            .max(1.0) as usize
    }
}

impl AudioProcessor for StereometerProcessor {
    type Output = StereometerSnapshot;

    fn process_block(&mut self, block: &AudioBlock<'_>) -> ProcessorUpdate<Self::Output> {
        let channels = block.channels.max(1);
        if block.frame_count() == 0 || channels < 2 {
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
            self.history
                .extend(&block.samples[block.samples.len() - capacity..]);
        } else {
            let overflow = self.history.len() + block.samples.len();
            if overflow > capacity {
                let remove =
                    (overflow - capacity).div_ceil(channels) * channels.min(self.history.len());
                self.history.drain(..remove);
            }
            self.history.extend(block.samples);
        }

        if self.history.len() < capacity {
            return ProcessorUpdate::None;
        }

        let data = self.history.make_contiguous();
        let target = self.config.target_sample_count.clamp(1, frames);

        // Extract XY pairs from stereo data
        self.snapshot.xy_points.clear();
        self.snapshot.xy_points.reserve(target);

        for index in 0..target {
            let frame = index * frames / target;
            let left = data[frame * channels];
            let right = data[frame * channels + 1];
            self.snapshot.xy_points.push((left, right));
        }

        // Placeholder correlation value
        self.snapshot.correlation = 0.0;

        ProcessorUpdate::Snapshot(self.snapshot.clone())
    }

    fn reset(&mut self) {
        self.snapshot = StereometerSnapshot::default();
        self.history.clear();
        self.history_channels = 0;
    }
}

impl Reconfigurable<StereometerConfig> for StereometerProcessor {
    fn update_config(&mut self, config: StereometerConfig) {
        self.config = config;
        self.reset();
    }
}
