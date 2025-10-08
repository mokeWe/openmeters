//! Real-time waveform DSP scaffolding.

use super::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::util::audio::DEFAULT_SAMPLE_RATE;

/// Strategy used to downsample the waveform when pixel budget is limited.
#[derive(Debug, Clone, Copy)]
pub enum DownsampleStrategy {
    /// Keep min/max values per bucket (preserves peaks).
    MinMax,
    /// Simple averaging (cheaper but can hide transients).
    Average,
}

/// Configuration for the waveform preview.
#[derive(Debug, Clone, Copy)]
pub struct WaveformConfig {
    pub sample_rate: f32,
    /// How many frames of history to retain (seconds).
    pub history_window: f32,
    /// Target number of samples per channel returned in each snapshot.
    pub target_sample_count: usize,
    pub downsample: DownsampleStrategy,
}

impl Default for WaveformConfig {
    fn default() -> Self {
        Self {
            sample_rate: DEFAULT_SAMPLE_RATE,
            history_window: 0.5,
            target_sample_count: 1_024,
            downsample: DownsampleStrategy::MinMax,
        }
    }
}

/// Snapshot storing resampled waveform data.
#[derive(Debug, Clone)]
pub struct WaveformSnapshot {
    pub channels: usize,
    pub samples: Vec<f32>,
}

impl Default for WaveformSnapshot {
    fn default() -> Self {
        Self {
            channels: 2,
            samples: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct WaveformProcessor {
    config: WaveformConfig,
    snapshot: WaveformSnapshot,
}

impl WaveformProcessor {
    pub fn new(config: WaveformConfig) -> Self {
        Self {
            config,
            snapshot: WaveformSnapshot::default(),
        }
    }
}

impl AudioProcessor for WaveformProcessor {
    type Output = WaveformSnapshot;

    fn process_block(&mut self, _block: &AudioBlock<'_>) -> ProcessorUpdate<Self::Output> {
        // TODO: append samples to history buffer and produce downsampled snapshot.
        ProcessorUpdate::None
    }

    fn reset(&mut self) {
        self.snapshot = WaveformSnapshot::default();
    }
}

impl Reconfigurable<WaveformConfig> for WaveformProcessor {
    fn update_config(&mut self, config: WaveformConfig) {
        self.config = config;
        self.reset();
    }
}
