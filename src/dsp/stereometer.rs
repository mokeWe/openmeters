//! Stereometer (vector scope & correlation meter) DSP scaffolding.

use super::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::util::audio::DEFAULT_SAMPLE_RATE;

/// Correlation range bounded between -1.0 and 1.0.
pub type Correlation = f32;

/// Configuration controlling the stereometer response.
#[derive(Debug, Clone, Copy)]
pub struct StereometerConfig {
    pub sample_rate: f32,
    /// Window length in seconds for correlation analysis.
    pub correlation_window: f32,
    /// Decay factor applied to previous XY traces (0.0-1.0).
    pub persistence: f32,
}

impl Default for StereometerConfig {
    fn default() -> Self {
        Self {
            sample_rate: DEFAULT_SAMPLE_RATE,
            correlation_window: 0.1,
            persistence: 0.8,
        }
    }
}

/// Snapshot containing the latest stereometer data.
#[derive(Debug, Clone)]
pub struct StereometerSnapshot {
    /// XY points representing the L/R vectorscope trace.
    pub xy_points: Vec<(f32, f32)>,
    /// Correlation coefficient computed over the recent window.
    pub correlation: Correlation,
}

impl Default for StereometerSnapshot {
    fn default() -> Self {
        Self {
            xy_points: Vec::new(),
            correlation: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StereometerProcessor {
    config: StereometerConfig,
    snapshot: StereometerSnapshot,
}

impl StereometerProcessor {
    pub fn new(config: StereometerConfig) -> Self {
        Self {
            config,
            snapshot: StereometerSnapshot::default(),
        }
    }
}

impl AudioProcessor for StereometerProcessor {
    type Output = StereometerSnapshot;

    fn process_block(&mut self, _block: &AudioBlock<'_>) -> ProcessorUpdate<Self::Output> {
        // TODO: compute XY trace & correlation from stereo pairs.
        ProcessorUpdate::None
    }

    fn reset(&mut self) {
        self.snapshot = StereometerSnapshot::default();
    }
}

impl Reconfigurable<StereometerConfig> for StereometerProcessor {
    fn update_config(&mut self, config: StereometerConfig) {
        self.config = config;
        self.reset();
    }
}
