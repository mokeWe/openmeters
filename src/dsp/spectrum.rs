//! Spectrum analyser DSP scaffolding.

use super::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::dsp::spectrogram::{SpectrogramConfig, WindowKind};
use crate::util::audio::DEFAULT_SAMPLE_RATE;

/// Output magnitude spectrum.
#[derive(Debug, Clone, Default)]
pub struct SpectrumSnapshot {
    pub frequency_bins: Vec<f32>,
    pub magnitudes_db: Vec<f32>,
    pub peak_frequency_hz: Option<f32>,
}

/// Configuration for the spectrum analyser.
#[derive(Debug, Clone, Copy)]
pub struct SpectrumConfig {
    pub sample_rate: f32,
    pub fft_size: usize,
    pub window: WindowKind,
    pub averaging: AveragingMode,
}

impl Default for SpectrumConfig {
    fn default() -> Self {
        Self {
            sample_rate: DEFAULT_SAMPLE_RATE,
            fft_size: 2048,
            window: WindowKind::Hann,
            averaging: AveragingMode::Exponential { factor: 0.5 },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum AveragingMode {
    None,
    Exponential { factor: f32 },
    PeakHold { decay_per_second: f32 },
}

#[derive(Debug, Clone)]
pub struct SpectrumProcessor {
    config: SpectrumConfig,
    /// Potential re-use of spectrogram staging buffer for FFT setup.
    pub(crate) staging: Option<SpectrogramConfig>,
    snapshot: SpectrumSnapshot,
}

impl SpectrumProcessor {
    pub fn new(config: SpectrumConfig) -> Self {
        Self {
            config,
            staging: None,
            snapshot: SpectrumSnapshot::default(),
        }
    }
}

impl AudioProcessor for SpectrumProcessor {
    type Output = SpectrumSnapshot;

    fn process_block(&mut self, _block: &AudioBlock<'_>) -> ProcessorUpdate<Self::Output> {
        // TODO: perform FFT, convert to magnitudes, apply averaging.
        ProcessorUpdate::None
    }

    fn reset(&mut self) {
        self.snapshot = SpectrumSnapshot::default();
    }
}

impl Reconfigurable<SpectrumConfig> for SpectrumProcessor {
    fn update_config(&mut self, config: SpectrumConfig) {
        self.config = config;
        self.reset();
    }
}
