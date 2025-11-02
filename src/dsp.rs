pub mod loudness;
pub mod oscilloscope;
pub mod spectrogram;
pub mod spectrum;
pub mod waveform;

use std::time::Instant;

#[derive(Debug, Clone, Copy)]
pub struct AudioBlock<'a> {
    pub samples: &'a [f32],
    pub channels: usize,
    pub sample_rate: f32,
    pub timestamp: Instant,
}

impl<'a> AudioBlock<'a> {
    pub fn new(samples: &'a [f32], channels: usize, sample_rate: f32, timestamp: Instant) -> Self {
        Self {
            samples,
            channels,
            sample_rate,
            timestamp,
        }
    }

    pub fn frame_count(&self) -> usize {
        self.samples.len() / self.channels.max(1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessorUpdate<T> {
    None,
    Snapshot(T),
}

pub trait AudioProcessor {
    type Output;

    fn process_block(&mut self, block: &AudioBlock<'_>) -> ProcessorUpdate<Self::Output>;
    fn reset(&mut self);
}

pub trait Reconfigurable<Cfg> {
    fn update_config(&mut self, config: Cfg);
}
