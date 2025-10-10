//! Core DSP scaffolding for upcoming audio visualisations.
//!
//! This module only provides the contracts and lightweight plumbing needed to
//! integrate future processors. Implementations can iterate on these building
//! blocks without having to reshape the public surface area.

pub mod loudness;
pub mod oscilloscope;
pub mod spectrogram;
pub mod spectrum;
pub mod waveform;

use std::time::Instant;

/// Borrowed audio samples provided to DSP processors.
#[derive(Debug, Clone, Copy)]
pub struct AudioBlock<'a> {
    /// Interleaved PCM samples.
    pub samples: &'a [f32],
    /// Number of channels encoded in `samples`.
    pub channels: usize,
    /// Sample-rate of the upstream capture pipeline.
    pub sample_rate: f32,
    /// Timestamp associated with the beginning of this block.
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

    /// Returns the length of the block in frames.
    pub fn frame_count(&self) -> usize {
        self.samples.len() / self.channels.max(1)
    }
}

/// Output emitted by a processor after consuming an [`AudioBlock`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessorUpdate<T> {
    /// No new result is ready for downstream consumers.
    None,
    /// A fresh snapshot is available.
    Snapshot(T),
}

/// Shared contract implemented by all DSP modules.
pub trait AudioProcessor {
    type Output;

    /// Consume a block of audio and optionally output an updated snapshot.
    fn process_block(&mut self, block: &AudioBlock<'_>) -> ProcessorUpdate<Self::Output>;

    /// Reset the processor, clearing any accumulated history.
    fn reset(&mut self);
}

/// Optional helper trait for processors that expose lightweight configuration updates.
pub trait Reconfigurable<Cfg> {
    fn update_config(&mut self, config: Cfg);
}
