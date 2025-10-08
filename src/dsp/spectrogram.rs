//! Spectrogram DSP implementation built on a short-time Fourier transform.
//!
// IMPORTANT TODO: spectrum reassignment method!

use super::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::util::audio::DEFAULT_SAMPLE_RATE;
use realfft::{RealFftPlanner, RealToComplex};
use rustc_hash::FxHashMap;
use rustfft::num_complex::Complex32;
use std::sync::{Arc, OnceLock, RwLock};
use std::time::{Duration, Instant};

const LOG_FACTOR: f32 = 10.0 * core::f32::consts::LOG10_E;
const POWER_EPSILON: f32 = 1.0e-18;
const DB_FLOOR: f32 = -120.0;

/// Configuration for spectrogram FFT analysis.
#[derive(Debug, Clone, Copy)]
pub struct SpectrogramConfig {
    pub sample_rate: f32,
    /// FFT size (must be a power of two for radix-2 implementations).
    pub fft_size: usize,
    /// Hop size between successive frames.
    pub hop_size: usize,
    /// Optional Hann/Hamming/Blackman window selection.
    pub window: WindowKind,
    /// Maximum retained history columns.
    pub history_length: usize,
}

impl Default for SpectrogramConfig {
    fn default() -> Self {
        Self {
            sample_rate: DEFAULT_SAMPLE_RATE,
            fft_size: 8192,
            hop_size: 1024,
            window: WindowKind::Hann,
            history_length: 240,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WindowKind {
    Rectangular,
    Hann,
    Hamming,
    Blackman,
}

impl WindowKind {
    pub(crate) fn coefficients(self, len: usize) -> Vec<f32> {
        match self {
            WindowKind::Rectangular => vec![1.0; len],
            WindowKind::Hann => (0..len)
                .map(|n| {
                    let phase = (n as f32) * core::f32::consts::TAU / (len as f32);
                    0.5 * (1.0 - phase.cos())
                })
                .collect(),
            WindowKind::Hamming => (0..len)
                .map(|n| {
                    let phase = (n as f32) * core::f32::consts::TAU / (len as f32);
                    0.54 - 0.46 * phase.cos()
                })
                .collect(),
            WindowKind::Blackman => {
                let a0 = 0.42;
                let a1 = 0.5;
                let a2 = 0.08;
                (0..len)
                    .map(|n| {
                        let phase = (n as f32) * core::f32::consts::TAU / (len as f32);
                        a0 - a1 * phase.cos() + a2 * (2.0 * phase).cos()
                    })
                    .collect()
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct WindowKey {
    kind: WindowKind,
    len: usize,
}

struct WindowCache {
    entries: RwLock<FxHashMap<WindowKey, Arc<[f32]>>>,
}

impl WindowCache {
    fn global() -> &'static WindowCache {
        static INSTANCE: OnceLock<WindowCache> = OnceLock::new();
        INSTANCE.get_or_init(|| WindowCache {
            entries: RwLock::new(rustc_hash::FxHashMap::default()),
        })
    }

    fn get(&self, kind: WindowKind, len: usize) -> Arc<[f32]> {
        if len == 0 {
            return Arc::from([]);
        }

        let key = WindowKey { kind, len };
        if let Some(existing) = self.entries.read().unwrap().get(&key) {
            return Arc::clone(existing);
        }

        let mut write = self.entries.write().unwrap();
        Arc::clone(
            write
                .entry(key)
                .or_insert_with(|| Arc::from(kind.coefficients(len))),
        )
    }
}

/// One column of log-power magnitudes.
#[derive(Debug, Clone)]
pub struct SpectrogramColumn {
    pub timestamp: Instant,
    pub magnitudes_db: Arc<[f32]>,
}

/// Incremental update emitted by the spectrogram processor.
#[derive(Debug, Clone)]
pub struct SpectrogramUpdate {
    pub fft_size: usize,
    pub hop_size: usize,
    pub sample_rate: f32,
    pub history_length: usize,
    pub reset: bool,
    pub new_columns: Vec<SpectrogramColumn>,
}

#[derive(Debug)]
struct SpectrogramHistory {
    slots: Vec<Option<SpectrogramColumn>>,
    capacity: usize,
    head: usize,
    len: usize,
}

impl SpectrogramHistory {
    fn new(capacity: usize) -> Self {
        let mut slots = Vec::with_capacity(capacity);
        if capacity > 0 {
            slots.resize(capacity, None);
        }
        Self {
            slots,
            capacity,
            head: 0,
            len: 0,
        }
    }

    fn set_capacity(&mut self, capacity: usize, evicted: &mut Vec<SpectrogramColumn>) {
        if capacity == self.capacity {
            evicted.clear();
            return;
        }

        self.clear_into(evicted);
        self.capacity = capacity;
        self.head = 0;
        self.len = 0;
        self.ensure_slot_count();

        if capacity == 0 {
            return;
        }

        let retain_start = evicted.len().saturating_sub(capacity);
        if retain_start > 0 {
            evicted.drain(..retain_start);
        }
        for column in evicted.drain(..) {
            debug_assert!(self.push(column).is_none());
        }
    }

    fn clear_into(&mut self, out: &mut Vec<SpectrogramColumn>) {
        out.clear();
        out.reserve(self.len);
        while let Some(column) = self.pop_front_internal() {
            out.push(column);
        }
        self.head = 0;
    }

    fn push(&mut self, column: SpectrogramColumn) -> Option<SpectrogramColumn> {
        if self.capacity == 0 {
            return Some(column);
        }

        if self.slots.len() != self.capacity {
            self.ensure_slot_count();
        }

        let insert_idx = (self.head + self.len) % self.capacity;
        if self.len == self.capacity {
            let evicted = self.slots[insert_idx]
                .replace(column)
                .expect("occupied slot");
            self.head = (self.head + 1) % self.capacity;
            Some(evicted)
        } else {
            debug_assert!(self.slots[insert_idx].is_none());
            self.slots[insert_idx] = Some(column);
            self.len += 1;
            None
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn len(&self) -> usize {
        self.len
    }

    fn ensure_slot_count(&mut self) {
        if self.capacity == 0 {
            self.slots.clear();
        } else {
            self.slots.resize(self.capacity, None);
        }
    }

    fn pop_front_internal(&mut self) -> Option<SpectrogramColumn> {
        if self.len == 0 {
            return None;
        }

        let idx = self.head;
        let column = self.slots[idx].take().expect("occupied slot");
        if self.len == 1 {
            self.head = 0;
        } else {
            self.head = (self.head + 1) % self.capacity;
        }
        self.len -= 1;
        Some(column)
    }
}

#[derive(Debug, Clone)]
struct SampleBuffer {
    data: Vec<f32>,
    read: usize,
    len: usize,
}

impl SampleBuffer {
    fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            data: vec![0.0; capacity],
            read: 0,
            len: 0,
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn push(&mut self, sample: f32) {
        if self.len == self.data.len() {
            let new_cap = (self.data.len() * 2).max(1);
            self.grow_to(new_cap);
        }

        let write = (self.read + self.len) % self.data.len();
        self.data[write] = sample;
        self.len += 1;
    }

    fn reserve_additional(&mut self, additional: usize) {
        let required = self.len + additional;
        if required <= self.data.len() {
            return;
        }

        let mut new_capacity = self.data.len().max(1);
        while new_capacity < required {
            new_capacity *= 2;
        }
        self.grow_to(new_capacity);
    }

    fn consume(&mut self, count: usize) {
        assert!(count <= self.len);
        if count == 0 {
            return;
        }
        self.read = (self.read + count) % self.data.len();
        self.len -= count;
    }

    fn copy_front_into(&self, target: &mut [f32]) {
        assert!(target.len() <= self.len);
        let cap = self.data.len();
        for (offset, slot) in target.iter_mut().enumerate() {
            *slot = self.data[(self.read + offset) % cap];
        }
    }

    fn clear(&mut self) {
        self.read = 0;
        self.len = 0;
    }

    fn resize_capacity(&mut self, capacity: usize) {
        if capacity == 0 {
            self.data.clear();
            self.read = 0;
            self.len = 0;
            return;
        }

        if capacity == self.data.len() {
            return;
        }

        if capacity < self.len {
            self.consume(self.len - capacity);
        }
        self.grow_to(capacity);
    }

    fn grow_to(&mut self, new_capacity: usize) {
        let mut new_data = vec![0.0; new_capacity];
        let cap = self.data.len().max(1);
        for (i, slot) in new_data.iter_mut().enumerate().take(self.len) {
            *slot = self.data[(self.read + i) % cap];
        }
        self.data = new_data;
        self.read = 0;
    }
}

pub struct SpectrogramProcessor {
    config: SpectrogramConfig,
    planner: RealFftPlanner<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
    window: Arc<[f32]>,
    real_buffer: Vec<f32>,
    spectrum_buffer: Vec<Complex32>,
    scratch_buffer: Vec<Complex32>,
    magnitude_buffer: Vec<f32>,
    bin_normalization: Vec<f32>,
    pcm_buffer: SampleBuffer,
    buffer_start_index: u64,
    start_instant: Option<Instant>,
    accumulated_offset: Duration,
    history: SpectrogramHistory,
    magnitude_pool: Vec<Arc<[f32]>>,
    evicted_columns: Vec<SpectrogramColumn>,
    pending_reset: bool,
}

impl SpectrogramProcessor {
    pub fn new(config: SpectrogramConfig) -> Self {
        let fft_size = config.fft_size;
        let history_len = config.history_length;
        let bins = fft_size / 2 + 1;
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(fft_size);
        let window = WindowCache::global().get(config.window, fft_size);
        let real_buffer = vec![0.0; fft_size];
        let spectrum_buffer = fft.make_output_vec();
        let scratch_buffer = fft.make_scratch_vec();
        let magnitude_buffer = vec![0.0; bins];
        let bin_normalization = Self::compute_bin_normalization(window.as_ref());
        let pcm_buffer = SampleBuffer::with_capacity(fft_size.saturating_mul(2));
        let history = SpectrogramHistory::new(history_len);
        Self {
            config,
            planner,
            fft,
            window,
            real_buffer,
            spectrum_buffer,
            scratch_buffer,
            magnitude_buffer,
            bin_normalization,
            pcm_buffer,
            buffer_start_index: 0,
            start_instant: None,
            accumulated_offset: Duration::default(),
            history,
            magnitude_pool: Vec::new(),
            evicted_columns: Vec::new(),
            pending_reset: true,
        }
    }

    pub fn config(&self) -> SpectrogramConfig {
        self.config
    }

    fn rebuild_fft(&mut self) {
        let fft_size = self.config.fft_size;
        let bins = fft_size / 2 + 1;
        self.fft = self.planner.plan_fft_forward(fft_size);
        self.window = WindowCache::global().get(self.config.window, fft_size);
        self.real_buffer.resize(fft_size, 0.0);
        self.spectrum_buffer = self.fft.make_output_vec();
        self.scratch_buffer = self.fft.make_scratch_vec();
        self.magnitude_buffer.resize(bins, 0.0);
        self.bin_normalization = Self::compute_bin_normalization(self.window.as_ref());
        self.magnitude_pool.retain(|buffer| buffer.len() == bins);
        self.pcm_buffer
            .resize_capacity(fft_size.saturating_mul(2).max(1));
        let mut evicted = std::mem::take(&mut self.evicted_columns);
        self.history.clear_into(&mut evicted);
        evicted
            .drain(..)
            .for_each(|column| self.recycle_column(column));
        self.history
            .set_capacity(self.config.history_length, &mut evicted);
        evicted
            .drain(..)
            .for_each(|column| self.recycle_column(column));
        self.evicted_columns = evicted;
        self.pcm_buffer.clear();
        self.buffer_start_index = 0;
        self.start_instant = None;
        self.accumulated_offset = Duration::default();
        self.pending_reset = true;
    }

    fn ensure_fft_capacity(&mut self) {
        if self.real_buffer.len() != self.config.fft_size
            || self.spectrum_buffer.len() != self.config.fft_size / 2 + 1
        {
            self.rebuild_fft();
        }
    }

    fn process_ready_windows(&mut self) -> Vec<SpectrogramColumn> {
        let mut new_columns = Vec::new();
        let fft_size = self.config.fft_size;
        let hop = self.config.hop_size;
        if fft_size == 0 || hop == 0 {
            return new_columns;
        }

        let hop_duration = if self.config.sample_rate > 0.0 {
            duration_from_samples(hop as u64, self.config.sample_rate)
        } else {
            Duration::default()
        };
        let bins = fft_size / 2 + 1;

        let Some(start_instant) = self.start_instant else {
            return new_columns;
        };
        let center_offset = duration_from_samples((fft_size / 2) as u64, self.config.sample_rate);

        while self.pcm_buffer.len() >= fft_size {
            self.pcm_buffer
                .copy_front_into(&mut self.real_buffer[..fft_size]);
            Self::remove_dc(&mut self.real_buffer[..fft_size]);
            Self::apply_window(&mut self.real_buffer[..fft_size], self.window.as_ref());

            self.fft
                .process_with_scratch(
                    &mut self.real_buffer,
                    &mut self.spectrum_buffer,
                    &mut self.scratch_buffer,
                )
                .expect("real FFT forward transform");

            self.spectrum_buffer
                .iter()
                .zip(&self.bin_normalization)
                .zip(&mut self.magnitude_buffer)
                .for_each(|((complex, norm), target)| {
                    let power = (complex.norm_sqr() * *norm).max(POWER_EPSILON);
                    *target = (power.ln() * LOG_FACTOR).max(DB_FLOOR);
                });

            let timestamp = start_instant + self.accumulated_offset + center_offset;

            let mut magnitudes = self.acquire_magnitude_storage(bins);
            Arc::get_mut(&mut magnitudes)
                .expect("pooled magnitude storage should be unique")
                .copy_from_slice(&self.magnitude_buffer[..bins]);

            let history_column = SpectrogramColumn {
                timestamp,
                magnitudes_db: Arc::clone(&magnitudes),
            };
            if let Some(evicted) = self.history.push(history_column) {
                self.recycle_column(evicted);
            }

            new_columns.push(SpectrogramColumn {
                timestamp,
                magnitudes_db: magnitudes,
            });

            self.pcm_buffer.consume(hop);
            self.buffer_start_index += hop as u64;
            self.accumulated_offset += hop_duration;
        }

        new_columns
    }
}

impl AudioProcessor for SpectrogramProcessor {
    type Output = SpectrogramUpdate;

    fn process_block(&mut self, block: &AudioBlock<'_>) -> ProcessorUpdate<Self::Output> {
        if block.frame_count() == 0 || block.channels == 0 {
            return ProcessorUpdate::None;
        }

        if self.config.sample_rate <= 0.0 {
            self.config.sample_rate = block.sample_rate;
        } else if (self.config.sample_rate - block.sample_rate).abs() > f32::EPSILON {
            let duration_elapsed =
                duration_from_samples(self.buffer_start_index, self.config.sample_rate);
            let previous_start = self.start_instant;
            self.config.sample_rate = block.sample_rate;
            self.rebuild_fft();
            self.start_instant = previous_start;
            self.accumulated_offset = duration_elapsed;
        }

        if self.start_instant.is_none() {
            self.start_instant = Some(block.timestamp);
        }

        self.ensure_fft_capacity();

        self.pcm_buffer.reserve_additional(block.frame_count());

        self.mixdown_interleaved(block.samples, block.channels);

        let new_columns = self.process_ready_windows();
        if new_columns.is_empty() {
            ProcessorUpdate::None
        } else {
            let reset = std::mem::take(&mut self.pending_reset);
            ProcessorUpdate::Snapshot(SpectrogramUpdate {
                fft_size: self.config.fft_size,
                hop_size: self.config.hop_size,
                sample_rate: self.config.sample_rate,
                history_length: self.config.history_length,
                reset,
                new_columns,
            })
        }
    }

    fn reset(&mut self) {
        let mut evicted = std::mem::take(&mut self.evicted_columns);
        self.history.clear_into(&mut evicted);
        for column in evicted.drain(..) {
            self.recycle_column(column);
        }
        self.evicted_columns = evicted;
        let target_capacity = self.config.fft_size.saturating_mul(2).max(1);
        self.pcm_buffer.resize_capacity(target_capacity);
        self.pcm_buffer.clear();
        self.buffer_start_index = 0;
        self.start_instant = None;
        self.pending_reset = true;
    }
}

impl SpectrogramProcessor {
    #[inline]
    fn mixdown_interleaved(&mut self, samples: &[f32], channels: usize) {
        match channels {
            1 => Self::push_all(&mut self.pcm_buffer, samples),
            2 => self.mixdown_stereo(samples),
            _ => self.mixdown_generic(samples, channels),
        }
    }

    #[inline(always)]
    fn push_all(buffer: &mut SampleBuffer, samples: &[f32]) {
        for &sample in samples {
            buffer.push(sample);
        }
    }

    #[inline]
    fn mixdown_stereo(&mut self, samples: &[f32]) {
        for chunk in samples.chunks(2) {
            let value = if chunk.len() == 2 {
                0.5 * (chunk[0] + chunk[1])
            } else {
                0.5 * chunk[0]
            };
            self.pcm_buffer.push(value);
        }
    }

    #[inline]
    fn mixdown_generic(&mut self, samples: &[f32], channels: usize) {
        debug_assert!(channels > 2);
        let inv = 1.0 / channels as f32;
        for chunk in samples.chunks(channels) {
            let sum: f32 = chunk.iter().sum();
            self.pcm_buffer.push(sum * inv);
        }
    }

    #[inline(always)]
    fn apply_window(buffer: &mut [f32], window: &[f32]) {
        debug_assert_eq!(buffer.len(), window.len());
        for (sample, coeff) in buffer.iter_mut().zip(window.iter()) {
            *sample *= *coeff;
        }
    }

    #[inline]
    fn remove_dc(buffer: &mut [f32]) {
        if buffer.is_empty() {
            return;
        }

        let mean = buffer.iter().sum::<f32>() / buffer.len() as f32;
        if mean.abs() <= f32::EPSILON {
            return;
        }

        for sample in buffer.iter_mut() {
            *sample -= mean;
        }
    }

    fn compute_bin_normalization(window: &[f32]) -> Vec<f32> {
        let fft_size = window.len();
        let bins = fft_size / 2 + 1;
        if bins == 0 {
            return Vec::new();
        }

        let window_sum: f32 = window.iter().sum();
        let inv_sum = if window_sum.abs() > f32::EPSILON {
            1.0 / window_sum
        } else if fft_size > 0 {
            1.0 / fft_size as f32
        } else {
            0.0
        };

        let dc_scale = inv_sum * inv_sum;
        let interior_scale = (2.0 * inv_sum) * (2.0 * inv_sum);
        let mut norms = vec![interior_scale; bins];
        norms[0] = dc_scale;
        if bins > 1 {
            norms[bins - 1] = dc_scale;
        }
        norms
    }

    fn acquire_magnitude_storage(&mut self, bins: usize) -> Arc<[f32]> {
        if bins == 0 {
            return Arc::from([]);
        }

        if let Some(index) = self
            .magnitude_pool
            .iter()
            .rposition(|buffer| buffer.len() == bins)
        {
            self.magnitude_pool.swap_remove(index)
        } else {
            Arc::<[f32]>::from(vec![0.0f32; bins])
        }
    }

    fn recycle_column(&mut self, column: SpectrogramColumn) {
        if Arc::strong_count(&column.magnitudes_db) == 1
            && Arc::weak_count(&column.magnitudes_db) == 0
        {
            self.magnitude_pool.push(column.magnitudes_db);
        }
    }
}

impl Reconfigurable<SpectrogramConfig> for SpectrogramProcessor {
    fn update_config(&mut self, config: SpectrogramConfig) {
        self.config = config;
        self.rebuild_fft();
    }
}

fn duration_from_samples(sample_index: u64, sample_rate: f32) -> Duration {
    if sample_rate <= 0.0 {
        return Duration::default();
    }
    let seconds = sample_index as f64 / sample_rate as f64;
    Duration::from_secs_f64(seconds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::{AudioBlock, ProcessorUpdate};
    use std::time::Instant;

    fn make_block(samples: Vec<f32>, channels: usize, sample_rate: f32) -> AudioBlock<'static> {
        AudioBlock::new(
            Box::leak(samples.into_boxed_slice()),
            channels,
            sample_rate,
            Instant::now(),
        )
    }

    #[test]
    fn detects_sine_frequency_peak() {
        let config = SpectrogramConfig {
            fft_size: 1024,
            hop_size: 512,
            history_length: 8,
            sample_rate: DEFAULT_SAMPLE_RATE,
            window: WindowKind::Hann,
        };
        let mut processor = SpectrogramProcessor::new(config);

        let bin_hz = config.sample_rate / config.fft_size as f32;
        let target_bin = 200usize;
        let freq = target_bin as f32 * bin_hz;
        let frames = config.fft_size * 2;
        let mut samples = Vec::with_capacity(frames);
        for n in 0..frames {
            let t = n as f32 / config.sample_rate;
            samples.push((2.0 * core::f32::consts::PI * freq * t).sin());
        }

        let block_samples = samples.clone();
        let block = make_block(block_samples, 1, config.sample_rate);

        let result = processor.process_block(&block);
        let update = match result {
            ProcessorUpdate::Snapshot(update) => update,
            ProcessorUpdate::None => panic!("expected snapshot"),
        };

        assert!(!update.new_columns.is_empty());
        let last = update.new_columns.last().unwrap();
        let max_index = last
            .magnitudes_db
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        let peak_freq = max_index as f32 * bin_hz;
        assert!((peak_freq - freq).abs() < bin_hz * 1.5);

        assert_eq!(max_index, target_bin);

        let peak_db = last.magnitudes_db[max_index];
        assert!(
            peak_db > -0.5 && peak_db < 0.5,
            "expected ~0 dBFS peak, saw {peak_db}",
        );
    }

    #[test]
    fn history_respects_limit() {
        let config = SpectrogramConfig {
            history_length: 4,
            ..SpectrogramConfig::default()
        };
        let mut processor = SpectrogramProcessor::new(config);
        let frames = config.fft_size * 4;
        let samples = vec![0.0f32; frames];
        let block = make_block(samples, 1, config.sample_rate);
        let _ = processor.process_block(&block);
        assert!(processor.history.len() <= config.history_length);
    }
}
