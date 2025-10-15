//! Real-time waveform DSP implementation for the scrolling waveform visual.

use super::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::util::audio::DEFAULT_SAMPLE_RATE;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub const MIN_SCROLL_SPEED: f32 = 10.0;
pub const MAX_SCROLL_SPEED: f32 = 1000.0;
pub const DEFAULT_SCROLL_SPEED: f32 = 80.0;
pub const MIN_COLUMN_CAPACITY: usize = 512;
pub const MAX_COLUMN_CAPACITY: usize = 16_384;
pub const DEFAULT_COLUMN_CAPACITY: usize = 4_096;
const MIN_FREQUENCY_HZ: f32 = 20.0;
const MAX_FREQUENCY_HZ: f32 = 20_000.0;
const RAW_HISTORY_SECONDS: f32 = 0.5;
const RAW_HISTORY_MIN_FRAMES: usize = 2_048;
const RAW_HISTORY_MAX_FRAMES: usize = 65_536;

/// Strategy used to downsample the waveform when pixel budget is limited.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DownsampleStrategy {
    /// Keep min/max values per bucket (preserves peaks).
    #[default]
    MinMax,
    /// Simple averaging (cheaper but can hide transients).
    Average,
}

/// Configuration for the waveform preview.
#[derive(Debug, Clone, Copy)]
pub struct WaveformConfig {
    pub sample_rate: f32,
    /// How quickly the waveform scrolls, expressed as columns per second.
    pub scroll_speed: f32,
    pub downsample: DownsampleStrategy,
    pub max_columns: usize,
}

impl Default for WaveformConfig {
    fn default() -> Self {
        Self {
            sample_rate: DEFAULT_SAMPLE_RATE,
            scroll_speed: DEFAULT_SCROLL_SPEED,
            downsample: DownsampleStrategy::MinMax,
            max_columns: DEFAULT_COLUMN_CAPACITY,
        }
    }
}

/// Preview column describing in-flight samples that have not yet been committed.
#[derive(Debug, Clone, Default)]
pub struct WaveformPreview {
    pub progress: f32,
    pub min_values: Arc<Vec<f32>>,
    pub max_values: Arc<Vec<f32>>,
    pub frequency_normalized: f32,
}

/// Snapshot storing resampled waveform data.
#[derive(Debug, Clone, Default)]
pub struct WaveformSnapshot {
    pub channels: usize,
    pub columns: usize,
    pub min_values: Arc<Vec<f32>>,
    pub max_values: Arc<Vec<f32>>,
    pub frequency_normalized: Arc<Vec<f32>>,
    pub column_spacing_seconds: f32,
    pub scroll_position: f32,
    pub downsample: DownsampleStrategy,
    pub preview: WaveformPreview,
    pub raw: RawWaveform,
}

#[derive(Debug, Clone, Default)]
pub struct RawWaveform {
    pub channels: usize,
    pub frames: usize,
    pub sample_rate: f32,
    pub samples: Arc<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct WaveformProcessor {
    config: WaveformConfig,
    snapshot: WaveformSnapshot,
    channels: usize,
    column_period_seconds: f32,
    bucket_elapsed_seconds: f64,
    min_columns: Vec<f32>,
    max_columns: Vec<f32>,
    freq_columns: Vec<f32>,
    column_count: usize,
    write_column: usize,
    total_columns_written: u64,
    current_min: Vec<f32>,
    current_max: Vec<f32>,
    current_sum: Vec<f32>,
    current_sum_squares: Vec<f32>,
    current_samples: usize,
    current_crossing_count: usize,
    first_crossing_time: Option<f64>,
    last_crossing_time: Option<f64>,
    bucket_prev_sample: Option<f32>,
    bucket_prev_time: f64,
    prefilters: Vec<PrefilterState>,
    track_average: bool,
    last_snapshot_write_column: usize,
    last_snapshot_column_count: usize,
    preview_update_stride: usize,
    samples_since_preview_update: usize,
    raw_history_capacity: usize,
    raw_history: Vec<RawChannelHistory>,
}

#[derive(Debug, Clone)]
struct RawChannelHistory {
    samples: Vec<f32>,
    write_pos: usize,
    len: usize,
}

impl RawChannelHistory {
    fn new(capacity: usize) -> Self {
        Self {
            samples: vec![0.0; capacity],
            write_pos: 0,
            len: 0,
        }
    }

    fn capacity(&self) -> usize {
        self.samples.len()
    }

    fn ensure_capacity(&mut self, capacity: usize) {
        if self.capacity() == capacity {
            return;
        }

        let mut resized = RawChannelHistory::new(capacity);
        let keep = self.len.min(capacity);
        let start = self.start_index();
        for i in 0..keep {
            let idx = (start + i) % self.capacity();
            resized.samples[i] = self.samples[idx];
        }
        resized.write_pos = keep % capacity.max(1);
        resized.len = keep;
        *self = resized;
    }

    fn clear(&mut self) {
        self.samples.fill(0.0);
        self.write_pos = 0;
        self.len = 0;
    }

    fn push(&mut self, sample: f32) {
        if self.samples.is_empty() {
            return;
        }
        self.samples[self.write_pos] = sample;
        self.write_pos = (self.write_pos + 1) % self.capacity();
        if self.len < self.capacity() {
            self.len += 1;
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn start_index(&self) -> usize {
        if self.len >= self.capacity() {
            self.write_pos
        } else if self.capacity() == 0 {
            0
        } else {
            (self.write_pos + self.capacity() - self.len) % self.capacity()
        }
    }

    fn write_interleaved(&self, dest: &mut [f32], channels: usize, channel_index: usize) {
        if self.len == 0 {
            return;
        }

        let mut index = self.start_index();
        let cap = self.capacity().max(1);
        for frame in 0..self.len {
            let dest_index = frame * channels + channel_index;
            dest[dest_index] = self.samples[index];
            index = (index + 1) % cap;
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct PrefilterState {
    prev1: f32,
    prev2: f32,
    initialized: bool,
}

impl PrefilterState {
    fn filter(&mut self, sample: f32) -> f32 {
        if !self.initialized {
            self.prev1 = sample;
            self.prev2 = sample;
            self.initialized = true;
        }

        let filtered = self.prev2 * 0.25 + self.prev1 * 0.5 + sample * 0.25;
        self.prev2 = self.prev1;
        self.prev1 = sample;
        filtered
    }
}

impl WaveformProcessor {
    pub fn new(config: WaveformConfig) -> Self {
        let clamped = clamp_config(config);
        let mut processor = Self {
            config: clamped,
            snapshot: WaveformSnapshot::default(),
            channels: 2,
            column_period_seconds: calculate_column_period_seconds(&clamped),
            bucket_elapsed_seconds: 0.0,
            min_columns: Vec::new(),
            max_columns: Vec::new(),
            freq_columns: Vec::new(),
            column_count: 0,
            write_column: 0,
            total_columns_written: 0,
            current_min: Vec::new(),
            current_max: Vec::new(),
            current_sum: Vec::new(),
            current_sum_squares: Vec::new(),
            current_samples: 0,
            current_crossing_count: 0,
            first_crossing_time: None,
            last_crossing_time: None,
            bucket_prev_sample: None,
            bucket_prev_time: 0.0,
            prefilters: Vec::new(),
            track_average: clamped.downsample == DownsampleStrategy::Average,
            last_snapshot_write_column: 0,
            last_snapshot_column_count: 0,
            preview_update_stride: 1,
            samples_since_preview_update: 0,
            raw_history_capacity: 0,
            raw_history: Vec::new(),
        };
        processor.rebuild_for_config();
        processor
    }

    pub fn config(&self) -> WaveformConfig {
        self.config
    }

    pub fn snapshot(&self) -> &WaveformSnapshot {
        &self.snapshot
    }

    fn target_columns(&self) -> usize {
        self.config.max_columns.max(1)
    }

    fn nominal_frames_per_column(&self) -> f32 {
        nominal_frames_per_column(&self.config)
    }

    fn update_preview_stride(&mut self) {
        let nominal_frames = self.nominal_frames_per_column();
        let stride = (nominal_frames / 4.0).round() as usize;
        self.preview_update_stride = stride.max(1);
    }

    fn rebuild_for_config(&mut self) {
        self.track_average = self.config.downsample == DownsampleStrategy::Average;
        self.column_period_seconds = calculate_column_period_seconds(&self.config);
        self.update_preview_stride();
        self.resize_columns();
        self.bucket_elapsed_seconds = 0.0;
        self.reset_current_bucket();
        self.ensure_raw_history();
        self.populate_snapshot();
    }

    fn resize_columns(&mut self) {
        let channels = self.channels.max(1);
        let capacity = self.target_columns();
        let total = channels * capacity;

        self.min_columns = vec![0.0; total];
        self.max_columns = vec![0.0; total];
        self.freq_columns = vec![0.0; capacity];
        self.column_count = 0;
        self.write_column = 0;
        self.total_columns_written = 0;
        self.bucket_prev_sample = None;
        self.bucket_prev_time = 0.0;
        self.bucket_elapsed_seconds = 0.0;
        self.prefilters.resize(channels, PrefilterState::default());

        self.reset_current_bucket();
        self.ensure_raw_history();
    }

    fn ensure_raw_history(&mut self) {
        let channels = self.channels.max(1);
        let target_capacity = self.compute_raw_history_capacity();

        if self.raw_history_capacity != target_capacity || self.raw_history.len() != channels {
            self.raw_history_capacity = target_capacity;
            self.raw_history = (0..channels)
                .map(|_| RawChannelHistory::new(target_capacity))
                .collect();
        } else {
            for history in &mut self.raw_history {
                history.ensure_capacity(target_capacity);
            }
        }
    }

    fn compute_raw_history_capacity(&self) -> usize {
        let sample_rate = self.config.sample_rate.max(1.0);
        let frames_from_seconds = (sample_rate * RAW_HISTORY_SECONDS).ceil() as usize;
        let min_frames = RAW_HISTORY_MIN_FRAMES;
        let max_frames = RAW_HISTORY_MAX_FRAMES;
        frames_from_seconds.max(min_frames).min(max_frames).max(1)
    }

    fn adjust_capacity(&mut self, old_capacity: usize, new_capacity: usize) {
        if new_capacity == old_capacity {
            return;
        }

        let new_capacity = new_capacity.max(1);
        let channels = self.channels.max(1);
        let stored = self.column_count.min(old_capacity);
        let keep = stored.min(new_capacity);

        let mut new_min = vec![0.0; channels * new_capacity];
        let mut new_max = vec![0.0; channels * new_capacity];
        let mut new_freq = vec![0.0; new_capacity];

        if keep > 0 {
            let continuous = stored < old_capacity;
            let start_index = if continuous {
                stored.saturating_sub(keep)
            } else {
                (self.write_column + old_capacity - keep) % old_capacity
            };

            for (channel, (dest_min_chunk, dest_max_chunk)) in new_min
                .chunks_mut(new_capacity)
                .zip(new_max.chunks_mut(new_capacity))
                .enumerate()
                .take(channels)
            {
                let src_base = channel * old_capacity;
                for (offset, (dest_min, dest_max)) in dest_min_chunk
                    .iter_mut()
                    .zip(dest_max_chunk.iter_mut())
                    .take(keep)
                    .enumerate()
                {
                    let src_idx = if continuous {
                        start_index + offset
                    } else {
                        (start_index + offset) % old_capacity
                    };
                    *dest_min = self.min_columns[src_base + src_idx];
                    *dest_max = self.max_columns[src_base + src_idx];
                }
            }

            for (offset, dest) in new_freq.iter_mut().take(keep).enumerate() {
                let src_idx = if continuous {
                    start_index + offset
                } else {
                    (start_index + offset) % old_capacity
                };
                *dest = self.freq_columns[src_idx];
            }
        }

        self.min_columns = new_min;
        self.max_columns = new_max;
        self.freq_columns = new_freq;
        self.column_count = keep;
        self.write_column = if keep >= new_capacity { 0 } else { keep };
        self.bucket_elapsed_seconds = self
            .bucket_elapsed_seconds
            .min(self.column_period_seconds as f64);

        self.populate_snapshot();
    }

    fn reset_current_bucket(&mut self) {
        ensure_len(&mut self.current_min, self.channels, f32::MAX);
        ensure_len(&mut self.current_max, self.channels, f32::MIN);
        ensure_len(&mut self.current_sum, self.channels, 0.0);
        ensure_len(&mut self.current_sum_squares, self.channels, 0.0);
        self.current_samples = 0;
        self.current_crossing_count = 0;
        self.first_crossing_time = None;
        self.last_crossing_time = None;
        self.bucket_prev_sample = None;
        self.bucket_prev_time = 0.0;
        self.samples_since_preview_update = 0;
    }

    fn flush_current_bucket(&mut self) {
        if self.current_samples == 0 {
            return;
        }

        let capacity = self.target_columns();
        let channels = self.channels.max(1);
        let index = self.write_column;
        let frames = self.current_samples as f32;

        for channel in 0..channels {
            let mean = self.current_sum[channel] / frames;
            let offset = channel * capacity + index;

            if self.track_average {
                let mean_square = self.current_sum_squares[channel] / frames;
                let variance = (mean_square - mean * mean).max(0.0);
                let amplitude = variance.sqrt();
                self.min_columns[offset] = mean - amplitude;
                self.max_columns[offset] = mean + amplitude;
            } else {
                let min_val = if self.current_min[channel] == f32::MAX {
                    0.0
                } else {
                    self.current_min[channel]
                };
                let max_val = if self.current_max[channel] == f32::MIN {
                    0.0
                } else {
                    self.current_max[channel]
                };
                self.min_columns[offset] = min_val;
                self.max_columns[offset] = max_val;
            }
        }

        let frequency = if self.current_crossing_count >= 2 {
            if let (Some(first), Some(last)) = (self.first_crossing_time, self.last_crossing_time) {
                let span = last - first;
                if span > f64::EPSILON {
                    let cycles = (self.current_crossing_count - 1) as f64 / 2.0;
                    (cycles / span) as f32
                } else {
                    0.0
                }
            } else {
                0.0
            }
        } else {
            0.0
        };
        self.freq_columns[index] = normalize_frequency(frequency);

        self.write_column = (self.write_column + 1) % capacity;
        if self.column_count < capacity {
            self.column_count += 1;
        }
        self.total_columns_written = self.total_columns_written.saturating_add(1);

        self.reset_current_bucket();
    }

    fn prefilter_sample(&mut self, channel: usize, sample: f32) -> f32 {
        if self.prefilters.len() <= channel {
            self.prefilters
                .resize(channel + 1, PrefilterState::default());
        }

        self.prefilters[channel].filter(sample)
    }

    fn push_samples(&mut self, samples: &[f32], channels: usize) {
        if samples.is_empty() || channels == 0 {
            return;
        }

        let sample_rate = self.config.sample_rate.max(1.0);
        let sample_period = 1.0 / sample_rate;
        let sample_period_f64 = sample_period as f64;

        for frame in samples.chunks_exact(channels) {
            let current_time = self.bucket_elapsed_seconds;

            frame
                .iter()
                .take(channels)
                .enumerate()
                .for_each(|(channel, sample)| {
                    if let Some(history) = self.raw_history.get_mut(channel) {
                        history.push(*sample);
                    }
                });

            let raw_primary = frame[0];
            let filtered = self.prefilter_sample(0, raw_primary);
            self.current_min[0] = self.current_min[0].min(raw_primary);
            self.current_max[0] = self.current_max[0].max(raw_primary);
            self.current_sum[0] += raw_primary;
            self.current_sum_squares[0] += raw_primary * raw_primary;

            if let Some(prev) = self.bucket_prev_sample {
                let prev_time = self.bucket_prev_time;
                let prev_f64 = prev as f64;
                let filtered_f64 = filtered as f64;
                let denom = filtered_f64 - prev_f64;
                if denom.abs() > f64::EPSILON {
                    let t = (-prev_f64) / denom;
                    if (0.0..=1.0).contains(&t) {
                        let crossing_time = prev_time + t * sample_period_f64;
                        if self.first_crossing_time.is_none() {
                            self.first_crossing_time = Some(crossing_time);
                        }
                        self.last_crossing_time = Some(crossing_time);
                        self.current_crossing_count = self.current_crossing_count.saturating_add(1);
                    }
                }
            }
            self.bucket_prev_sample = Some(filtered);
            self.bucket_prev_time = current_time;

            frame
                .iter()
                .enumerate()
                .skip(1)
                .take(channels.saturating_sub(1))
                .for_each(|(channel, sample)| {
                    let raw = *sample;
                    self.prefilter_sample(channel, raw);
                    self.current_min[channel] = self.current_min[channel].min(raw);
                    self.current_max[channel] = self.current_max[channel].max(raw);
                    self.current_sum[channel] += raw;
                    self.current_sum_squares[channel] += raw * raw;
                });

            self.current_samples += 1;
            self.bucket_elapsed_seconds += sample_period_f64;
            self.samples_since_preview_update = self.samples_since_preview_update.saturating_add(1);

            if self.preview_update_stride == 0
                || self.samples_since_preview_update >= self.preview_update_stride
            {
                self.update_preview_snapshot();
                self.samples_since_preview_update = 0;
            }

            let flush_threshold =
                (self.column_period_seconds as f64 - sample_period_f64 * 0.5).max(0.0);
            if self.bucket_elapsed_seconds >= flush_threshold && self.current_samples > 0 {
                self.flush_current_bucket();
                self.bucket_elapsed_seconds =
                    (self.bucket_elapsed_seconds - self.column_period_seconds as f64).max(0.0);
                self.update_preview_snapshot();
            }
        }
    }

    fn populate_snapshot(&mut self) {
        let channels = self.channels.max(1);
        let capacity = self.target_columns();
        let columns = self.column_count.min(capacity);
        let spacing_seconds = self.column_period_seconds;
        self.populate_raw_snapshot(channels);

        let min_values = Arc::make_mut(&mut self.snapshot.min_values);
        let max_values = Arc::make_mut(&mut self.snapshot.max_values);
        let freq_values = Arc::make_mut(&mut self.snapshot.frequency_normalized);

        resize_vec(min_values, columns * channels);
        resize_vec(max_values, columns * channels);
        resize_vec(freq_values, columns);

        if columns == 0 {
            min_values.fill(0.0);
            max_values.fill(0.0);
            freq_values.fill(0.0);
            self.snapshot.channels = channels;
            self.snapshot.columns = 0;
            self.snapshot.column_spacing_seconds = spacing_seconds;
            self.snapshot.scroll_position = self.total_columns_written as f32;
            self.snapshot.downsample = self.config.downsample;
            self.last_snapshot_write_column = self.write_column;
            self.last_snapshot_column_count = self.column_count;
            return;
        }

        let start = if self.column_count < capacity {
            0
        } else {
            self.write_column
        };

        for (channel, (dest_min_chunk, dest_max_chunk)) in min_values
            .chunks_mut(columns)
            .zip(max_values.chunks_mut(columns))
            .enumerate()
            .take(channels)
        {
            let src_base = channel * capacity;
            for (offset, (dest_min, dest_max)) in dest_min_chunk
                .iter_mut()
                .zip(dest_max_chunk.iter_mut())
                .take(columns)
                .enumerate()
            {
                let src = (start + offset) % capacity;
                *dest_min = self.min_columns[src_base + src];
                *dest_max = self.max_columns[src_base + src];
            }
        }

        for (offset, dest) in freq_values.iter_mut().take(columns).enumerate() {
            let src = (start + offset) % capacity;
            *dest = self.freq_columns[src];
        }

        self.snapshot.channels = channels;
        self.snapshot.columns = columns;
        self.snapshot.column_spacing_seconds = spacing_seconds;
        self.snapshot.scroll_position = self.total_columns_written as f32;
        self.snapshot.downsample = self.config.downsample;
        self.last_snapshot_write_column = self.write_column;
        self.last_snapshot_column_count = self.column_count;
    }

    fn populate_raw_snapshot(&mut self, channels: usize) {
        let raw_frames = self
            .raw_history
            .first()
            .map(|history| history.len())
            .unwrap_or(0);
        let raw_frame_count = raw_frames.min(self.raw_history_capacity);
        let samples = Arc::make_mut(&mut self.snapshot.raw.samples);

        if raw_frame_count == 0 {
            samples.clear();
        } else {
            let required = raw_frame_count * channels;
            if samples.len() < required {
                samples.resize(required, 0.0);
            } else {
                samples[..required].fill(0.0);
                samples.truncate(required);
            }

            for (channel_index, history) in self.raw_history.iter().enumerate() {
                if channel_index >= channels {
                    break;
                }
                history.write_interleaved(samples, channels, channel_index);
            }
        }

        self.snapshot.raw.channels = channels;
        self.snapshot.raw.frames = raw_frame_count;
        self.snapshot.raw.sample_rate = self.config.sample_rate;
    }

    fn update_preview_snapshot(&mut self) {
        let channels = self.channels.max(1);
        if self.current_samples == 0 {
            let preview_min = Arc::make_mut(&mut self.snapshot.preview.min_values);
            let preview_max = Arc::make_mut(&mut self.snapshot.preview.max_values);
            self.snapshot.preview.progress = if self.column_period_seconds > 0.0 {
                (self.bucket_elapsed_seconds as f32 / self.column_period_seconds).clamp(0.0, 1.0)
            } else {
                0.0
            };
            preview_min.clear();
            preview_max.clear();
            self.snapshot.preview.frequency_normalized = 0.0;
            return;
        }

        let frames = self.current_samples as f32;
        let preview_min = Arc::make_mut(&mut self.snapshot.preview.min_values);
        let preview_max = Arc::make_mut(&mut self.snapshot.preview.max_values);
        resize_vec(preview_min, channels);
        resize_vec(preview_max, channels);

        preview_min
            .iter_mut()
            .zip(preview_max.iter_mut())
            .enumerate()
            .take(channels)
            .for_each(|(channel, (min_slot, max_slot))| {
                let mean = self.current_sum[channel] / frames;
                let (min_value, max_value) = if self.track_average {
                    let mean_square = self.current_sum_squares[channel] / frames;
                    let variance = (mean_square - mean * mean).max(0.0);
                    let amplitude = variance.sqrt();
                    (mean - amplitude, mean + amplitude)
                } else {
                    let min_val = if self.current_min[channel] == f32::MAX {
                        0.0
                    } else {
                        self.current_min[channel]
                    };
                    let max_val = if self.current_max[channel] == f32::MIN {
                        0.0
                    } else {
                        self.current_max[channel]
                    };
                    (min_val, max_val)
                };

                *min_slot = min_value;
                *max_slot = max_value;
            });

        let frequency = if self.current_crossing_count >= 2 {
            if let (Some(first), Some(last)) = (self.first_crossing_time, self.last_crossing_time) {
                let span = last - first;
                if span > f64::EPSILON {
                    let cycles = (self.current_crossing_count - 1) as f64 / 2.0;
                    (cycles / span) as f32
                } else {
                    0.0
                }
            } else {
                0.0
            }
        } else {
            0.0
        };
        self.snapshot.preview.frequency_normalized = normalize_frequency(frequency);
        self.snapshot.preview.progress = if self.column_period_seconds > 0.0 {
            (self.bucket_elapsed_seconds as f32 / self.column_period_seconds).clamp(0.0, 1.0)
        } else {
            0.0
        };
    }
}

impl AudioProcessor for WaveformProcessor {
    type Output = WaveformSnapshot;

    fn process_block(&mut self, block: &AudioBlock<'_>) -> ProcessorUpdate<Self::Output> {
        if block.frame_count() == 0 {
            return ProcessorUpdate::None;
        }

        let channels = block.channels.max(1);
        if channels != self.channels {
            self.channels = channels;
            self.resize_columns();
        }

        let sample_rate = block.sample_rate.max(1.0);
        if (self.config.sample_rate - sample_rate).abs() > f32::EPSILON {
            self.config.sample_rate = sample_rate;
            self.rebuild_for_config();
        }

        self.push_samples(block.samples, channels);

        let raw_frames_available = self
            .raw_history
            .first()
            .map(|history| history.len())
            .unwrap_or(0);

        if self.write_column != self.last_snapshot_write_column
            || self.column_count != self.last_snapshot_column_count
            || raw_frames_available != self.snapshot.raw.frames
        {
            self.populate_snapshot();
        }

        self.update_preview_snapshot();

        let progress = if self.column_period_seconds > 0.0 {
            (self.bucket_elapsed_seconds as f32 / self.column_period_seconds).clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.snapshot.scroll_position = self.total_columns_written as f32 + progress;

        ProcessorUpdate::Snapshot(self.snapshot.clone())
    }

    fn reset(&mut self) {
        self.snapshot = WaveformSnapshot::default();
        self.snapshot.downsample = self.config.downsample;
        self.column_count = 0;
        self.write_column = 0;
        self.total_columns_written = 0;
        self.min_columns.clear();
        self.max_columns.clear();
        self.freq_columns.clear();
        self.current_min.clear();
        self.current_max.clear();
        self.current_sum.clear();
        self.current_sum_squares.clear();
        self.current_samples = 0;
        self.current_crossing_count = 0;
        self.first_crossing_time = None;
        self.last_crossing_time = None;
        self.bucket_prev_sample = None;
        self.bucket_prev_time = 0.0;
        self.bucket_elapsed_seconds = 0.0;
        self.prefilters.clear();
        self.last_snapshot_write_column = 0;
        self.last_snapshot_column_count = 0;
        self.preview_update_stride = 1;
        self.samples_since_preview_update = 0;
        for history in &mut self.raw_history {
            history.clear();
        }
        self.raw_history_capacity = self.compute_raw_history_capacity();
        self.ensure_raw_history();

        self.rebuild_for_config();
    }
}

impl Reconfigurable<WaveformConfig> for WaveformProcessor {
    fn update_config(&mut self, config: WaveformConfig) {
        let clamped = clamp_config(config);
        let old_capacity = self.config.max_columns;
        let max_changed = old_capacity != clamped.max_columns;
        let requires_rebuild = (self.config.sample_rate - clamped.sample_rate).abs() > f32::EPSILON
            || (self.config.scroll_speed - clamped.scroll_speed).abs() > f32::EPSILON
            || self.config.downsample != clamped.downsample;

        self.config = clamped;
        if requires_rebuild {
            self.rebuild_for_config();
        } else if max_changed {
            self.adjust_capacity(old_capacity, self.config.max_columns);
        }
    }
}

fn clamp_config(mut config: WaveformConfig) -> WaveformConfig {
    config.sample_rate = config.sample_rate.max(1.0);
    config.scroll_speed = config
        .scroll_speed
        .clamp(MIN_SCROLL_SPEED, MAX_SCROLL_SPEED);
    config.max_columns = config
        .max_columns
        .clamp(MIN_COLUMN_CAPACITY, MAX_COLUMN_CAPACITY);
    config
}

fn calculate_column_period_seconds(config: &WaveformConfig) -> f32 {
    let speed = config.scroll_speed.max(MIN_SCROLL_SPEED);
    1.0 / speed
}

fn nominal_frames_per_column(config: &WaveformConfig) -> f32 {
    let sample_rate = config.sample_rate.max(1.0);
    let speed = config.scroll_speed.max(MIN_SCROLL_SPEED);
    sample_rate / speed
}

fn resize_vec(vec: &mut Vec<f32>, len: usize) {
    if vec.len() > len {
        vec.truncate(len);
    } else if vec.len() < len {
        vec.resize(len, 0.0);
    }
}

fn ensure_len(vec: &mut Vec<f32>, len: usize, fill: f32) {
    if vec.len() != len {
        vec.resize(len, fill);
    } else {
        vec.fill(fill);
    }
}

fn normalize_frequency(value: f32) -> f32 {
    if value <= MIN_FREQUENCY_HZ {
        return 0.0;
    }

    let clamped = value.clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ);
    let min_log = MIN_FREQUENCY_HZ.log10();
    let max_log = MAX_FREQUENCY_HZ.log10();
    let normalized = (clamped.log10() - min_log) / (max_log - min_log);
    normalized.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::{AudioBlock, ProcessorUpdate};
    use std::f32::consts::PI;
    use std::time::Instant;

    fn make_block<'a>(samples: &'a [f32], channels: usize, sample_rate: f32) -> AudioBlock<'a> {
        AudioBlock::new(samples, channels, sample_rate, Instant::now())
    }

    #[test]
    fn downsampling_produces_min_max_pairs() {
        let mut processor = WaveformProcessor::new(WaveformConfig {
            sample_rate: DEFAULT_SAMPLE_RATE,
            scroll_speed: 120.0,
            downsample: DownsampleStrategy::MinMax,
            max_columns: DEFAULT_COLUMN_CAPACITY,
        });

        let channels = 1;
        let frames_per_column = nominal_frames_per_column(&processor.config).ceil() as usize;
        let mut samples = Vec::new();
        for i in 0..frames_per_column {
            let value = if i % 2 == 0 { 0.5 } else { -0.25 };
            samples.push(value);
        }

        let block = make_block(&samples, channels, processor.config.sample_rate);
        let snapshot = match processor.process_block(&block) {
            ProcessorUpdate::Snapshot(snapshot) => snapshot,
            ProcessorUpdate::None => panic!("expected snapshot"),
        };

        assert_eq!(snapshot.columns, 1);
        assert_eq!(snapshot.min_values.len(), 1);
        assert_eq!(snapshot.max_values.len(), 1);

        let min_value = snapshot.min_values[0];
        let max_value = snapshot.max_values[0];
        assert!(max_value > 0.49, "expected to preserve positive peaks");
        assert!(min_value < -0.24, "expected to preserve negative troughs");
        assert!((max_value - 0.5).abs() < 1e-3);
        assert!((min_value + 0.25).abs() < 1e-3);
        assert_eq!(snapshot.raw.channels, channels);
        assert!(snapshot.raw.frames > 0);
    }

    #[test]
    fn average_downsampling_produces_envelope() {
        let mut processor = WaveformProcessor::new(WaveformConfig {
            sample_rate: DEFAULT_SAMPLE_RATE,
            scroll_speed: MIN_SCROLL_SPEED,
            downsample: DownsampleStrategy::Average,
            max_columns: DEFAULT_COLUMN_CAPACITY,
        });

        let channels = 1;
        let frames_per_column = nominal_frames_per_column(&processor.config).ceil() as usize;
        let mut samples = Vec::with_capacity(frames_per_column);
        for i in 0..frames_per_column {
            let value = if i % 2 == 0 { 0.6 } else { -0.2 };
            samples.push(value);
        }

        let block = make_block(&samples, channels, processor.config.sample_rate);
        let snapshot = match processor.process_block(&block) {
            ProcessorUpdate::Snapshot(snapshot) => snapshot,
            ProcessorUpdate::None => panic!("expected snapshot"),
        };

        assert_eq!(snapshot.columns, 1);
        assert_eq!(snapshot.min_values.len(), 1);
        assert_eq!(snapshot.max_values.len(), 1);

        let max_value = snapshot.max_values[0];
        let min_value = snapshot.min_values[0];
        assert!((max_value - 0.6).abs() < 1e-3);
        assert!((min_value + 0.2).abs() < 1e-3);
        assert_eq!(snapshot.raw.channels, channels);
        assert!(snapshot.raw.frames > 0);
    }

    #[test]
    fn estimates_frequency_for_sine_wave() {
        let frequency_hz = 440.0;
        let mut processor = WaveformProcessor::new(WaveformConfig {
            sample_rate: 48_000.0,
            scroll_speed: 200.0,
            downsample: DownsampleStrategy::MinMax,
            max_columns: DEFAULT_COLUMN_CAPACITY,
        });

        let channels = 1;
        let frames_per_column = nominal_frames_per_column(&processor.config).ceil() as usize;
        let total_frames = frames_per_column * 8;
        let mut samples = Vec::with_capacity(total_frames);
        for n in 0..total_frames {
            let t = n as f32 / processor.config.sample_rate;
            samples.push((2.0 * PI * frequency_hz * t).sin());
        }

        let block = make_block(&samples, channels, processor.config.sample_rate);
        let snapshot = match processor.process_block(&block) {
            ProcessorUpdate::Snapshot(snapshot) => snapshot,
            ProcessorUpdate::None => panic!("expected snapshot"),
        };

        let last_frequency = snapshot
            .frequency_normalized
            .last()
            .copied()
            .unwrap_or_default();
        assert!(last_frequency > 0.44 && last_frequency < 0.46);
        assert_eq!(snapshot.raw.channels, channels);
        assert!(snapshot.raw.frames > 0);
    }

    #[test]
    fn raw_history_preserves_recent_samples() {
        let mut processor = WaveformProcessor::new(WaveformConfig {
            sample_rate: 48_000.0,
            scroll_speed: MIN_SCROLL_SPEED,
            downsample: DownsampleStrategy::MinMax,
            max_columns: DEFAULT_COLUMN_CAPACITY,
        });

        let channels = 1;
        let total_frames = 4_096;
        let mut samples = Vec::with_capacity(total_frames);
        for n in 0..total_frames {
            samples.push(n as f32 / total_frames as f32);
        }

        let block = make_block(&samples, channels, processor.config.sample_rate);
        let snapshot = match processor.process_block(&block) {
            ProcessorUpdate::Snapshot(snapshot) => snapshot,
            ProcessorUpdate::None => panic!("expected snapshot"),
        };

        assert_eq!(snapshot.raw.channels, channels);
        assert!(snapshot.raw.frames > 0);
        let expected_tail = samples.last().copied().unwrap_or_default();
        let actual_tail = snapshot
            .raw
            .samples
            .chunks(channels)
            .last()
            .and_then(|chunk| chunk.first())
            .copied()
            .unwrap_or_default();
        assert!((actual_tail - expected_tail).abs() < 1e-6);
    }
}
