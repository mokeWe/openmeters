//! Waveform DSP implementation.

use super::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::util::audio::DEFAULT_SAMPLE_RATE;
use serde::{Deserialize, Serialize};

pub const MIN_SCROLL_SPEED: f32 = 10.0;
pub const MAX_SCROLL_SPEED: f32 = 1000.0;
const DEFAULT_SCROLL_SPEED: f32 = 80.0;
pub const MIN_COLUMN_CAPACITY: usize = 512;
pub const MAX_COLUMN_CAPACITY: usize = 16_384;
pub const DEFAULT_COLUMN_CAPACITY: usize = 4_096;
const MIN_FREQUENCY_HZ: f32 = 20.0;
const MAX_FREQUENCY_HZ: f32 = 20_000.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DownsampleStrategy {
    #[default]
    MinMax,
    Average,
}

#[derive(Debug, Clone, Copy)]
pub struct WaveformConfig {
    pub sample_rate: f32,
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

#[derive(Debug, Clone, Default)]
pub struct WaveformPreview {
    pub progress: f32,
    pub min_values: Vec<f32>,
    pub max_values: Vec<f32>,
    pub frequency_normalized: Vec<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct WaveformSnapshot {
    pub channels: usize,
    pub columns: usize,
    pub min_values: Vec<f32>,
    pub max_values: Vec<f32>,
    pub frequency_normalized: Vec<f32>,
    pub column_spacing_seconds: f32,
    pub scroll_position: f32,
    pub downsample: DownsampleStrategy,
    pub preview: WaveformPreview,
}

#[derive(Debug, Clone, Copy, Default)]
struct ChannelAccumulator {
    min: f32,
    max: f32,
    sum: f32,
    sum_squares: f32,
}

impl ChannelAccumulator {
    fn reset(&mut self) {
        *self = Self {
            min: f32::MAX,
            max: f32::MIN,
            sum: 0.0,
            sum_squares: 0.0,
        };
    }

    fn accumulate(&mut self, sample: f32) {
        self.min = self.min.min(sample);
        self.max = self.max.max(sample);
        self.sum += sample;
        self.sum_squares += sample * sample;
    }

    fn compute_range(&self, sample_count: usize, use_average: bool) -> (f32, f32) {
        if sample_count == 0 {
            return (0.0, 0.0);
        }

        let count = sample_count as f32;
        let mean = self.sum / count;

        if use_average {
            let variance = ((self.sum_squares / count) - mean * mean).max(0.0);
            let amplitude = variance.sqrt();
            (mean - amplitude, mean + amplitude)
        } else {
            let min = if self.min == f32::MAX { 0.0 } else { self.min };
            let max = if self.max == f32::MIN { 0.0 } else { self.max };
            (min, max)
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct Prefilter {
    prev1: f32,
    prev2: f32,
    initialized: bool,
}

impl Prefilter {
    fn process(&mut self, sample: f32) -> f32 {
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
    accumulators: Vec<ChannelAccumulator>,
    sample_count: usize,
    crossing_counts: Vec<usize>,
    first_crossing_times: Vec<Option<f64>>,
    last_crossing_times: Vec<Option<f64>>,
    prev_samples: Vec<Option<f32>>,
    prev_times: Vec<f64>,
    prefilters: Vec<Prefilter>,
    use_average: bool,
    snapshot_dirty: bool,
}

impl WaveformProcessor {
    pub fn new(config: WaveformConfig) -> Self {
        let clamped = clamp_config(config);
        let mut processor = Self {
            config: clamped,
            snapshot: WaveformSnapshot::default(),
            channels: 2,
            column_period_seconds: column_period_seconds(&clamped),
            bucket_elapsed_seconds: 0.0,
            min_columns: Vec::new(),
            max_columns: Vec::new(),
            freq_columns: Vec::new(),
            column_count: 0,
            write_column: 0,
            total_columns_written: 0,
            accumulators: Vec::new(),
            sample_count: 0,
            crossing_counts: Vec::new(),
            first_crossing_times: Vec::new(),
            last_crossing_times: Vec::new(),
            prev_samples: Vec::new(),
            prev_times: Vec::new(),
            prefilters: Vec::new(),
            use_average: clamped.downsample == DownsampleStrategy::Average,
            snapshot_dirty: false,
        };
        processor.rebuild_for_config();
        processor
    }

    pub fn config(&self) -> WaveformConfig {
        self.config
    }

    fn rebuild_for_config(&mut self) {
        self.use_average = self.config.downsample == DownsampleStrategy::Average;
        self.column_period_seconds = column_period_seconds(&self.config);
        self.resize_columns();
        self.bucket_elapsed_seconds = 0.0;
        self.reset_bucket();
    }

    fn resize_columns(&mut self) {
        let channels = self.channels.max(1);
        let capacity = self.config.max_columns.max(1);

        self.min_columns = vec![0.0; channels * capacity];
        self.max_columns = vec![0.0; channels * capacity];
        self.freq_columns = vec![0.0; channels * capacity];
        self.column_count = 0;
        self.write_column = 0;
        self.total_columns_written = 0;
        self.prefilters.resize(channels, Prefilter::default());
        self.reset_bucket();
    }

    fn reset_bucket(&mut self) {
        let channels = self.channels;

        self.accumulators
            .resize(channels, ChannelAccumulator::default());
        for acc in &mut self.accumulators {
            acc.reset();
        }
        self.sample_count = 0;
        self.crossing_counts.resize(channels, 0);
        for count in &mut self.crossing_counts {
            *count = 0;
        }
        self.first_crossing_times.resize(channels, None);
        self.last_crossing_times.resize(channels, None);
        self.prev_samples.resize(channels, None);
        self.prev_times.resize(channels, 0.0);
    }

    fn adjust_capacity(&mut self, old_capacity: usize, new_capacity: usize) {
        if new_capacity == old_capacity {
            return;
        }

        let channels = self.channels.max(1);
        let new_cap = new_capacity.max(1);
        let stored = self.column_count.min(old_capacity);
        let keep = stored.min(new_cap);

        let mut new_min = vec![0.0; channels * new_cap];
        let mut new_max = vec![0.0; channels * new_cap];
        let mut new_freq = vec![0.0; channels * new_cap];

        if keep > 0 {
            let ring_wrap = stored >= old_capacity;
            let start = if ring_wrap {
                (self.write_column + old_capacity - keep) % old_capacity
            } else {
                stored.saturating_sub(keep)
            };

            for (ch, (min_dst, max_dst)) in new_min
                .chunks_mut(new_cap)
                .zip(new_max.chunks_mut(new_cap))
                .enumerate()
                .take(channels)
            {
                let src_base = ch * old_capacity;
                for (i, (min, max)) in min_dst.iter_mut().zip(max_dst).take(keep).enumerate() {
                    let src = if ring_wrap {
                        (start + i) % old_capacity
                    } else {
                        start + i
                    };
                    *min = self.min_columns[src_base + src];
                    *max = self.max_columns[src_base + src];
                }
            }

            for (ch, freq_dst) in new_freq.chunks_mut(new_cap).enumerate().take(channels) {
                let src_base = ch * old_capacity;
                for (i, dst) in freq_dst.iter_mut().take(keep).enumerate() {
                    let src = if ring_wrap {
                        (start + i) % old_capacity
                    } else {
                        start + i
                    };
                    *dst = self.freq_columns[src_base + src];
                }
            }
        }

        self.min_columns = new_min;
        self.max_columns = new_max;
        self.freq_columns = new_freq;
        self.column_count = keep;
        self.write_column = if keep >= new_cap { 0 } else { keep };
        self.bucket_elapsed_seconds = self
            .bucket_elapsed_seconds
            .min(self.column_period_seconds as f64);
        self.snapshot_dirty = true;
    }

    fn flush_bucket(&mut self) {
        if self.sample_count == 0 {
            return;
        }

        let capacity = self.config.max_columns.max(1);
        let channels = self.channels.max(1);
        let idx = self.write_column;

        for (ch, acc) in self.accumulators.iter().take(channels).enumerate() {
            let (min, max) = acc.compute_range(self.sample_count, self.use_average);
            let offset = ch * capacity + idx;
            self.min_columns[offset] = min;
            self.max_columns[offset] = max;
        }

        let bucket_duration = self.bucket_elapsed_seconds;

        for ch in 0..channels {
            let offset = ch * capacity + idx;
            let previous = if self.column_count == 0 {
                None
            } else {
                let prev_index = (self.write_column + capacity - 1) % capacity;
                Some(self.freq_columns[ch * capacity + prev_index])
            };
            self.freq_columns[offset] = compute_frequency(self, ch, bucket_duration, previous);
        }

        self.write_column = (self.write_column + 1) % capacity;
        if self.column_count < capacity {
            self.column_count += 1;
        }
        self.total_columns_written = self.total_columns_written.saturating_add(1);
        self.snapshot_dirty = true;
        self.reset_bucket();
    }

    fn process_samples(&mut self, samples: &[f32], channels: usize) {
        if samples.is_empty() || channels == 0 {
            return;
        }

        let sample_period = (1.0 / self.config.sample_rate.max(1.0)) as f64;

        for frame in samples.chunks_exact(channels) {
            let current_time = self.bucket_elapsed_seconds;
            for (ch, &sample) in frame.iter().enumerate().take(channels) {
                let filtered = {
                    let prefilter = &mut self.prefilters[ch];
                    prefilter.process(sample)
                };

                if let Some(acc) = self.accumulators.get_mut(ch) {
                    acc.accumulate(sample);
                }

                let prev_sample = self.prev_samples[ch];
                if let Some(prev) = prev_sample {
                    let prev_time = self.prev_times[ch];
                    let denom = (filtered - prev) as f64;
                    if denom.abs() > f64::EPSILON {
                        let t = (-prev as f64) / denom;
                        if (0.0..=1.0).contains(&t) {
                            let crossing_time = prev_time + t * sample_period;
                            if self.first_crossing_times[ch].is_none() {
                                self.first_crossing_times[ch] = Some(crossing_time);
                            }
                            self.last_crossing_times[ch] = Some(crossing_time);
                            self.crossing_counts[ch] = self.crossing_counts[ch].saturating_add(1);
                        }
                    }
                }

                self.prev_samples[ch] = Some(filtered);
                self.prev_times[ch] = current_time;
            }

            self.sample_count += 1;
            self.bucket_elapsed_seconds += sample_period;

            // Flush bucket when period is complete
            if self.bucket_elapsed_seconds
                >= (self.column_period_seconds as f64 - sample_period * 0.5).max(0.0)
                && self.sample_count > 0
            {
                self.flush_bucket();
                self.bucket_elapsed_seconds =
                    (self.bucket_elapsed_seconds - self.column_period_seconds as f64).max(0.0);
            }
        }
    }

    fn update_snapshot(&mut self) {
        let channels = self.channels.max(1);
        let capacity = self.config.max_columns.max(1);
        let columns = self.column_count.min(capacity);
        let spacing = self.column_period_seconds;

        self.snapshot.min_values.resize(columns * channels, 0.0);
        self.snapshot.max_values.resize(columns * channels, 0.0);
        self.snapshot
            .frequency_normalized
            .resize(columns * channels, 0.0);

        if columns == 0 {
            self.snapshot.channels = channels;
            self.snapshot.columns = 0;
            self.snapshot.column_spacing_seconds = spacing;
            self.snapshot.downsample = self.config.downsample;
            self.snapshot.frequency_normalized.clear();
            return;
        }

        let start = if self.column_count < capacity {
            0
        } else {
            self.write_column
        };

        for (ch, (min_dst, max_dst)) in self
            .snapshot
            .min_values
            .chunks_mut(columns)
            .zip(self.snapshot.max_values.chunks_mut(columns))
            .enumerate()
            .take(channels)
        {
            let src_base = ch * capacity;
            for (i, (min, max)) in min_dst.iter_mut().zip(max_dst).take(columns).enumerate() {
                let src = (start + i) % capacity;
                *min = self.min_columns[src_base + src];
                *max = self.max_columns[src_base + src];
            }
        }

        for (ch, freq_dst) in self
            .snapshot
            .frequency_normalized
            .chunks_mut(columns)
            .enumerate()
            .take(channels)
        {
            let src_base = ch * capacity;
            for (i, dst) in freq_dst.iter_mut().take(columns).enumerate() {
                let src = (start + i) % capacity;
                *dst = self.freq_columns[src_base + src];
            }
        }

        self.snapshot.channels = channels;
        self.snapshot.columns = columns;
        self.snapshot.column_spacing_seconds = spacing;
        self.snapshot.downsample = self.config.downsample;
        self.snapshot_dirty = false;
    }

    fn update_preview(&mut self) {
        let channels = self.channels.max(1);
        let progress = if self.column_period_seconds > 0.0 {
            (self.bucket_elapsed_seconds as f32 / self.column_period_seconds).clamp(0.0, 1.0)
        } else {
            0.0
        };

        self.snapshot.preview.progress = progress;

        if self.sample_count == 0 {
            self.snapshot.preview.min_values.clear();
            self.snapshot.preview.max_values.clear();
            self.snapshot.preview.frequency_normalized.clear();
            return;
        }

        self.snapshot.preview.min_values.resize(channels, 0.0);
        self.snapshot.preview.max_values.resize(channels, 0.0);
        self.snapshot
            .preview
            .frequency_normalized
            .resize(channels, 0.0);

        let bucket_duration = self.bucket_elapsed_seconds;
        let capacity = self.config.max_columns.max(1);

        for (ch, acc) in self.accumulators.iter().take(channels).enumerate() {
            let (min, max) = acc.compute_range(self.sample_count, self.use_average);
            self.snapshot.preview.min_values[ch] = min;
            self.snapshot.preview.max_values[ch] = max;
            let previous = if self.column_count == 0 {
                None
            } else {
                let prev_index = (self.write_column + capacity - 1) % capacity;
                Some(self.freq_columns[ch * capacity + prev_index])
            };
            self.snapshot.preview.frequency_normalized[ch] =
                compute_frequency(self, ch, bucket_duration, previous);
        }
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

        self.process_samples(block.samples, channels);

        if self.snapshot_dirty {
            self.update_snapshot();
        }

        self.update_preview();

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
        self.crossing_counts.clear();
        self.first_crossing_times.clear();
        self.last_crossing_times.clear();
        self.prev_samples.clear();
        self.prev_times.clear();
        self.prefilters.clear();
        self.bucket_elapsed_seconds = 0.0;
        self.snapshot_dirty = false;
        self.rebuild_for_config();
    }
}

impl Reconfigurable<WaveformConfig> for WaveformProcessor {
    fn update_config(&mut self, config: WaveformConfig) {
        let clamped = clamp_config(config);
        let old_capacity = self.config.max_columns;
        let max_changed = old_capacity != clamped.max_columns;
        let needs_rebuild = (self.config.sample_rate - clamped.sample_rate).abs() > f32::EPSILON
            || (self.config.scroll_speed - clamped.scroll_speed).abs() > f32::EPSILON
            || self.config.downsample != clamped.downsample;

        self.config = clamped;
        if needs_rebuild {
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

#[inline]
fn column_period_seconds(config: &WaveformConfig) -> f32 {
    1.0 / config.scroll_speed.max(MIN_SCROLL_SPEED)
}

#[inline]
fn compute_frequency(
    processor: &WaveformProcessor,
    channel: usize,
    bucket_duration: f64,
    previous: Option<f32>,
) -> f32 {
    let crossing_count = processor.crossing_counts.get(channel).copied().unwrap_or(0);

    if crossing_count < 2 {
        return fallback_frequency(crossing_count, bucket_duration, previous);
    }

    let (Some(&Some(first)), Some(&Some(last))) = (
        processor.first_crossing_times.get(channel),
        processor.last_crossing_times.get(channel),
    ) else {
        return fallback_frequency(crossing_count, bucket_duration, previous);
    };

    let span = last - first;
    if span > f64::EPSILON {
        let cycles = (crossing_count - 1) as f64 / 2.0;
        normalize_frequency((cycles / span) as f32)
    } else {
        fallback_frequency(crossing_count, bucket_duration, previous)
    }
}

#[inline]
fn fallback_frequency(crossing_count: usize, bucket_duration: f64, previous: Option<f32>) -> f32 {
    if crossing_count == 0 || bucket_duration <= f64::EPSILON {
        return previous.unwrap_or(0.0);
    }

    let cycles = crossing_count as f64 / 2.0;
    if cycles <= f64::EPSILON {
        return previous.unwrap_or(0.0);
    }

    let freq_hz = (cycles / bucket_duration) as f32;
    normalize_frequency(freq_hz)
}

#[inline]
fn normalize_frequency(freq: f32) -> f32 {
    if freq <= MIN_FREQUENCY_HZ {
        return 0.0;
    }
    let clamped = freq.clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ);
    ((clamped.log10() - MIN_FREQUENCY_HZ.log10())
        / (MAX_FREQUENCY_HZ.log10() - MIN_FREQUENCY_HZ.log10()))
    .clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;
    use std::time::Instant;

    fn make_block<'a>(samples: &'a [f32], channels: usize, sample_rate: f32) -> AudioBlock<'a> {
        AudioBlock::new(samples, channels, sample_rate, Instant::now())
    }

    fn nominal_frames_per_column(config: &WaveformConfig) -> f32 {
        config.sample_rate.max(1.0) / config.scroll_speed.max(MIN_SCROLL_SPEED)
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
        assert!(
            (last_frequency - 0.44).abs() < 0.02,
            "expected normalized frequency near 0.44, got {}",
            last_frequency
        );
    }

    #[test]
    fn frequency_estimate_remains_stable_at_high_scroll() {
        let frequency_hz = 220.0;
        let mut processor = WaveformProcessor::new(WaveformConfig {
            sample_rate: 48_000.0,
            scroll_speed: 400.0,
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
        assert!(
            last_frequency > 0.3 && last_frequency < 0.5,
            "expected reasonable normalized frequency, got {}",
            last_frequency
        );
    }

    #[test]
    fn produces_columns_over_time() {
        let mut processor = WaveformProcessor::new(WaveformConfig {
            sample_rate: 48_000.0,
            scroll_speed: 100.0,
            downsample: DownsampleStrategy::MinMax,
            max_columns: DEFAULT_COLUMN_CAPACITY,
        });

        let channels = 2;
        let frames_per_column = nominal_frames_per_column(&processor.config).ceil() as usize;
        let total_frames = frames_per_column * 5;
        let mut samples = Vec::with_capacity(total_frames * channels);
        for _n in 0..total_frames {
            samples.push(0.1);
            samples.push(-0.1);
        }

        let block = make_block(&samples, channels, processor.config.sample_rate);
        let snapshot = match processor.process_block(&block) {
            ProcessorUpdate::Snapshot(snapshot) => snapshot,
            ProcessorUpdate::None => panic!("expected snapshot"),
        };

        assert_eq!(snapshot.channels, channels);
        assert!(snapshot.columns >= 4, "should produce multiple columns");
        assert_eq!(snapshot.min_values.len(), snapshot.columns * channels);
        assert_eq!(snapshot.max_values.len(), snapshot.columns * channels);
    }
}
