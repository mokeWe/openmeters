//! Real-time waveform DSP implementation for the scrolling waveform visual.

use super::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::util::audio::DEFAULT_SAMPLE_RATE;

pub const MIN_SCROLL_SPEED: f32 = 10.0;
pub const MAX_SCROLL_SPEED: f32 = 1000.0;
pub const DEFAULT_SCROLL_SPEED: f32 = 80.0;
const MIN_FREQUENCY_HZ: f32 = 20.0;
const MAX_FREQUENCY_HZ: f32 = 20_000.0;
const HISTORY_SECONDS: f32 = 0.75;
const HISTORY_MIN_FRAMES: usize = 4_096;
const HISTORY_MAX_FRAMES: usize = 262_144;

/// Configuration for the waveform preview.
#[derive(Debug, Clone, Copy)]
pub struct WaveformConfig {
    pub sample_rate: f32,
    /// How quickly the waveform scrolls, expressed as pixels per second.
    pub scroll_speed: f32,
}

impl Default for WaveformConfig {
    fn default() -> Self {
        Self {
            sample_rate: DEFAULT_SAMPLE_RATE,
            scroll_speed: DEFAULT_SCROLL_SPEED,
        }
    }
}

/// Snapshot containing raw waveform samples ready for presentation.
#[derive(Debug, Clone, Default)]
pub struct WaveformSnapshot {
    pub channels: usize,
    pub frames: usize,
    pub sample_rate: f32,
    pub samples: Vec<f32>,
    pub frequency_normalized: Vec<f32>,
    pub scroll_position: f32,
}

#[derive(Debug, Clone)]
pub struct WaveformProcessor {
    config: WaveformConfig,
    snapshot: WaveformSnapshot,
    channels: usize,
    history_capacity: usize,
    raw_history: Vec<RawChannelHistory>,
    frequency_history: ScalarHistory,
    total_samples_written: u64,
    current_frequency: f32,
    last_sample: Option<f32>,
    last_rising_crossing: Option<f64>,
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
            let idx = (start + i) % self.capacity().max(1);
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
        self.write_pos = (self.write_pos + 1) % self.capacity().max(1);
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
        if self.len == 0 || dest.is_empty() {
            return;
        }

        let mut index = self.start_index();
        for frame in 0..self.len {
            let sample = self.samples[index];
            let dest_index = frame * channels + channel_index;
            if dest_index < dest.len() {
                dest[dest_index] = sample;
            }
            index = (index + 1) % self.capacity().max(1);
        }
    }
}

#[derive(Debug, Clone)]
struct ScalarHistory {
    values: Vec<f32>,
    write_pos: usize,
    len: usize,
}

impl ScalarHistory {
    fn new(capacity: usize) -> Self {
        Self {
            values: vec![0.0; capacity],
            write_pos: 0,
            len: 0,
        }
    }

    fn capacity(&self) -> usize {
        self.values.len()
    }

    fn ensure_capacity(&mut self, capacity: usize) {
        if self.capacity() == capacity {
            return;
        }

        let mut resized = ScalarHistory::new(capacity);
        let keep = self.len.min(capacity);
        let start = self.start_index();
        for i in 0..keep {
            let idx = (start + i) % self.capacity().max(1);
            resized.values[i] = self.values[idx];
        }
        resized.write_pos = keep % capacity.max(1);
        resized.len = keep;
        *self = resized;
    }

    fn clear(&mut self) {
        self.values.fill(0.0);
        self.write_pos = 0;
        self.len = 0;
    }

    fn push(&mut self, value: f32) {
        if self.values.is_empty() {
            return;
        }

        self.values[self.write_pos] = value;
        self.write_pos = (self.write_pos + 1) % self.capacity().max(1);
        if self.len < self.capacity() {
            self.len += 1;
        }
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

    fn write_linear(&self, dest: &mut [f32]) {
        if self.len == 0 || dest.is_empty() {
            return;
        }

        let mut index = self.start_index();
        for frame in 0..self.len {
            if frame < dest.len() {
                dest[frame] = self.values[index];
            }
            index = (index + 1) % self.capacity().max(1);
        }
    }
}

impl WaveformProcessor {
    pub fn new(config: WaveformConfig) -> Self {
        let clamped = clamp_config(config);
        let mut processor = Self {
            config: clamped,
            snapshot: WaveformSnapshot::default(),
            channels: 2,
            history_capacity: 0,
            raw_history: Vec::new(),
            frequency_history: ScalarHistory::new(0),
            total_samples_written: 0,
            current_frequency: 0.0,
            last_sample: None,
            last_rising_crossing: None,
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

    fn rebuild_for_config(&mut self) {
        self.history_capacity = self.compute_history_capacity();
        self.raw_history = (0..self.channels.max(1))
            .map(|_| RawChannelHistory::new(self.history_capacity))
            .collect();
        self.frequency_history = ScalarHistory::new(self.history_capacity);
        self.populate_snapshot();
    }

    fn ensure_capacity(&mut self) {
        let target = self.compute_history_capacity();
        if target == self.history_capacity {
            for history in &mut self.raw_history {
                history.ensure_capacity(target);
            }
            self.frequency_history.ensure_capacity(target);
            return;
        }

        self.history_capacity = target;
        self.raw_history
            .iter_mut()
            .for_each(|history| history.ensure_capacity(target));
        self.frequency_history.ensure_capacity(target);
    }

    fn compute_history_capacity(&self) -> usize {
        let sample_rate = self.config.sample_rate.max(1.0);
        let frames_from_seconds = (sample_rate * HISTORY_SECONDS).ceil() as usize;
        frames_from_seconds
            .max(HISTORY_MIN_FRAMES)
            .min(HISTORY_MAX_FRAMES)
            .max(1)
    }

    fn ingest_samples(&mut self, samples: &[f32], channels: usize) {
        if samples.is_empty() || channels == 0 {
            return;
        }

        let sample_rate = self.config.sample_rate.max(1.0);
        for frame in samples.chunks_exact(channels) {
            let primary = frame[0];
            let normalized_frequency = self.update_frequency(primary, sample_rate);

            for (channel_index, history) in self.raw_history.iter_mut().enumerate() {
                let sample = frame.get(channel_index).copied().unwrap_or(0.0);
                history.push(sample);
            }

            self.frequency_history.push(normalized_frequency);
            self.total_samples_written = self.total_samples_written.saturating_add(1);
        }
    }

    fn update_frequency(&mut self, sample: f32, sample_rate: f32) -> f32 {
        let prev = self.last_sample.unwrap_or(sample);
        let mut frequency = self.current_frequency;

        if prev <= 0.0 && sample > 0.0 {
            let slope = sample - prev;
            if slope.abs() > f32::EPSILON {
                let frac = (-prev) / slope;
                let crossing = (self.total_samples_written as f64) + frac as f64;
                if let Some(last) = self.last_rising_crossing {
                    let period = crossing - last;
                    if period > f64::EPSILON {
                        let hz = (sample_rate as f64) / period;
                        frequency = normalize_frequency(hz as f32);
                    }
                }
                self.last_rising_crossing = Some(crossing);
            }
        }

        self.last_sample = Some(sample);
        self.current_frequency = frequency;
        frequency
    }

    fn populate_snapshot(&mut self) {
        let channels = self.channels.max(1);
        let frames = self
            .raw_history
            .first()
            .map(|history| history.len())
            .unwrap_or(0);

        let required_samples = frames.saturating_mul(channels);
        if self.snapshot.samples.len() < required_samples {
            self.snapshot.samples.resize(required_samples, 0.0);
        }
        if frames == 0 {
            self.snapshot.samples.clear();
        }

        for (channel_index, history) in self.raw_history.iter().enumerate() {
            history.write_interleaved(&mut self.snapshot.samples, channels, channel_index);
        }

        if self.snapshot.frequency_normalized.len() < frames {
            self.snapshot
                .frequency_normalized
                .resize(frames, self.current_frequency);
        }
        if frames == 0 {
            self.snapshot.frequency_normalized.clear();
        } else {
            self.frequency_history
                .write_linear(&mut self.snapshot.frequency_normalized[..frames]);
        }

        self.snapshot.channels = channels;
        self.snapshot.frames = frames;
        self.snapshot.sample_rate = self.config.sample_rate;
    }

    fn frames_per_pixel(&self) -> f32 {
        let sample_rate = self.config.sample_rate.max(1.0);
        let speed = self.config.scroll_speed.max(MIN_SCROLL_SPEED);
        sample_rate / speed
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
            self.rebuild_for_config();
        }

        let sample_rate = block.sample_rate.max(1.0);
        if (self.config.sample_rate - sample_rate).abs() > f32::EPSILON {
            self.config.sample_rate = sample_rate;
            self.rebuild_for_config();
        }

        self.ensure_capacity();
        self.ingest_samples(block.samples, channels);
        self.populate_snapshot();

        let frames_per_pixel = self.frames_per_pixel().max(1.0);
        let advance = block.frame_count() as f32 / frames_per_pixel;
        self.snapshot.scroll_position += advance;

        ProcessorUpdate::Snapshot(self.snapshot.clone())
    }

    fn reset(&mut self) {
        self.snapshot = WaveformSnapshot::default();
        self.history_capacity = self.compute_history_capacity();
        self.raw_history
            .iter_mut()
            .for_each(|history| history.clear());
        self.frequency_history.clear();
        self.total_samples_written = 0;
        self.current_frequency = 0.0;
        self.last_sample = None;
        self.last_rising_crossing = None;
    }
}

impl Reconfigurable<WaveformConfig> for WaveformProcessor {
    fn update_config(&mut self, config: WaveformConfig) {
        let clamped = clamp_config(config);
        let requires_rebuild = (self.config.sample_rate - clamped.sample_rate).abs() > f32::EPSILON;
        let speed_changed = (self.config.scroll_speed - clamped.scroll_speed).abs() > f32::EPSILON;

        self.config = clamped;
        if requires_rebuild {
            self.rebuild_for_config();
        }

        if speed_changed {
            // Keep scroll continuity by anchoring to last known position.
            self.snapshot.scroll_position = self.snapshot.scroll_position;
        }
    }
}

fn clamp_config(mut config: WaveformConfig) -> WaveformConfig {
    config.sample_rate = config.sample_rate.max(1.0);
    config.scroll_speed = config
        .scroll_speed
        .clamp(MIN_SCROLL_SPEED, MAX_SCROLL_SPEED);
    config
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
    fn retains_recent_samples() {
        let sample_rate = 48_000.0;
        let mut processor = WaveformProcessor::new(WaveformConfig {
            sample_rate,
            scroll_speed: DEFAULT_SCROLL_SPEED,
        });

        let frames = 10_000;
        let mut samples = Vec::with_capacity(frames * 2);
        for n in 0..frames {
            let value = (n as f32 / frames as f32) * 2.0 - 1.0;
            samples.push(value);
            samples.push(-value);
        }

        let block = make_block(&samples, 2, sample_rate);
        let snapshot = match processor.process_block(&block) {
            ProcessorUpdate::Snapshot(snapshot) => snapshot,
            ProcessorUpdate::None => panic!("expected snapshot"),
        };

        assert_eq!(snapshot.channels, 2);
        assert!(snapshot.frames <= processor.history_capacity);
        assert_eq!(snapshot.samples.len(), snapshot.frames * snapshot.channels);
        let last_frame = snapshot.samples.chunks(snapshot.channels).last().unwrap();
        assert!((last_frame[0] - ((frames - 1) as f32 / frames as f32 * 2.0 - 1.0)).abs() < 1e-6);
    }

    #[test]
    fn frequency_estimation_tracks_sine_wave() {
        let sample_rate = 48_000.0;
        let frequency = 440.0;
        let mut processor = WaveformProcessor::new(WaveformConfig {
            sample_rate,
            scroll_speed: DEFAULT_SCROLL_SPEED,
        });

        let total_frames = (sample_rate * 0.2) as usize;
        let mut samples = Vec::with_capacity(total_frames);
        for n in 0..total_frames {
            let t = n as f32 / sample_rate;
            samples.push((2.0 * PI * frequency * t).sin());
        }

        let block = make_block(&samples, 1, sample_rate);
        let snapshot = match processor.process_block(&block) {
            ProcessorUpdate::Snapshot(snapshot) => snapshot,
            ProcessorUpdate::None => panic!("expected snapshot"),
        };

        assert_eq!(snapshot.channels, 1);
        assert!(snapshot.frames > 0);
        let last_frequency = snapshot
            .frequency_normalized
            .last()
            .copied()
            .unwrap_or_default();
        assert!(last_frequency > 0.4 && last_frequency < 0.46);
    }

    #[test]
    fn reset_clears_history() {
        let mut processor = WaveformProcessor::new(WaveformConfig::default());
        let samples = vec![0.5; 2048];
        let block = make_block(&samples, 1, DEFAULT_SAMPLE_RATE);
        processor.process_block(&block);
        processor.reset();

        assert_eq!(processor.snapshot.frames, 0);
        assert!(processor.snapshot.samples.is_empty());
        assert!(processor.snapshot.frequency_normalized.is_empty());
    }
}
