//! UI wrapper for the spectrogram DSP processor and renderer.

use crate::audio::meter_tap::MeterFormat;
use crate::dsp::spectrogram::{
    SpectrogramColumn, SpectrogramConfig, SpectrogramProcessor as CoreSpectrogramProcessor,
    SpectrogramUpdate,
};
use crate::dsp::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::ui::render::spectrogram::{
    ColumnBuffer, ColumnBufferPool, SPECTROGRAM_PALETTE_SIZE, SpectrogramColumnUpdate,
    SpectrogramParams, SpectrogramPrimitive,
};
use crate::ui::theme;
use crate::util::audio::DEFAULT_SAMPLE_RATE;
use iced::advanced::Renderer as _;
use iced::advanced::renderer::{self, Quad};
use iced::advanced::widget::{Tree, tree};
use iced::advanced::{Layout, Widget, layout, mouse};
use iced::{Background, Color, Element, Length, Rectangle, Size};
use iced_wgpu::primitive::Renderer as _;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::fmt;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::Instant;

const DEFAULT_CHANNELS: usize = 2;
const DEFAULT_DB_FLOOR: f32 = -96.0;
const DEFAULT_DB_CEILING: f32 = 0.0;
const PALETTE_STOPS: usize = SPECTROGRAM_PALETTE_SIZE;
const MIN_INCREMENTAL_UPDATES: u32 = 16;
/// Maximum number of spectral rows we can store in a single GPU texture.
/// Mirrors `wgpu::Limits::downlevel_defaults().max_texture_dimension_2d` which
/// is 8192 on all supported backends. Staying within this guard prevents
/// `Device::create_texture` validation failures when large FFT/zero-padding
/// combinations are selected.
const MAX_TEXTURE_BINS: usize = 8_192;

static NEXT_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

fn next_instance_id() -> u64 {
    NEXT_INSTANCE_ID.fetch_add(1, Ordering::Relaxed)
}

/// Bridges the DSP spectrogram processor into the UI layer by
/// constructing audio blocks and forwarding configuration changes.
pub struct SpectrogramProcessor {
    inner: CoreSpectrogramProcessor,
    channels: usize,
}

impl fmt::Debug for SpectrogramProcessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpectrogramProcessor")
            .field("channels", &self.channels)
            .finish_non_exhaustive()
    }
}

impl SpectrogramProcessor {
    pub fn new(sample_rate: f32) -> Self {
        let config = SpectrogramConfig {
            sample_rate,
            use_reassignment: true,
            ..Default::default()
        };
        Self {
            inner: CoreSpectrogramProcessor::new(config),
            channels: DEFAULT_CHANNELS,
        }
    }

    pub fn ingest(
        &mut self,
        samples: &[f32],
        format: MeterFormat,
    ) -> ProcessorUpdate<SpectrogramUpdate> {
        if samples.is_empty() {
            return ProcessorUpdate::None;
        }

        let channels = format.channels.max(1);
        if self.channels != channels {
            self.channels = channels;
        }

        let sample_rate = format.sample_rate.max(1.0);
        let mut config = self.inner.config();
        if (config.sample_rate - sample_rate).abs() > f32::EPSILON {
            config.sample_rate = sample_rate;
            self.inner.update_config(config);
        }

        let block = AudioBlock::new(samples, self.channels, sample_rate, Instant::now());

        self.inner.process_block(&block)
    }

    #[allow(dead_code)]
    pub fn update_config(&mut self, config: SpectrogramConfig) {
        self.inner.update_config(config);
    }

    pub fn config(&self) -> SpectrogramConfig {
        self.inner.config()
    }
}

/// Captures all UI-facing spectrogram state, including cached columns,
/// palette information, and configuration derived from incoming updates.
#[derive(Debug, Clone)]
pub struct SpectrogramState {
    buffer: RefCell<SpectrogramBuffer>,
    style: SpectrogramStyle,
    palette: [Color; PALETTE_STOPS],
    palette_cache: RefCell<PaletteCache>,
    last_timestamp: Option<Instant>,
    history: VecDeque<SpectrogramColumn>,
    fft_size: usize,
    hop_size: usize,
    sample_rate: f32,
    history_length: usize,
    synchro_bins_hz: Option<Arc<[f32]>>,
    instance_id: u64,
}

impl SpectrogramState {
    pub fn new() -> Self {
        let style = SpectrogramStyle::default();
        let palette = theme::spectrogram_palette();
        let default_cfg = SpectrogramConfig::default();
        let palette_cache = RefCell::new(PaletteCache::new(&style, &palette));

        Self {
            buffer: RefCell::new(SpectrogramBuffer::new()),
            style,
            palette,
            palette_cache,
            last_timestamp: None,
            history: VecDeque::new(),
            fft_size: default_cfg.fft_size,
            hop_size: default_cfg.hop_size,
            sample_rate: default_cfg.sample_rate,
            history_length: default_cfg.history_length,
            synchro_bins_hz: None,
            instance_id: next_instance_id(),
        }
    }

    #[allow(dead_code)]
    pub fn set_style(&mut self, style: SpectrogramStyle) {
        if self.style == style {
            return;
        }

        let floor_changed = (self.style.floor_db - style.floor_db).abs() > f32::EPSILON;
        let ceiling_changed = (self.style.ceiling_db - style.ceiling_db).abs() > f32::EPSILON;
        let contrast_changed = (self.style.contrast - style.contrast).abs() > f32::EPSILON;

        self.style = style;
        self.palette_cache.borrow_mut().dirty = true;

        if floor_changed || ceiling_changed {
            let use_synchro = {
                let buffer = self.buffer.borrow();
                buffer.using_synchro()
            };
            let synchro_bins = if use_synchro {
                self.synchro_bins_hz.as_deref()
            } else {
                None
            };
            self.buffer.borrow_mut().rebuild_from_history(
                &self.history,
                RebuildContext {
                    style: &self.style,
                    history_length: self.history_length,
                    sample_rate: self.sample_rate,
                    fft_size: self.fft_size,
                    use_synchro,
                    synchro_bins_hz: synchro_bins,
                },
            );
        } else {
            self.buffer.borrow_mut().mark_dirty();
        }

        if contrast_changed && !(floor_changed || ceiling_changed) {
            self.buffer.borrow_mut().mark_dirty();
        }
    }

    #[allow(dead_code)]
    pub fn set_palette(&mut self, palette: [Color; PALETTE_STOPS]) {
        if self.palette == palette {
            return;
        }

        self.palette = palette;
        self.palette_cache.borrow_mut().dirty = true;
        self.buffer.borrow_mut().mark_dirty();
    }

    pub fn apply_update(&mut self, update: &SpectrogramUpdate) {
        let history_length_changed = self.history_length != update.history_length;
        let fft_size_changed = self.fft_size != update.fft_size;
        let hop_size_changed = self.hop_size != update.hop_size;
        let sample_rate_changed = (self.sample_rate - update.sample_rate).abs() > f32::EPSILON;

        let mut needs_rebuild = update.reset
            || history_length_changed
            || fft_size_changed
            || hop_size_changed
            || sample_rate_changed;

        if update.reset {
            self.history.clear();
            self.last_timestamp = None;
        }

        let (use_synchro, synchro_bins) =
            Self::compute_synchro_state(&update.new_columns, &update.synchro_bins_hz);
        let incoming_height = Self::pending_height(&update.new_columns, use_synchro);

        let existing_height = Self::history_height(&self.history, use_synchro);
        if incoming_height
            .zip(existing_height)
            .is_some_and(|(incoming, existing)| incoming != existing)
        {
            needs_rebuild = true;
        }

        Self::push_history(
            &mut self.history,
            &update.new_columns,
            update.history_length,
        );

        self.synchro_bins_hz = match (use_synchro, synchro_bins) {
            (true, Some(bins)) => Some(bins),
            _ => None,
        };

        let desired_height = Self::history_height(&self.history, use_synchro)
            .map(|len| len as u32)
            .unwrap_or(0);

        let buffer_requires_rebuild = {
            let buffer = self.buffer.borrow();
            buffer.requires_rebuild(update, use_synchro, desired_height)
        };
        needs_rebuild |= buffer_requires_rebuild;

        let mut buffer = self.buffer.borrow_mut();
        buffer.clear_pending();
        let synchro_bins = self.synchro_bins_hz.as_deref();
        if needs_rebuild {
            buffer.rebuild_from_history(
                &self.history,
                RebuildContext {
                    style: &self.style,
                    history_length: update.history_length,
                    sample_rate: update.sample_rate,
                    fft_size: update.fft_size,
                    use_synchro,
                    synchro_bins_hz: synchro_bins,
                },
            );
        } else if !update.new_columns.is_empty() {
            buffer.append_columns(&update.new_columns, &self.style, use_synchro, synchro_bins);
        }

        self.history_length = update.history_length;
        self.fft_size = update.fft_size;
        self.hop_size = update.hop_size;
        self.sample_rate = update.sample_rate;

        if let Some(last) = update.new_columns.last() {
            self.last_timestamp = Some(last.timestamp);
        } else if update.reset {
            self.last_timestamp = None;
        }
    }

    pub fn visual_params(&self, bounds: Rectangle) -> Option<SpectrogramParams> {
        let (palette, background) = self.cached_palette_and_background();

        let mut buffer = self.buffer.borrow_mut();

        let width = buffer.texture_width();
        let height = buffer.texture_height();
        let column_count = buffer.column_count();

        if width == 0 || height == 0 || column_count == 0 {
            buffer.clear_pending();
            return None;
        }

        Some(SpectrogramParams {
            instance_id: self.instance_id,
            bounds,
            texture_width: width,
            texture_height: height,
            column_count,
            latest_column: buffer.latest_column(),
            base_data: buffer.take_base_data(),
            column_updates: buffer.take_updates(),
            palette,
            background,
            contrast: self.style.contrast,
        })
    }

    fn cached_palette_and_background(&self) -> ([[f32; 4]; PALETTE_STOPS], [f32; 4]) {
        let mut cache = self.palette_cache.borrow_mut();
        if cache.dirty {
            cache.refresh(&self.style, &self.palette);
        }
        (cache.palette, cache.background)
    }

    fn compute_synchro_state(
        columns: &[SpectrogramColumn],
        bins: &Option<Arc<[f32]>>,
    ) -> (bool, Option<Arc<[f32]>>) {
        match bins {
            Some(bins) if !bins.is_empty() => {
                let expected_len = bins.len();
                let compatible = columns.is_empty()
                    || columns.iter().all(|column| {
                        column
                            .synchro_magnitudes_db
                            .as_ref()
                            .map(|values| values.len() == expected_len)
                            .unwrap_or(false)
                    });
                if compatible {
                    (true, Some(Arc::clone(bins)))
                } else {
                    (false, None)
                }
            }
            _ => (false, None),
        }
    }

    fn column_height(column: &SpectrogramColumn, use_synchro: bool) -> Option<usize> {
        SpectrogramBuffer::column_values(column, use_synchro)
            .map(|values| values.len().min(MAX_TEXTURE_BINS))
    }

    fn history_height(history: &VecDeque<SpectrogramColumn>, use_synchro: bool) -> Option<usize> {
        history
            .iter()
            .find_map(|column| Self::column_height(column, use_synchro))
    }

    fn pending_height(columns: &[SpectrogramColumn], use_synchro: bool) -> Option<usize> {
        columns
            .iter()
            .find_map(|column| Self::column_height(column, use_synchro))
    }

    fn push_history(
        history: &mut VecDeque<SpectrogramColumn>,
        columns: &[SpectrogramColumn],
        capacity: usize,
    ) {
        if capacity == 0 {
            history.clear();
            return;
        }

        if columns.is_empty() {
            return;
        }

        history.extend(columns.iter().cloned());

        if history.len() > capacity {
            let overflow = history.len() - capacity;
            history.drain(0..overflow);
        }
    }
}

/// Visual appearance controls that can be overridden by user preferences.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectrogramStyle {
    pub background: Color,
    pub floor_db: f32,
    pub ceiling_db: f32,
    pub opacity: f32,
    pub contrast: f32,
}

impl Default for SpectrogramStyle {
    fn default() -> Self {
        Self {
            background: Color::from_rgba(0.0, 0.0, 0.0, 0.0),
            floor_db: DEFAULT_DB_FLOOR,
            ceiling_db: DEFAULT_DB_CEILING,
            opacity: 0.95,
            contrast: 1.4,
        }
    }
}

/// Circular texture backing for the spectrogram visualization coupled with
/// bookkeeping for incremental GPU updates.
#[derive(Debug)]
struct SpectrogramBuffer {
    capacity: u32,
    height: u32,
    values: Vec<f32>,
    write_index: u32,
    column_count: u32,
    last_timestamp: Option<Instant>,
    pending_base: Option<Arc<[f32]>>,
    pending_updates: Vec<SpectrogramColumnUpdate>,
    sample_rate: f32,
    fft_size: usize,
    row_bin_positions: Vec<f32>,
    row_lower_bins: Vec<usize>,
    row_upper_bins: Vec<usize>,
    row_interp_weights: Vec<f32>,
    using_synchro: bool,
    grid_bin_frequencies: Vec<f32>,
    update_buffer_pool: ColumnBufferPool,
}

/// Bundles configuration needed when rebuilding the buffer from scratch so we
/// can pass a single struct instead of a long parameter list.
struct RebuildContext<'a> {
    style: &'a SpectrogramStyle,
    history_length: usize,
    sample_rate: f32,
    fft_size: usize,
    use_synchro: bool,
    synchro_bins_hz: Option<&'a [f32]>,
}

impl Clone for SpectrogramBuffer {
    fn clone(&self) -> Self {
        Self {
            capacity: self.capacity,
            height: self.height,
            values: self.values.clone(),
            write_index: self.write_index,
            column_count: self.column_count,
            last_timestamp: self.last_timestamp,
            pending_base: self.pending_base.clone(),
            pending_updates: self.pending_updates.clone(),
            sample_rate: self.sample_rate,
            fft_size: self.fft_size,
            row_bin_positions: self.row_bin_positions.clone(),
            row_lower_bins: self.row_lower_bins.clone(),
            row_upper_bins: self.row_upper_bins.clone(),
            row_interp_weights: self.row_interp_weights.clone(),
            using_synchro: self.using_synchro,
            grid_bin_frequencies: self.grid_bin_frequencies.clone(),
            update_buffer_pool: self.update_buffer_pool.clone(),
        }
    }
}

impl SpectrogramBuffer {
    fn new() -> Self {
        Self {
            capacity: 0,
            height: 0,
            values: Vec::new(),
            write_index: 0,
            column_count: 0,
            last_timestamp: None,
            pending_base: None,
            pending_updates: Vec::new(),
            sample_rate: DEFAULT_SAMPLE_RATE,
            fft_size: 0,
            row_bin_positions: Vec::new(),
            row_lower_bins: Vec::new(),
            row_upper_bins: Vec::new(),
            row_interp_weights: Vec::new(),
            using_synchro: false,
            grid_bin_frequencies: Vec::new(),
            update_buffer_pool: ColumnBufferPool::new(),
        }
    }

    fn texture_width(&self) -> u32 {
        self.capacity
    }

    fn texture_height(&self) -> u32 {
        self.height
    }

    fn column_count(&self) -> u32 {
        self.column_count
    }

    fn using_synchro(&self) -> bool {
        self.using_synchro
    }

    fn has_dimensions(&self) -> bool {
        self.capacity != 0 && self.height != 0
    }

    fn column_values(column: &SpectrogramColumn, use_synchro: bool) -> Option<&[f32]> {
        if use_synchro {
            column
                .synchro_magnitudes_db
                .as_ref()
                .map(|values| values.as_ref())
        } else {
            Some(column.magnitudes_db.as_ref())
        }
    }

    fn latest_column(&self) -> u32 {
        if self.column_count == 0 || self.capacity == 0 {
            0
        } else {
            (self.write_index + self.capacity - 1) % self.capacity
        }
    }

    fn take_base_data(&mut self) -> Option<Arc<[f32]>> {
        self.pending_base.take()
    }

    fn take_updates(&mut self) -> Vec<SpectrogramColumnUpdate> {
        std::mem::take(&mut self.pending_updates)
    }

    fn clear_pending(&mut self) {
        self.pending_base = None;
        self.pending_updates.clear();
    }

    fn queue_full_refresh(&mut self) {
        if self.values.is_empty() {
            self.clear_pending();
            return;
        }

        self.pending_base = Some(Arc::from(self.values.clone().into_boxed_slice()));
        self.pending_updates.clear();
    }

    fn requires_rebuild(
        &self,
        update: &SpectrogramUpdate,
        use_synchro: bool,
        desired_height: u32,
    ) -> bool {
        if !self.has_dimensions() {
            return true;
        }

        self.using_synchro != use_synchro
            || (desired_height > 0 && self.texture_height() != desired_height)
            || self.capacity != update.history_length as u32
            || (self.sample_rate - update.sample_rate).abs() > f32::EPSILON
            || self.fft_size != update.fft_size
    }

    fn mark_dirty(&mut self) {
        if self.values.is_empty() {
            self.clear_pending();
        } else if self.pending_base.is_none() || !self.pending_updates.is_empty() {
            self.queue_full_refresh();
        }
    }

    #[allow(dead_code)]
    fn reset(&mut self) {
        self.capacity = 0;
        self.height = 0;
        self.values.clear();
        self.write_index = 0;
        self.column_count = 0;
        self.last_timestamp = None;
        self.clear_pending();
        self.row_bin_positions.clear();
        self.row_lower_bins.clear();
        self.row_upper_bins.clear();
        self.row_interp_weights.clear();
        self.using_synchro = false;
        self.grid_bin_frequencies.clear();
        self.update_buffer_pool = ColumnBufferPool::new();
    }

    fn rebuild_from_history(
        &mut self,
        history: &VecDeque<SpectrogramColumn>,
        params: RebuildContext<'_>,
    ) {
        let RebuildContext {
            style,
            history_length,
            sample_rate,
            fft_size,
            use_synchro,
            synchro_bins_hz,
        } = params;

        self.clear_pending();

        self.capacity = history_length as u32;
        self.sample_rate = if sample_rate <= 0.0 {
            DEFAULT_SAMPLE_RATE
        } else {
            sample_rate
        };
        self.fft_size = fft_size.max(1);
        self.using_synchro = use_synchro;

        let requested_height = history
            .iter()
            .find_map(|column| {
                SpectrogramBuffer::column_values(column, use_synchro)
                    .map(|values| values.len().min(MAX_TEXTURE_BINS) as u32)
            })
            .unwrap_or(0);
        self.height = requested_height;

        if self.capacity == 0 || self.height == 0 {
            self.values.clear();
            self.write_index = 0;
            self.column_count = 0;
            self.last_timestamp = history.back().map(|column| column.timestamp);
            self.row_bin_positions.clear();
            self.row_lower_bins.clear();
            self.row_upper_bins.clear();
            self.row_interp_weights.clear();
            self.using_synchro = false;
            self.sync_grid_frequencies(None);
            self.pending_base = None;
            self.pending_updates.clear();
            return;
        }

        self.sync_grid_frequencies(synchro_bins_hz);

        self.values
            .resize(self.capacity as usize * self.height as usize, 0.0);
        self.write_index = 0;
        self.column_count = 0;
        self.last_timestamp = None;
        self.rebuild_row_positions();

        for column in history.iter() {
            let Some(values) = SpectrogramBuffer::column_values(column, use_synchro) else {
                continue;
            };
            if values.len() < self.height as usize {
                continue;
            }
            self.push_column(values, style);
        }

        if self.column_count > 0 {
            self.queue_full_refresh();
        } else {
            self.clear_pending();
        }
        self.last_timestamp = history.back().map(|column| column.timestamp);
    }

    fn append_columns(
        &mut self,
        columns: &[SpectrogramColumn],
        style: &SpectrogramStyle,
        use_synchro: bool,
        synchro_bins_hz: Option<&[f32]>,
    ) {
        if !self.has_dimensions() {
            return;
        }

        self.ensure_row_mapping(use_synchro);
        self.sync_grid_frequencies(synchro_bins_hz);

        for column in columns {
            let Some(values) = SpectrogramBuffer::column_values(column, use_synchro) else {
                continue;
            };

            if values.len() < self.height as usize {
                continue;
            }

            let physical_index = self.push_column(values, style);
            let start = (physical_index as usize) * (self.height as usize);
            let mut buffer = self.update_buffer_pool.acquire(self.height as usize);
            buffer.copy_from_slice(&self.values[start..start + self.height as usize]);
            let values = Arc::new(ColumnBuffer::new(buffer, self.update_buffer_pool.clone()));
            self.pending_updates.push(SpectrogramColumnUpdate {
                column_index: physical_index,
                values,
            });
        }

        if self.pending_updates.len() as u32 >= self.max_incremental_updates() {
            self.queue_full_refresh();
        }

        if let Some(last) = columns.last() {
            self.last_timestamp = Some(last.timestamp);
        }
    }

    fn push_column(&mut self, magnitudes_db: &[f32], style: &SpectrogramStyle) -> u32 {
        if self.capacity == 0 || self.height == 0 {
            return 0;
        }

        let height = self.height as usize;
        let column = self.write_index;
        let start = (column as usize) * height;

        if self.values.len() != (self.capacity as usize * height) {
            self.values.resize(self.capacity as usize * height, 0.0);
        }

        if self.needs_row_sampling_refresh() {
            self.rebuild_row_positions();
        }

        let floor = style.floor_db;
        let ceiling = style.ceiling_db;
        let range = (ceiling - floor).max(f32::EPSILON);
        let inv_range = 1.0 / range;
        let bin_count = magnitudes_db.len();
        if bin_count == 0 {
            return column;
        }

        for row in 0..height {
            let lower = self.row_lower_bins[row].min(bin_count - 1);
            let upper = self.row_upper_bins[row].min(bin_count - 1);
            let frac = self.row_interp_weights[row].clamp(0.0, 1.0);

            let sample = if lower == upper {
                magnitudes_db[lower]
            } else {
                let lower_val = magnitudes_db[lower];
                let upper_val = magnitudes_db[upper];
                lower_val + (upper_val - lower_val) * frac
            };
            let normalized = (sample.clamp(floor, ceiling) - floor) * inv_range;
            let target = start + row;
            self.values[target] = normalized.clamp(0.0, 1.0);
        }

        let written = column;
        if self.column_count < self.capacity {
            self.column_count += 1;
        }
        self.write_index = (self.write_index + 1) % self.capacity.max(1);
        written
    }

    fn rebuild_row_positions(&mut self) {
        self.row_bin_positions.clear();
        self.row_lower_bins.clear();
        self.row_upper_bins.clear();
        self.row_interp_weights.clear();

        if self.height == 0 {
            return;
        }

        let height = self.height as usize;
        if self.using_synchro {
            for row in 0..height {
                self.row_bin_positions.push(row as f32);
                self.row_lower_bins.push(row);
                self.row_upper_bins.push(row);
                self.row_interp_weights.push(0.0);
            }
            return;
        }

        if self.fft_size == 0 || self.sample_rate <= 0.0 {
            return;
        }

        let bin_count = self.fft_size / 2 + 1;
        if bin_count == 0 {
            return;
        }

        let nyquist = (self.sample_rate / 2.0).max(1.0);
        let mut min_freq = self.sample_rate / self.fft_size as f32;
        if min_freq < 20.0 {
            min_freq = 20.0;
        }

        let use_log_scale = min_freq < nyquist;
        let ratio = if use_log_scale {
            (nyquist / min_freq).max(1.0)
        } else {
            1.0
        };

        for row in 0..height {
            let normalized_top = if height <= 1 {
                0.0
            } else {
                row as f32 / (height as f32 - 1.0)
            };
            let frequency = if use_log_scale {
                let exponent = 1.0 - normalized_top;
                min_freq * ratio.powf(exponent)
            } else {
                let span = (bin_count - 1) as f32;
                let linear_bin = normalized_top * span;
                (linear_bin * self.sample_rate) / self.fft_size as f32
            };

            let mut bin_position = (frequency * self.fft_size as f32) / self.sample_rate;
            let max_bin = (bin_count - 1) as f32;
            if bin_position.is_nan() || !bin_position.is_finite() {
                bin_position = 0.0;
            }
            bin_position = bin_position.clamp(0.0, max_bin);
            self.row_bin_positions.push(bin_position);

            let lower = bin_position.floor().clamp(0.0, max_bin) as usize;
            let upper = (lower + 1).min(bin_count - 1);
            let frac = (bin_position - lower as f32).clamp(0.0, 1.0);
            self.row_lower_bins.push(lower);
            self.row_upper_bins.push(upper);
            self.row_interp_weights.push(frac);
        }
    }

    fn needs_row_sampling_refresh(&self) -> bool {
        let height = self.height as usize;
        self.row_lower_bins.len() != height
            || self.row_upper_bins.len() != height
            || self.row_interp_weights.len() != height
    }

    fn max_incremental_updates(&self) -> u32 {
        let capacity = self.capacity.max(1);
        let dynamic = capacity / 2;
        dynamic.max(MIN_INCREMENTAL_UPDATES)
    }

    fn ensure_row_mapping(&mut self, use_synchro: bool) {
        if !self.has_dimensions() {
            return;
        }

        let needs_refresh = self.using_synchro != use_synchro || self.needs_row_sampling_refresh();
        if needs_refresh {
            self.using_synchro = use_synchro;
            self.rebuild_row_positions();
        }
    }

    fn sync_grid_frequencies(&mut self, synchro_bins_hz: Option<&[f32]>) {
        match synchro_bins_hz {
            Some(freqs) if !freqs.is_empty() => {
                let limit = self.height.min(freqs.len() as u32) as usize;
                if self.grid_bin_frequencies.len() != limit {
                    self.grid_bin_frequencies.clear();
                    self.grid_bin_frequencies.extend_from_slice(&freqs[..limit]);
                } else {
                    self.grid_bin_frequencies[..limit].copy_from_slice(&freqs[..limit]);
                }
            }
            _ => {
                if !self.grid_bin_frequencies.is_empty() {
                    self.grid_bin_frequencies.clear();
                }
            }
        }
    }
}

#[cfg(test)]
fn sample_column(column: &[f32], bin_position: f32) -> f32 {
    if column.is_empty() {
        return 0.0;
    }

    let lower = bin_position.floor().clamp(0.0, (column.len() - 1) as f32) as usize;
    let upper = (lower + 1).min(column.len() - 1);
    if lower == upper {
        return column[lower];
    }

    let frac = (bin_position - lower as f32).clamp(0.0, 1.0);
    let lower_val = column[lower];
    let upper_val = column[upper];
    lower_val + (upper_val - lower_val) * frac
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn row_positions_follow_log_profile() {
        let mut buffer = SpectrogramBuffer::new();
        buffer.height = 8;
        buffer.sample_rate = DEFAULT_SAMPLE_RATE;
        buffer.fft_size = 2048;
        buffer.rebuild_row_positions();

        assert_eq!(buffer.row_bin_positions.len(), 8);
        let mut last = f32::INFINITY;
        for (idx, value) in buffer.row_bin_positions.iter().enumerate() {
            assert!(
                value.is_finite(),
                "row {idx} produced a non-finite bin index"
            );
            assert!(
                *value <= last + 1e-4,
                "row positions should be non-increasing"
            );
            last = *value;
        }

        let first = buffer.row_bin_positions.first().copied().unwrap();
        let last = buffer.row_bin_positions.last().copied().unwrap();
        assert!(first > last, "top row should map to higher-frequency bins");
    }

    #[test]
    fn sample_column_interpolates_between_bins() {
        let column = [0.0, -12.0, -24.0];
        let exact = sample_column(&column, 1.0);
        assert!((exact + 12.0).abs() < 1e-6);

        let midpoint = sample_column(&column, 1.5);
        assert!((midpoint + 18.0).abs() < 1e-6);
    }

    #[test]
    fn synchro_columns_use_log_grid_height() {
        let mut state = SpectrogramState::new();
        let column = SpectrogramColumn {
            timestamp: Instant::now(),
            magnitudes_db: Arc::from(vec![-96.0_f32; 8].into_boxed_slice()),
            reassigned: None,
            synchro_magnitudes_db: Some(Arc::from(
                vec![-30.0, -25.0, -20.0, -15.0].into_boxed_slice(),
            )),
        };

        let update = SpectrogramUpdate {
            fft_size: 1024,
            hop_size: 256,
            sample_rate: DEFAULT_SAMPLE_RATE,
            history_length: 4,
            reset: true,
            reassignment_enabled: true,
            synchro_bins_hz: Some(Arc::from(vec![40.0, 80.0, 160.0, 320.0].into_boxed_slice())),
            new_columns: vec![column],
        };

        state.apply_update(&update);

        let buffer = state.buffer.borrow();
        assert!(buffer.using_synchro());
        assert_eq!(buffer.texture_height(), 4);
    }

    #[test]
    fn synchro_columns_place_high_frequencies_on_top() {
        let mut state = SpectrogramState::new();
        let column = SpectrogramColumn {
            timestamp: Instant::now(),
            magnitudes_db: Arc::from(vec![-96.0_f32; 8].into_boxed_slice()),
            reassigned: None,
            synchro_magnitudes_db: Some(Arc::from(
                vec![-10.0, -20.0, -30.0, -40.0].into_boxed_slice(),
            )),
        };

        let update = SpectrogramUpdate {
            fft_size: 1024,
            hop_size: 256,
            sample_rate: DEFAULT_SAMPLE_RATE,
            history_length: 4,
            reset: true,
            reassignment_enabled: true,
            synchro_bins_hz: Some(Arc::from(
                vec![10_000.0, 2_500.0, 625.0, 156.25].into_boxed_slice(),
            )),
            new_columns: vec![column],
        };

        state.apply_update(&update);

        let buffer = state.buffer.borrow();
        assert!(buffer.using_synchro());
        assert_eq!(buffer.texture_height(), 4);
        assert_eq!(buffer.grid_bin_frequencies.len(), 4);
        let highest = buffer.grid_bin_frequencies.first().copied().unwrap();
        let lowest = buffer.grid_bin_frequencies.last().copied().unwrap();
        assert!(highest > lowest);

        let top = buffer.values.first().copied().unwrap_or_default();
        let bottom_index = buffer.texture_height() as usize - 1;
        let bottom = buffer.values.get(bottom_index).copied().unwrap_or_default();
        assert!(
            top > bottom,
            "top row should represent higher magnitudes (high frequencies)"
        );
    }

    #[test]
    fn clamps_height_when_bins_exceed_texture_limit() {
        let mut state = SpectrogramState::new();
        let oversized = MAX_TEXTURE_BINS + 5;
        let magnitudes = vec![-42.0_f32; oversized];

        let column = SpectrogramColumn {
            timestamp: Instant::now(),
            magnitudes_db: Arc::from(magnitudes.into_boxed_slice()),
            reassigned: None,
            synchro_magnitudes_db: None,
        };

        let update = SpectrogramUpdate {
            fft_size: 16_384,
            hop_size: 2_048,
            sample_rate: DEFAULT_SAMPLE_RATE,
            history_length: 8,
            reset: true,
            reassignment_enabled: false,
            synchro_bins_hz: None,
            new_columns: vec![column],
        };

        state.apply_update(&update);

        let buffer = state.buffer.borrow();
        assert_eq!(buffer.texture_height() as usize, MAX_TEXTURE_BINS);
        assert_eq!(buffer.column_count(), 1);
    }
}

fn build_palette_rgba(palette: &[Color; PALETTE_STOPS], opacity: f32) -> [[f32; 4]; PALETTE_STOPS] {
    let mut rgba = [[0.0; 4]; PALETTE_STOPS];
    for (idx, color) in palette.iter().enumerate() {
        rgba[idx] = color_to_rgba_with_opacity(*color, opacity);
    }
    rgba
}

/// Memoizes palette conversions so we avoid recomputing RGBA values unless
/// the style or palette actually change.
#[derive(Debug, Clone)]
struct PaletteCache {
    palette: [[f32; 4]; PALETTE_STOPS],
    background: [f32; 4],
    dirty: bool,
}

impl PaletteCache {
    fn new(style: &SpectrogramStyle, palette: &[Color; PALETTE_STOPS]) -> Self {
        let palette_rgba = build_palette_rgba(palette, style.opacity);
        let background = color_to_rgba_with_opacity(style.background, style.opacity);
        Self {
            palette: palette_rgba,
            background,
            dirty: false,
        }
    }

    fn refresh(&mut self, style: &SpectrogramStyle, palette: &[Color; PALETTE_STOPS]) {
        self.palette = build_palette_rgba(palette, style.opacity);
        self.background = color_to_rgba_with_opacity(style.background, style.opacity);
        self.dirty = false;
    }
}

/// widget wrapper that renders the spectrogram buffer
#[derive(Debug)]
pub struct Spectrogram<'a> {
    state: &'a SpectrogramState,
}

impl<'a> Spectrogram<'a> {
    pub fn new(state: &'a SpectrogramState) -> Self {
        Self { state }
    }
}

impl<'a, Message> Widget<Message, iced::Theme, iced::Renderer> for Spectrogram<'a> {
    fn tag(&self) -> tree::Tag {
        tree::Tag::stateless()
    }

    fn state(&self) -> tree::State {
        tree::State::new(())
    }

    fn size(&self) -> Size<Length> {
        Size::new(Length::Fill, Length::Fill)
    }

    fn layout(
        &self,
        _tree: &mut Tree,
        _renderer: &iced::Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        let size = limits.resolve(Length::Fill, Length::Fill, Size::new(0.0, 0.0));
        layout::Node::new(size)
    }

    fn draw(
        &self,
        _tree: &Tree,
        renderer: &mut iced::Renderer,
        _theme: &iced::Theme,
        _style: &renderer::Style,
        layout: Layout<'_>,
        _cursor: mouse::Cursor,
        _viewport: &Rectangle,
    ) {
        let bounds = layout.bounds();
        let background = self.state.style.background;
        renderer.fill_quad(
            Quad {
                bounds,
                border: Default::default(),
                shadow: Default::default(),
            },
            Background::Color(background),
        );

        if let Some(params) = self.state.visual_params(bounds) {
            renderer.draw_primitive(bounds, SpectrogramPrimitive::new(params));
        }
    }

    fn children(&self) -> Vec<Tree> {
        Vec::new()
    }

    fn diff(&self, _tree: &mut Tree) {}
}

pub fn widget<'a, Message>(state: &'a SpectrogramState) -> Element<'a, Message>
where
    Message: 'a,
{
    Element::new(Spectrogram::new(state))
}

fn color_to_rgba_with_opacity(color: Color, opacity: f32) -> [f32; 4] {
    let mut rgba = theme::color_to_rgba(color);
    rgba[3] = rgba[3].clamp(0.0, 1.0) * opacity.clamp(0.0, 1.0);
    rgba
}
