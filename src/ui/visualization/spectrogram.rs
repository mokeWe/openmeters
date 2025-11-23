//! UI wrapper for the spectrogram DSP processor and renderer.

use crate::audio::meter_tap::MeterFormat;
use crate::dsp::spectrogram::{
    FrequencyScale, SpectrogramColumn, SpectrogramConfig,
    SpectrogramProcessor as CoreSpectrogramProcessor, SpectrogramUpdate, hz_to_mel, mel_to_hz,
};
use crate::dsp::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::ui::render::spectrogram::{
    ColumnBuffer, ColumnBufferPool, SPECTROGRAM_PALETTE_SIZE, SpectrogramColumnUpdate,
    SpectrogramParams, SpectrogramPrimitive,
};
use crate::ui::theme;
use crate::util::audio::DEFAULT_SAMPLE_RATE;
use crate::util::audio::musical::MusicalNote;
use iced::advanced::Renderer as _;
use iced::advanced::renderer::{self, Quad};
use iced::advanced::text::Renderer as TextRenderer;
use iced::advanced::widget::{Tree, tree};
use iced::advanced::{Layout, Widget, layout, mouse};
use iced::{Background, Color, Element, Length, Point, Rectangle, Size};
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

const TOOLTIP_TEXT_SIZE: f32 = 14.0;
const TOOLTIP_PADDING: f32 = 8.0;
const TOOLTIP_OFFSET: f32 = 12.0;
const TOOLTIP_SHADOW_OFFSET: f32 = 1.0;
const TOOLTIP_SHADOW_OPACITY: f32 = 0.3;

const MAX_TEXTURE_BINS: usize = 8_192;

static NEXT_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

fn next_instance_id() -> u64 {
    NEXT_INSTANCE_ID.fetch_add(1, Ordering::Relaxed)
}

pub struct SpectrogramProcessor {
    inner: CoreSpectrogramProcessor,
    channels: usize,
    cached_sample_rate: f32,
}

impl fmt::Debug for SpectrogramProcessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpectrogramProcessor")
            .field("channels", &self.channels)
            .field("cached_sample_rate", &self.cached_sample_rate)
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
            cached_sample_rate: sample_rate,
        }
    }

    #[inline]
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
        if (self.cached_sample_rate - sample_rate).abs() > f32::EPSILON {
            self.cached_sample_rate = sample_rate;
            let mut config = self.inner.config();
            config.sample_rate = sample_rate;
            self.inner.update_config(config);
        }

        let block = AudioBlock {
            samples,
            channels: self.channels,
            sample_rate: self.cached_sample_rate,
            timestamp: Instant::now(),
        };

        self.inner.process_block(&block)
    }

    pub fn update_config(&mut self, config: SpectrogramConfig) {
        self.cached_sample_rate = config.sample_rate;
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
    history: VecDeque<SpectrogramColumn>,
    fft_size: usize,
    hop_size: usize,
    sample_rate: f32,
    frequency_scale: FrequencyScale,
    history_length: usize,
    synchro_bins_hz: Option<Arc<[f32]>>,
    instance_id: u64,
}

impl SpectrogramState {
    pub fn new() -> Self {
        let style = SpectrogramStyle::default();
        let palette = theme::DEFAULT_SPECTROGRAM_PALETTE;
        let default_cfg = SpectrogramConfig::default();
        let palette_cache = RefCell::new(PaletteCache::new(&style, &palette));

        Self {
            buffer: RefCell::new(SpectrogramBuffer::new()),
            style,
            palette,
            palette_cache,
            history: VecDeque::new(),
            fft_size: default_cfg.fft_size,
            hop_size: default_cfg.hop_size,
            sample_rate: default_cfg.sample_rate,
            frequency_scale: default_cfg.frequency_scale,
            history_length: default_cfg.history_length,
            synchro_bins_hz: None,
            instance_id: next_instance_id(),
        }
    }

    pub fn set_palette(&mut self, palette: [Color; PALETTE_STOPS]) {
        if self.palette == palette {
            return;
        }

        self.palette = palette;
        self.palette_cache
            .borrow_mut()
            .refresh(&self.style, &palette);
        self.buffer.borrow_mut().mark_dirty();
    }

    pub fn palette(&self) -> [Color; PALETTE_STOPS] {
        self.palette
    }

    pub fn apply_update(&mut self, update: &SpectrogramUpdate) {
        if update.new_columns.is_empty() && !update.reset {
            return;
        }

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

        let synchro_bins = self.synchro_bins_hz.as_deref();

        let desired_height = Self::history_height(&self.history, use_synchro)
            .map(|len| len as u32)
            .unwrap_or(0);

        let mut buffer = self.buffer.borrow_mut();

        if !needs_rebuild {
            needs_rebuild = buffer.requires_rebuild(update, use_synchro, desired_height);
        }

        buffer.clear_pending();

        if needs_rebuild {
            buffer.rebuild_from_history(
                &self.history,
                RebuildContext {
                    style: &self.style,
                    history_length: update.history_length,
                    sample_rate: update.sample_rate,
                    fft_size: update.fft_size,
                    frequency_scale: update.frequency_scale,
                    use_synchro,
                    synchro_bins_hz: synchro_bins,
                },
            );
        } else if !update.new_columns.is_empty() {
            buffer.append_columns(&update.new_columns, &self.style, use_synchro, synchro_bins);
        }

        // Update cached state (after dropping buffer borrow)
        drop(buffer);

        self.history_length = update.history_length;
        self.fft_size = update.fft_size;
        self.hop_size = update.hop_size;
        self.sample_rate = update.sample_rate;
        self.frequency_scale = update.frequency_scale;
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
        let cache = self.palette_cache.borrow();
        (cache.palette, cache.background)
    }

    fn compute_synchro_state(
        columns: &[SpectrogramColumn],
        bins: &Option<Arc<[f32]>>,
    ) -> (bool, Option<Arc<[f32]>>) {
        let bins = match bins.as_ref() {
            Some(b) if !b.is_empty() => b,
            _ => return (false, None),
        };

        let compatible = columns.iter().all(|col| {
            col.synchro_magnitudes_db
                .as_ref()
                .is_some_and(|v| v.len() == bins.len())
        });

        if compatible {
            (true, Some(Arc::clone(bins)))
        } else {
            (false, None)
        }
    }

    fn column_height(column: &SpectrogramColumn, use_synchro: bool) -> Option<usize> {
        SpectrogramBuffer::column_values(column, use_synchro)
            .map(|values| values.len().min(MAX_TEXTURE_BINS))
    }

    fn first_height(
        columns: impl IntoIterator<Item = impl std::borrow::Borrow<SpectrogramColumn>>,
        use_synchro: bool,
    ) -> Option<usize> {
        columns
            .into_iter()
            .find_map(|col| Self::column_height(col.borrow(), use_synchro))
    }

    fn history_height(history: &VecDeque<SpectrogramColumn>, use_synchro: bool) -> Option<usize> {
        Self::first_height(history, use_synchro)
    }

    fn pending_height(columns: &[SpectrogramColumn], use_synchro: bool) -> Option<usize> {
        Self::first_height(columns, use_synchro)
    }

    fn push_history(
        history: &mut VecDeque<SpectrogramColumn>,
        columns: &[SpectrogramColumn],
        capacity: usize,
    ) {
        if capacity == 0 {
            history.clear();
        } else if !columns.is_empty() {
            history.extend(columns.iter().cloned());
            if history.len() > capacity {
                history.drain(0..history.len() - capacity);
            }
        }
    }
}

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

#[derive(Debug, Clone)]
struct SpectrogramBuffer {
    capacity: u32,
    height: u32,
    values: Vec<f32>,
    write_index: u32,
    column_count: u32,
    pending_base: Option<Arc<Vec<f32>>>,
    pending_updates: Arc<Vec<SpectrogramColumnUpdate>>,
    sample_rate: f32,
    fft_size: usize,
    frequency_scale: FrequencyScale,
    row_bin_positions: Arc<Vec<f32>>,
    row_lower_bins: Arc<Vec<usize>>,
    row_upper_bins: Arc<Vec<usize>>,
    row_interp_weights: Arc<Vec<f32>>,
    using_synchro: bool,
    grid_bin_frequencies: Arc<Vec<f32>>,
    update_buffer_pool: ColumnBufferPool,
}

struct RebuildContext<'a> {
    style: &'a SpectrogramStyle,
    history_length: usize,
    sample_rate: f32,
    fft_size: usize,
    frequency_scale: FrequencyScale,
    use_synchro: bool,
    synchro_bins_hz: Option<&'a [f32]>,
}

impl SpectrogramBuffer {
    fn new() -> Self {
        Self {
            capacity: 0,
            height: 0,
            values: Vec::new(),
            write_index: 0,
            column_count: 0,
            pending_base: None,
            pending_updates: Arc::new(Vec::new()),
            sample_rate: DEFAULT_SAMPLE_RATE,
            fft_size: 0,
            frequency_scale: FrequencyScale::default(),
            row_bin_positions: Arc::new(Vec::new()),
            row_lower_bins: Arc::new(Vec::new()),
            row_upper_bins: Arc::new(Vec::new()),
            row_interp_weights: Arc::new(Vec::new()),
            using_synchro: false,
            grid_bin_frequencies: Arc::new(Vec::new()),
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

    #[cfg(test)]
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
        if self.column_count == 0 {
            0
        } else {
            (self.write_index + self.capacity - 1) % self.capacity.max(1)
        }
    }

    fn take_base_data(&mut self) -> Option<Arc<Vec<f32>>> {
        self.pending_base.take()
    }

    fn take_updates(&mut self) -> Vec<SpectrogramColumnUpdate> {
        if self.pending_updates.is_empty() {
            Vec::new()
        } else {
            let updates = Arc::make_mut(&mut self.pending_updates);
            std::mem::take(updates)
        }
    }

    fn clear_pending(&mut self) {
        self.pending_base = None;
        if !self.pending_updates.is_empty() {
            self.pending_updates = Arc::new(Vec::new());
        }
    }

    fn queue_full_refresh(&mut self) {
        self.clear_pending();
        if !self.values.is_empty() {
            self.pending_base = Some(Arc::new(self.values.clone()));
        }
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
        // If palette/style changes, refresh entire buffer if non-empty
        if !self.values.is_empty()
            && (self.pending_base.is_none() || !self.pending_updates.is_empty())
        {
            self.queue_full_refresh();
        } else {
            self.clear_pending();
        }
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
            frequency_scale,
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
        self.frequency_scale = frequency_scale;
        self.using_synchro = use_synchro;

        self.height = history
            .iter()
            .find_map(|column| {
                SpectrogramBuffer::column_values(column, use_synchro)
                    .map(|values| values.len().min(MAX_TEXTURE_BINS) as u32)
            })
            .unwrap_or(0);

        if !self.has_dimensions() {
            self.values = Vec::new();
            self.write_index = 0;
            self.column_count = 0;
            self.row_bin_positions = Arc::new(Vec::new());
            self.row_lower_bins = Arc::new(Vec::new());
            self.row_upper_bins = Arc::new(Vec::new());
            self.row_interp_weights = Arc::new(Vec::new());
            self.using_synchro = false;
            self.grid_bin_frequencies = Arc::new(Vec::new());
            self.clear_pending();
            return;
        }

        self.sync_grid_frequencies(synchro_bins_hz);

        self.values = Vec::with_capacity(self.capacity as usize * self.height as usize);
        self.values
            .resize(self.capacity as usize * self.height as usize, 0.0);
        self.write_index = 0;
        self.column_count = 0;
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
            Arc::make_mut(&mut self.pending_updates).push(SpectrogramColumnUpdate {
                column_index: physical_index,
                values,
            });
        }

        if self.pending_updates.len() as u32 >= self.max_incremental_updates() {
            self.queue_full_refresh();
        }
    }

    fn push_column(&mut self, magnitudes_db: &[f32], style: &SpectrogramStyle) -> u32 {
        if !self.has_dimensions() || magnitudes_db.is_empty() {
            return 0;
        }

        let height = self.height as usize;
        let column = self.write_index;
        let start = (column as usize) * height;

        if self.needs_row_sampling_refresh() {
            self.rebuild_row_positions();
        }

        if self.values.len() != (self.capacity as usize * height) {
            self.values.resize(self.capacity as usize * height, 0.0);
        }

        let inv_range = 1.0 / (style.ceiling_db - style.floor_db).max(f32::EPSILON);
        let floor_db = style.floor_db;
        let ceiling_db = style.ceiling_db;
        let bin_count = magnitudes_db.len();
        let bin_count_minus_1 = bin_count.saturating_sub(1);

        let lower_bins = self.row_lower_bins.as_slice();
        let upper_bins = self.row_upper_bins.as_slice();
        let interp_weights = self.row_interp_weights.as_slice();
        let output_slice = &mut self.values[start..start + height];

        for row in 0..height {
            let lower_idx = lower_bins[row].min(bin_count_minus_1);
            let upper_idx = upper_bins[row].min(bin_count_minus_1);
            let frac = interp_weights[row];

            let lower_val = magnitudes_db[lower_idx];
            let upper_val = magnitudes_db[upper_idx];
            let sample = frac.mul_add(upper_val - lower_val, lower_val);

            let normalized =
                ((sample.clamp(floor_db, ceiling_db) - floor_db) * inv_range).clamp(0.0, 1.0);

            output_slice[row] = normalized;
        }

        if self.column_count < self.capacity {
            self.column_count += 1;
        }
        self.write_index = (self.write_index + 1) % self.capacity.max(1);
        column
    }

    fn rebuild_row_positions(&mut self) {
        let height = self.height as usize;
        if height == 0 {
            self.row_bin_positions = Arc::new(Vec::new());
            self.row_lower_bins = Arc::new(Vec::new());
            self.row_upper_bins = Arc::new(Vec::new());
            self.row_interp_weights = Arc::new(Vec::new());
            return;
        }

        let mut positions = Vec::with_capacity(height);
        let mut lower_bins = Vec::with_capacity(height);
        let mut upper_bins = Vec::with_capacity(height);
        let mut weights = Vec::with_capacity(height);

        // Synchrosqueezed bins are already in display order, use identity mapping
        if self.using_synchro {
            positions.extend((0..height).map(|i| i as f32));
            lower_bins.extend(0..height);
            upper_bins.extend(0..height);
            weights.resize(height, 0.0);

            self.row_bin_positions = Arc::new(positions);
            self.row_lower_bins = Arc::new(lower_bins);
            self.row_upper_bins = Arc::new(upper_bins);
            self.row_interp_weights = Arc::new(weights);
            return;
        }

        let bin_count = self.fft_size / 2 + 1;
        if bin_count == 0 || self.sample_rate <= 0.0 {
            self.row_bin_positions = Arc::new(Vec::new());
            self.row_lower_bins = Arc::new(Vec::new());
            self.row_upper_bins = Arc::new(Vec::new());
            self.row_interp_weights = Arc::new(Vec::new());
            return;
        }

        let max_bin = (bin_count - 1) as f32;
        let height_denom = (height - 1).max(1) as f32;

        for row in 0..height {
            let normalized_y = row as f32 / height_denom;
            let frequency = calculate_frequency(
                normalized_y,
                self.sample_rate,
                self.fft_size,
                self.frequency_scale,
            );
            let bin_position =
                ((frequency * self.fft_size as f32) / self.sample_rate).clamp(0.0, max_bin);

            positions.push(bin_position);
            let lower = bin_position.floor() as usize;
            let upper = (lower + 1).min(bin_count - 1);
            lower_bins.push(lower);
            upper_bins.push(upper);
            weights.push((bin_position - lower as f32).clamp(0.0, 1.0));
        }

        self.row_bin_positions = Arc::new(positions);
        self.row_lower_bins = Arc::new(lower_bins);
        self.row_upper_bins = Arc::new(upper_bins);
        self.row_interp_weights = Arc::new(weights);
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
                let mut new_freqs = Vec::with_capacity(limit);
                new_freqs.extend_from_slice(&freqs[..limit]);
                self.grid_bin_frequencies = Arc::new(new_freqs);
            }
            _ => {
                if !self.grid_bin_frequencies.is_empty() {
                    self.grid_bin_frequencies = Arc::new(Vec::new());
                }
            }
        }
    }
}

fn calculate_frequency(
    normalized_y: f32,
    sample_rate: f32,
    fft_size: usize,
    scale: FrequencyScale,
) -> f32 {
    let nyquist = (sample_rate / 2.0).max(1.0);
    let inv_y = 1.0 - normalized_y;

    match scale {
        FrequencyScale::Linear => nyquist * inv_y,
        FrequencyScale::Logarithmic | FrequencyScale::Mel => {
            let min_freq = (sample_rate / fft_size as f32).max(20.0);

            if matches!(scale, FrequencyScale::Logarithmic) {
                let ratio = (nyquist / min_freq).max(1.0);
                min_freq * ratio.powf(inv_y)
            } else {
                let min_mel = hz_to_mel(min_freq);
                let max_mel = hz_to_mel(nyquist);
                mel_to_hz(min_mel + (max_mel - min_mel) * inv_y)
            }
        }
    }
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
        let column = [0.0_f32, -12.0, -24.0];

        // Test exact bin position
        let exact = column[1];
        assert!((exact + 12.0).abs() < 1e-6);

        // Test interpolation at midpoint
        let midpoint = column[1] + (column[2] - column[1]) * 0.5;
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
            frequency_scale: FrequencyScale::default(),
            history_length: 4,
            reset: true,
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
            frequency_scale: FrequencyScale::default(),
            history_length: 4,
            reset: true,
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
            frequency_scale: FrequencyScale::default(),
            history_length: 8,
            reset: true,
            synchro_bins_hz: None,
            new_columns: vec![column],
        };

        state.apply_update(&update);

        let buffer = state.buffer.borrow();
        assert_eq!(buffer.texture_height() as usize, MAX_TEXTURE_BINS);
        assert_eq!(buffer.column_count(), 1);
    }
}

/// Memoizes palette conversions so we avoid recomputing RGBA values unless
/// the style or palette actually change.
#[derive(Debug, Clone)]
struct PaletteCache {
    palette: [[f32; 4]; PALETTE_STOPS],
    background: [f32; 4],
}

impl PaletteCache {
    fn new(style: &SpectrogramStyle, palette: &[Color; PALETTE_STOPS]) -> Self {
        Self {
            palette: Self::convert_palette(palette, style.opacity),
            background: theme::color_to_rgba_with_opacity(style.background, style.opacity),
        }
    }

    fn refresh(&mut self, style: &SpectrogramStyle, palette: &[Color; PALETTE_STOPS]) {
        self.palette = Self::convert_palette(palette, style.opacity);
        self.background = theme::color_to_rgba_with_opacity(style.background, style.opacity);
    }

    fn convert_palette(
        palette: &[Color; PALETTE_STOPS],
        opacity: f32,
    ) -> [[f32; 4]; PALETTE_STOPS] {
        palette.map(|color| theme::color_to_rgba_with_opacity(color, opacity))
    }
}

#[derive(Debug, Clone, Default)]
struct TooltipState {
    cursor: Option<Point>,
}

/// widget wrapper that renders the spectrogram buffer
#[derive(Debug)]
pub struct Spectrogram<'a> {
    state: &'a RefCell<SpectrogramState>,
}

impl<'a> Spectrogram<'a> {
    pub fn new(state: &'a RefCell<SpectrogramState>) -> Self {
        Self { state }
    }

    /// Calculate the frequency at a given Y position in the bounds.
    fn frequency_at_y(&self, y: f32, bounds: Rectangle) -> Option<f32> {
        if bounds.height <= 0.0 || !bounds.contains(Point::new(bounds.x, y)) {
            return None;
        }

        let normalized_y = (y - bounds.y) / bounds.height;
        let state = self.state.borrow();

        // If synchrosqueezed bins are active, directly look up frequency
        if let Some(bins) = state.synchro_bins_hz.as_deref() {
            let bin_index = (normalized_y * bins.len() as f32).floor() as usize;
            return bins.get(bin_index).copied();
        }

        if state.fft_size == 0 || state.sample_rate <= 0.0 {
            return None;
        }

        let frequency = calculate_frequency(
            normalized_y,
            state.sample_rate,
            state.fft_size,
            state.frequency_scale,
        );

        (frequency.is_finite() && frequency > 0.0).then_some(frequency)
    }

    /// Draw the frequency tooltip near the cursor.
    fn draw_tooltip(
        &self,
        renderer: &mut iced::Renderer,
        theme: &iced::Theme,
        bounds: Rectangle,
        cursor_pos: Point,
    ) {
        use iced::advanced::graphics::text::Paragraph as RenderParagraph;
        use iced::advanced::text::{self, Paragraph as _};

        let Some(freq_hz) = self.frequency_at_y(cursor_pos.y, bounds) else {
            return;
        };

        let tooltip_text = match MusicalNote::from_frequency(freq_hz) {
            Some(note) => format!("{:.1} Hz | {}", freq_hz, note.format()),
            None => format!("{:.1} Hz", freq_hz),
        };

        let text_size = RenderParagraph::with_text(text::Text {
            content: &tooltip_text,
            bounds: Size::INFINITY,
            size: iced::Pixels(TOOLTIP_TEXT_SIZE),
            line_height: text::LineHeight::default(),
            font: iced::Font::default(),
            horizontal_alignment: iced::alignment::Horizontal::Left,
            vertical_alignment: iced::alignment::Vertical::Top,
            shaping: text::Shaping::Basic,
            wrapping: text::Wrapping::None,
        })
        .min_bounds();

        let tooltip_size = Size::new(
            text_size.width + TOOLTIP_PADDING * 2.0,
            text_size.height + TOOLTIP_PADDING * 2.0,
        );

        let tooltip_x =
            if cursor_pos.x + TOOLTIP_OFFSET + tooltip_size.width <= bounds.x + bounds.width {
                cursor_pos.x + TOOLTIP_OFFSET
            } else {
                (cursor_pos.x - TOOLTIP_OFFSET - tooltip_size.width).max(bounds.x)
            };

        let tooltip_y = (cursor_pos.y - tooltip_size.height * 0.5)
            .clamp(bounds.y, bounds.y + bounds.height - tooltip_size.height);

        let tooltip_bounds = Rectangle::new(Point::new(tooltip_x, tooltip_y), tooltip_size);

        let palette = theme.extended_palette();

        // Shadow
        renderer.fill_quad(
            Quad {
                bounds: Rectangle::new(
                    Point::new(
                        tooltip_bounds.x + TOOLTIP_SHADOW_OFFSET,
                        tooltip_bounds.y + TOOLTIP_SHADOW_OFFSET,
                    ),
                    tooltip_size,
                ),
                border: Default::default(),
                shadow: Default::default(),
            },
            Background::Color(theme::with_alpha(
                palette.background.base.color,
                TOOLTIP_SHADOW_OPACITY,
            )),
        );

        // Background
        renderer.fill_quad(
            Quad {
                bounds: tooltip_bounds,
                border: theme::sharp_border(),
                shadow: Default::default(),
            },
            Background::Color(palette.background.strong.color),
        );

        // Text
        let text_origin = Point::new(
            tooltip_bounds.x + TOOLTIP_PADDING,
            tooltip_bounds.y + TOOLTIP_PADDING,
        );
        renderer.fill_text(
            text::Text {
                content: tooltip_text,
                bounds: Size::new(text_size.width, text_size.height),
                size: iced::Pixels(TOOLTIP_TEXT_SIZE),
                font: iced::Font::default(),
                horizontal_alignment: iced::alignment::Horizontal::Left,
                vertical_alignment: iced::alignment::Vertical::Top,
                line_height: text::LineHeight::default(),
                shaping: text::Shaping::Basic,
                wrapping: text::Wrapping::None,
            },
            text_origin,
            palette.background.base.text,
            Rectangle::new(text_origin, text_size),
        );
    }
}

impl<'a, Message> Widget<Message, iced::Theme, iced::Renderer> for Spectrogram<'a> {
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<TooltipState>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(TooltipState::default())
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

    fn on_event(
        &mut self,
        tree: &mut Tree,
        event: iced::Event,
        layout: Layout<'_>,
        _cursor: mouse::Cursor,
        _renderer: &iced::Renderer,
        _clipboard: &mut dyn iced::advanced::Clipboard,
        _shell: &mut iced::advanced::Shell<'_, Message>,
        _viewport: &Rectangle,
    ) -> iced::advanced::graphics::core::event::Status {
        let state = tree.state.downcast_mut::<TooltipState>();
        let bounds = layout.bounds();

        match event {
            iced::Event::Mouse(mouse::Event::CursorMoved { position }) => {
                state.cursor = bounds.contains(position).then_some(position);
            }
            iced::Event::Mouse(mouse::Event::CursorLeft) => {
                state.cursor = None;
            }
            _ => {}
        }

        iced::advanced::graphics::core::event::Status::Ignored
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut iced::Renderer,
        theme: &iced::Theme,
        _style: &renderer::Style,
        layout: Layout<'_>,
        _cursor: mouse::Cursor,
        _viewport: &Rectangle,
    ) {
        let bounds = layout.bounds();
        let state = self.state.borrow();

        renderer.fill_quad(
            Quad {
                bounds,
                border: Default::default(),
                shadow: Default::default(),
            },
            Background::Color(state.style.background),
        );

        if let Some(params) = state.visual_params(bounds) {
            renderer.draw_primitive(bounds, SpectrogramPrimitive::new(params));
        }

        // Draw tooltip if cursor is over the widget
        let tooltip_state = tree.state.downcast_ref::<TooltipState>();
        if let Some(cursor_pos) = tooltip_state.cursor
            && bounds.contains(cursor_pos)
        {
            renderer.with_layer(bounds, |renderer| {
                self.draw_tooltip(renderer, theme, bounds, cursor_pos);
            });
        }
    }

    fn mouse_interaction(
        &self,
        _tree: &Tree,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        _viewport: &Rectangle,
        _renderer: &iced::Renderer,
    ) -> mouse::Interaction {
        if cursor.is_over(layout.bounds()) {
            mouse::Interaction::Crosshair
        } else {
            mouse::Interaction::default()
        }
    }
}

pub fn widget<'a, Message>(state: &'a RefCell<SpectrogramState>) -> Element<'a, Message>
where
    Message: 'a,
{
    Element::new(Spectrogram::new(state))
}
