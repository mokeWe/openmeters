//! UI wrapper for the spectrogram DSP processor and renderer.

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

static NEXT_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

fn next_instance_id() -> u64 {
    NEXT_INSTANCE_ID.fetch_add(1, Ordering::Relaxed)
}

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
        let mut config = SpectrogramConfig::default();
        config.sample_rate = sample_rate;
        Self {
            inner: CoreSpectrogramProcessor::new(config),
            channels: DEFAULT_CHANNELS,
        }
    }

    pub fn ingest(&mut self, samples: &[f32]) -> ProcessorUpdate<SpectrogramUpdate> {
        if samples.is_empty() {
            return ProcessorUpdate::None;
        }

        let block = AudioBlock::new(
            samples,
            self.channels,
            self.inner.config().sample_rate,
            Instant::now(),
        );

        self.inner.process_block(&block)
    }

    #[allow(dead_code)]
    pub fn update_config(&mut self, config: SpectrogramConfig) {
        self.inner.update_config(config);
    }
}

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
    instance_id: u64,
}

impl SpectrogramState {
    pub fn new() -> Self {
        let style = SpectrogramStyle::default();
        let palette = default_palette();
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
            self.buffer.borrow_mut().rebuild_from_history(
                &self.history,
                &self.style,
                self.history_length,
                self.sample_rate,
                self.fft_size,
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
        let mut needs_rebuild = update.reset;

        if self.history_length != update.history_length {
            self.history_length = update.history_length;
            needs_rebuild = true;
        }

        if self.fft_size != update.fft_size || self.hop_size != update.hop_size {
            needs_rebuild = true;
        }

        if (self.sample_rate - update.sample_rate).abs() > f32::EPSILON {
            needs_rebuild = true;
        }

        if update.reset {
            self.history.clear();
            self.last_timestamp = None;
        }

        let incoming_height = update
            .new_columns
            .first()
            .map(|column| column.magnitudes_db.len());

        if let (Some(incoming), Some(existing)) = (
            incoming_height,
            self.history
                .front()
                .map(|column| column.magnitudes_db.len()),
        ) && incoming != existing
        {
            needs_rebuild = true;
        }

        for column in &update.new_columns {
            self.history.push_back(column.clone());
            if self.history.len() > self.history_length {
                self.history.pop_front();
            }
        }

        while self.history.len() > self.history_length {
            self.history.pop_front();
        }

        {
            let buffer = self.buffer.borrow();
            let desired_height = self
                .history
                .front()
                .map(|column| column.magnitudes_db.len() as u32)
                .unwrap_or(0);

            if desired_height > 0 && buffer.texture_height() != desired_height {
                needs_rebuild = true;
            }

            if buffer.capacity() != self.history_length as u32 {
                needs_rebuild = true;
            }

            if (buffer.sample_rate() - update.sample_rate).abs() > f32::EPSILON {
                needs_rebuild = true;
            }

            if buffer.fft_size() != update.fft_size {
                needs_rebuild = true;
            }
        }

        let mut buffer = self.buffer.borrow_mut();
        buffer.clear_pending();
        if needs_rebuild {
            buffer.rebuild_from_history(
                &self.history,
                &self.style,
                self.history_length,
                update.sample_rate,
                update.fft_size,
            );
        } else if !update.new_columns.is_empty() {
            buffer.append_columns(&update.new_columns, &self.style);
        }

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
    update_buffer_pool: ColumnBufferPool,
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

    fn latest_column(&self) -> u32 {
        if self.column_count == 0 || self.capacity == 0 {
            0
        } else {
            (self.write_index + self.capacity - 1) % self.capacity
        }
    }

    fn capacity(&self) -> u32 {
        self.capacity
    }

    fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    fn fft_size(&self) -> usize {
        self.fft_size
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

    fn mark_dirty(&mut self) {
        if self.values.is_empty() {
            self.clear_pending();
            return;
        }

        if self.pending_base.is_some() && self.pending_updates.is_empty() {
            return;
        }

        self.pending_base = Some(Arc::from(self.values.clone().into_boxed_slice()));
        self.pending_updates.clear();
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
        self.update_buffer_pool = ColumnBufferPool::new();
    }

    fn rebuild_from_history(
        &mut self,
        history: &VecDeque<SpectrogramColumn>,
        style: &SpectrogramStyle,
        history_length: usize,
        sample_rate: f32,
        fft_size: usize,
    ) {
        self.clear_pending();

        self.capacity = history_length as u32;
        self.sample_rate = if sample_rate <= 0.0 {
            DEFAULT_SAMPLE_RATE
        } else {
            sample_rate
        };
        self.fft_size = fft_size.max(1);

        let height = history
            .front()
            .map(|column| column.magnitudes_db.len() as u32)
            .unwrap_or(0);
        self.height = height;

        if self.capacity == 0 || self.height == 0 {
            self.values.clear();
            self.write_index = 0;
            self.column_count = 0;
            self.last_timestamp = history.back().map(|column| column.timestamp);
            self.row_bin_positions.clear();
            self.row_lower_bins.clear();
            self.row_upper_bins.clear();
            self.row_interp_weights.clear();
            self.pending_base = None;
            self.pending_updates.clear();
            return;
        }

        self.values
            .resize(self.capacity as usize * self.height as usize, 0.0);
        self.write_index = 0;
        self.column_count = 0;
        self.last_timestamp = None;
        self.rebuild_row_positions();

        for column in history.iter() {
            self.push_column(column.magnitudes_db.as_ref(), style);
        }

        if self.column_count > 0 {
            self.pending_base = Some(Arc::from(self.values.clone().into_boxed_slice()));
        } else {
            self.pending_base = None;
        }
        self.pending_updates.clear();
        self.last_timestamp = history.back().map(|column| column.timestamp);
    }

    fn append_columns(&mut self, columns: &[SpectrogramColumn], style: &SpectrogramStyle) {
        if self.capacity == 0 || self.height == 0 {
            return;
        }

        if self.needs_row_sampling_refresh() {
            self.rebuild_row_positions();
        }

        for column in columns {
            if column.magnitudes_db.len() != self.height as usize {
                continue;
            }

            let physical_index = self.push_column(column.magnitudes_db.as_ref(), style);
            let start = (physical_index as usize) * (self.height as usize);
            let mut buffer = self.update_buffer_pool.acquire(self.height as usize);
            buffer.copy_from_slice(&self.values[start..start + self.height as usize]);
            let values = Arc::new(ColumnBuffer::new(
                buffer.into_boxed_slice(),
                self.update_buffer_pool.clone(),
            ));
            self.pending_updates.push(SpectrogramColumnUpdate {
                column_index: physical_index,
                values,
            });
        }

        if self.pending_updates.len() as u32 >= self.max_incremental_updates() {
            self.pending_base = Some(Arc::from(self.values.clone().into_boxed_slice()));
            self.pending_updates.clear();
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

        if self.height == 0 || self.fft_size == 0 || self.sample_rate <= 0.0 {
            return;
        }

        let height = self.height as usize;
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
}

fn default_palette() -> [Color; PALETTE_STOPS] {
    [
        Color::from_rgba(0.05, 0.08, 0.18, 0.0),
        Color::from_rgba(0.13, 0.20, 0.46, 1.0),
        Color::from_rgba(0.11, 0.48, 0.63, 1.0),
        Color::from_rgba(0.94, 0.75, 0.29, 1.0),
        Color::from_rgba(0.98, 0.93, 0.65, 1.0),
    ]
}

fn build_palette_rgba(palette: &[Color; PALETTE_STOPS], opacity: f32) -> [[f32; 4]; PALETTE_STOPS] {
    let mut rgba = [[0.0; 4]; PALETTE_STOPS];
    for (idx, color) in palette.iter().enumerate() {
        rgba[idx] = color_to_rgba_with_opacity(*color, opacity);
    }
    rgba
}

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
