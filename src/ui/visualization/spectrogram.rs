use crate::audio::meter_tap::MeterFormat;
use crate::dsp::spectrogram::{
    FrequencyScale, SpectrogramColumn, SpectrogramConfig,
    SpectrogramProcessor as CoreSpectrogramProcessor, SpectrogramUpdate,
};
use crate::dsp::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::ui::render::spectrogram::{
    ColumnBuffer, ColumnBufferPool, SPECTROGRAM_PALETTE_SIZE, SpectrogramColumnUpdate,
    SpectrogramParams, SpectrogramPrimitive,
};
use crate::ui::theme;
use crate::util::audio::musical::MusicalNote;
use crate::util::audio::{DEFAULT_SAMPLE_RATE, hz_to_mel, mel_to_hz};
use iced::advanced::graphics::text::Paragraph;
use iced::advanced::renderer::{self, Quad};
use iced::advanced::text::{self, Paragraph as _, Renderer as TextRenderer};
use iced::advanced::widget::{Tree, tree};
use iced::advanced::{Layout, Renderer as _, Widget, layout, mouse};
use iced::{Background, Color, Element, Length, Point, Rectangle, Size};
use iced_wgpu::primitive::Renderer as _;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::Instant;

const DB_FLOOR: f32 = -120.0;
const DB_CEILING: f32 = 0.0;
const MAX_TEXTURE_BINS: usize = 8_192;
const TOOLTIP_SIZE: f32 = 14.0;
const TOOLTIP_PAD: f32 = 8.0;

fn norm_to_freq(inv: f32, nyquist: f32, min_freq: f32, scale: FrequencyScale) -> f32 {
    match scale {
        FrequencyScale::Linear => nyquist * inv,
        FrequencyScale::Logarithmic => min_freq * (nyquist / min_freq).max(1.0).powf(inv),
        FrequencyScale::Mel => {
            mel_to_hz(hz_to_mel(min_freq) + (hz_to_mel(nyquist) - hz_to_mel(min_freq)) * inv)
        }
    }
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

pub struct SpectrogramProcessor {
    inner: CoreSpectrogramProcessor,
    channels: usize,
    sample_rate: f32,
}

impl SpectrogramProcessor {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            inner: CoreSpectrogramProcessor::new(SpectrogramConfig {
                sample_rate,
                use_reassignment: true,
                ..Default::default()
            }),
            channels: 2,
            sample_rate,
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
        self.channels = format.channels.max(1);
        let rate = format.sample_rate.max(1.0);
        if (self.sample_rate - rate).abs() > f32::EPSILON {
            self.sample_rate = rate;
            let mut cfg = self.inner.config();
            cfg.sample_rate = rate;
            self.inner.update_config(cfg);
        }
        self.inner.process_block(&AudioBlock {
            samples,
            channels: self.channels,
            sample_rate: self.sample_rate,
            timestamp: Instant::now(),
        })
    }

    pub fn update_config(&mut self, config: SpectrogramConfig) {
        self.sample_rate = config.sample_rate;
        self.inner.update_config(config);
    }

    pub fn config(&self) -> SpectrogramConfig {
        self.inner.config()
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
            background: theme::with_alpha(theme::BG_BASE, 0.0),
            floor_db: DB_FLOOR,
            ceiling_db: DB_CEILING,
            opacity: 0.95,
            contrast: 1.4,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct BinMapping {
    lower: Vec<usize>,
    upper: Vec<usize>,
    weight: Vec<f32>,
}

impl BinMapping {
    fn new(
        height: usize,
        fft_size: usize,
        sample_rate: f32,
        scale: FrequencyScale,
        passthrough: bool,
    ) -> Self {
        if height == 0 {
            return Self::default();
        }
        if passthrough {
            let idx: Vec<usize> = (0..height).collect();
            return Self {
                lower: idx.clone(),
                upper: idx,
                weight: vec![0.0; height],
            };
        }
        let (bins, max_bin, denom) = (
            fft_size / 2 + 1,
            (fft_size / 2) as f32,
            (height - 1).max(1) as f32,
        );
        let (nyq, min_f) = (
            (sample_rate / 2.0).max(1.0),
            (sample_rate / fft_size as f32).max(20.0),
        );
        let (mut lower, mut upper, mut weight) = (
            Vec::with_capacity(height),
            Vec::with_capacity(height),
            Vec::with_capacity(height),
        );
        for row in 0..height {
            let pos = ((norm_to_freq(1.0 - row as f32 / denom, nyq, min_f, scale)
                * fft_size as f32)
                / sample_rate)
                .clamp(0.0, max_bin);
            let lo = pos.floor() as usize;
            lower.push(lo);
            upper.push((lo + 1).min(bins - 1));
            weight.push(pos - lo as f32);
        }
        Self {
            lower,
            upper,
            weight,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct SpectrogramBuffer {
    values: Vec<f32>,
    capacity: u32,
    height: u32,
    write_idx: u32,
    col_count: u32,
    pending_base: Option<Arc<Vec<f32>>>,
    pending_cols: Vec<SpectrogramColumnUpdate>,
    mapping: BinMapping,
    #[allow(clippy::default_trait_access)]
    sample_rate: f32,
    fft_size: usize,
    scale: FrequencyScale,
    display_freqs: Arc<[f32]>,
    pool: ColumnBufferPool,
}

impl SpectrogramBuffer {
    fn new() -> Self {
        Self {
            sample_rate: DEFAULT_SAMPLE_RATE,
            display_freqs: Arc::from([]),
            ..Default::default()
        }
    }

    fn rebuild(
        &mut self,
        history: &VecDeque<SpectrogramColumn>,
        upd: &SpectrogramUpdate,
        style: &SpectrogramStyle,
    ) {
        (self.pending_base, self.pending_cols) = (None, Vec::new());
        (self.capacity, self.sample_rate, self.fft_size, self.scale) = (
            upd.history_length as u32,
            upd.sample_rate.max(1.0),
            upd.fft_size.max(1),
            upd.frequency_scale,
        );
        self.height = history
            .iter()
            .map(|c| c.magnitudes_db.len().min(MAX_TEXTURE_BINS) as u32)
            .find(|&h| h > 0)
            .unwrap_or(0);
        if self.capacity == 0 || self.height == 0 {
            (self.values, self.write_idx, self.col_count, self.mapping) =
                (vec![], 0, 0, BinMapping::default());
            return;
        }
        let passthrough = upd.display_bins_hz.as_ref().is_some_and(|b| {
            !b.is_empty() && history.iter().all(|c| c.magnitudes_db.len() == b.len())
        });
        self.display_freqs = upd
            .display_bins_hz
            .clone()
            .filter(|b| passthrough && b.len() >= self.height as usize)
            .map(|b| Arc::from(&b[..self.height as usize]))
            .unwrap_or_else(|| Arc::from([]));
        self.mapping = BinMapping::new(
            self.height as usize,
            self.fft_size,
            self.sample_rate,
            self.scale,
            passthrough,
        );
        self.values = vec![0.0; self.capacity as usize * self.height as usize];
        (self.write_idx, self.col_count) = (0, 0);
        let h = self.height as usize;
        for col in history {
            if col.magnitudes_db.len() >= h {
                self.push_column(&col.magnitudes_db, style);
            }
        }
        if self.col_count > 0 {
            self.pending_base = Some(Arc::new(self.values.clone()));
        }
    }

    fn append(&mut self, columns: &[SpectrogramColumn], style: &SpectrogramStyle) {
        if self.capacity == 0 || self.height == 0 {
            return;
        }
        let h = self.height as usize;
        for col in columns.iter().filter(|c| c.magnitudes_db.len() >= h) {
            let idx = self.push_column(&col.magnitudes_db, style);
            let start = idx as usize * h;
            let mut buf = self.pool.acquire(h);
            buf.copy_from_slice(&self.values[start..start + h]);
            self.pending_cols.push(SpectrogramColumnUpdate {
                column_index: idx,
                values: Arc::new(ColumnBuffer::new(buf, self.pool.clone())),
            });
        }
        if self.pending_cols.len() as u32 >= (self.capacity / 2).max(16) {
            self.pending_cols.clear();
            self.pending_base = Some(Arc::new(self.values.clone()));
        }
    }

    fn push_column(&mut self, mags: &[f32], style: &SpectrogramStyle) -> u32 {
        let (h, col, n) = (self.height as usize, self.write_idx, mags.len());
        let inv = 1.0 / (style.ceiling_db - style.floor_db).max(f32::EPSILON);
        let out = &mut self.values[col as usize * h..(col as usize + 1) * h];
        for (i, v) in out.iter_mut().enumerate() {
            let (lo, hi, w) = (
                self.mapping.lower[i].min(n - 1),
                self.mapping.upper[i].min(n - 1),
                self.mapping.weight[i],
            );
            *v = ((mags[lo] + w * (mags[hi] - mags[lo])).clamp(style.floor_db, style.ceiling_db)
                - style.floor_db)
                * inv;
        }
        if self.col_count < self.capacity {
            self.col_count += 1;
        }
        self.write_idx = (self.write_idx + 1) % self.capacity;
        col
    }

    fn needs_rebuild(&self, upd: &SpectrogramUpdate, new_height: Option<u32>) -> bool {
        self.capacity == 0
            || self.height == 0
            || self.capacity != upd.history_length as u32
            || new_height.is_some_and(|h| h > 0 && h != self.height)
            || (self.sample_rate - upd.sample_rate).abs() > f32::EPSILON
            || self.fft_size != upd.fft_size
    }

    fn latest_column(&self) -> u32 {
        if self.col_count == 0 {
            0
        } else {
            (self.write_idx + self.capacity - 1) % self.capacity
        }
    }
}

#[derive(Clone, Debug)]
pub struct SpectrogramState {
    buffer: RefCell<SpectrogramBuffer>,
    style: SpectrogramStyle,
    palette: [Color; SPECTROGRAM_PALETTE_SIZE],
    history: VecDeque<SpectrogramColumn>,
    instance_id: u64,
}

impl SpectrogramState {
    pub fn new() -> Self {
        Self {
            buffer: RefCell::new(SpectrogramBuffer::new()),
            style: SpectrogramStyle::default(),
            palette: theme::DEFAULT_SPECTROGRAM_PALETTE,
            history: VecDeque::new(),
            instance_id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    pub fn set_palette(&mut self, palette: [Color; SPECTROGRAM_PALETTE_SIZE]) {
        if self.palette != palette {
            self.palette = palette;
            let values = self.buffer.borrow().values.clone();
            self.buffer.borrow_mut().pending_base = Some(Arc::new(values));
        }
    }

    pub fn palette(&self) -> [Color; SPECTROGRAM_PALETTE_SIZE] {
        self.palette
    }

    pub fn apply_update(&mut self, upd: &SpectrogramUpdate) {
        if upd.new_columns.is_empty() && !upd.reset {
            return;
        }
        if upd.reset {
            self.history.clear();
        }
        self.history.extend(upd.new_columns.iter().cloned());
        if self.history.len() > upd.history_length {
            self.history
                .drain(0..self.history.len() - upd.history_length);
        }
        let new_h = upd
            .new_columns
            .iter()
            .map(|c| c.magnitudes_db.len().min(MAX_TEXTURE_BINS) as u32)
            .find(|&h| h > 0);
        let mut buf = self.buffer.borrow_mut();
        if upd.reset || buf.needs_rebuild(upd, new_h) {
            buf.rebuild(&self.history, upd, &self.style);
        } else if !upd.new_columns.is_empty() {
            buf.append(&upd.new_columns, &self.style);
        }
    }

    pub fn visual_params(&self, bounds: Rectangle) -> Option<SpectrogramParams> {
        let buf = self.buffer.borrow();
        if buf.capacity == 0 || buf.height == 0 || buf.col_count == 0 {
            return None;
        }
        let op = self.style.opacity.clamp(0.0, 1.0);
        let to_rgba = |c: Color| [c.r, c.g, c.b, c.a * op];
        Some(SpectrogramParams {
            instance_id: self.instance_id,
            bounds,
            texture_width: buf.capacity,
            texture_height: buf.height,
            column_count: buf.col_count,
            latest_column: buf.latest_column(),
            base_data: buf.pending_base.clone(),
            column_updates: buf.pending_cols.clone(),
            palette: self.palette.map(to_rgba),
            background: to_rgba(self.style.background),
            contrast: self.style.contrast,
        })
    }

    fn frequency_at_y(&self, y: f32, bounds: Rectangle) -> Option<f32> {
        if bounds.height <= 0.0 || y < bounds.y || y > bounds.y + bounds.height {
            return None;
        }
        let norm = (y - bounds.y) / bounds.height;
        let buf = self.buffer.borrow();
        if !buf.display_freqs.is_empty() {
            return buf
                .display_freqs
                .get((norm * buf.display_freqs.len() as f32).floor() as usize)
                .copied();
        }
        if buf.fft_size == 0 || buf.sample_rate <= 0.0 {
            return None;
        }
        let (nyq, min_f) = (
            (buf.sample_rate / 2.0).max(1.0),
            (buf.sample_rate / buf.fft_size as f32).max(20.0),
        );
        let freq = norm_to_freq(1.0 - norm, nyq, min_f, buf.scale);
        (freq.is_finite() && freq > 0.0).then_some(freq)
    }

    fn clear_pending(&self) {
        let mut buf = self.buffer.borrow_mut();
        (buf.pending_base, buf.pending_cols) = (None, vec![]);
    }
}

#[derive(Default)]
struct TooltipState {
    cursor: Option<Point>,
}

pub struct Spectrogram<'a> {
    state: &'a RefCell<SpectrogramState>,
}

impl<'a> Spectrogram<'a> {
    pub fn new(state: &'a RefCell<SpectrogramState>) -> Self {
        Self { state }
    }

    fn draw_tooltip(
        &self,
        renderer: &mut iced::Renderer,
        theme: &iced::Theme,
        bounds: Rectangle,
        cursor: Point,
    ) {
        let state = self.state.borrow();
        let Some(freq) = state.frequency_at_y(cursor.y, bounds) else {
            return;
        };
        let content = MusicalNote::from_frequency(freq)
            .map(|n| format!("{:.1} Hz | {}", freq, n.format()))
            .unwrap_or_else(|| format!("{:.1} Hz", freq));
        let (font, line_h, align_x, align_y, shaping, wrap) = (
            iced::Font::default(),
            text::LineHeight::default(),
            iced::alignment::Horizontal::Left.into(),
            iced::alignment::Vertical::Top,
            text::Shaping::Basic,
            text::Wrapping::None,
        );
        let tsz = Paragraph::with_text(text::Text {
            content: content.as_str(),
            bounds: Size::INFINITE,
            size: iced::Pixels(TOOLTIP_SIZE),
            line_height: line_h,
            font,
            align_x,
            align_y,
            shaping,
            wrapping: wrap,
        })
        .min_bounds();
        let sz = Size::new(
            tsz.width + TOOLTIP_PAD * 2.0,
            tsz.height + TOOLTIP_PAD * 2.0,
        );
        let x = if cursor.x + 12.0 + sz.width <= bounds.x + bounds.width {
            cursor.x + 12.0
        } else {
            (cursor.x - 12.0 - sz.width).max(bounds.x)
        };
        let tb = Rectangle::new(
            Point::new(
                x,
                (cursor.y - sz.height * 0.5).clamp(bounds.y, bounds.y + bounds.height - sz.height),
            ),
            sz,
        );
        let pal = theme.extended_palette();
        let mut q = |b, brd, bg| {
            renderer.fill_quad(
                Quad {
                    bounds: b,
                    border: brd,
                    shadow: Default::default(),
                    snap: true,
                },
                Background::Color(bg),
            )
        };
        q(
            Rectangle::new(Point::new(tb.x + 1.0, tb.y + 1.0), sz),
            Default::default(),
            theme::with_alpha(pal.background.base.color, 0.3),
        );
        q(tb, theme::sharp_border(), pal.background.strong.color);
        let pos = Point::new(tb.x + TOOLTIP_PAD, tb.y + TOOLTIP_PAD);
        renderer.fill_text(
            text::Text {
                content,
                bounds: tsz,
                size: iced::Pixels(TOOLTIP_SIZE),
                line_height: line_h,
                font,
                align_x,
                align_y,
                shaping,
                wrapping: wrap,
            },
            pos,
            pal.background.base.text,
            Rectangle::new(pos, tsz),
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
        &mut self,
        _: &mut Tree,
        _: &iced::Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        layout::Node::new(limits.resolve(Length::Fill, Length::Fill, Size::ZERO))
    }

    fn update(
        &mut self,
        tree: &mut Tree,
        event: &iced::Event,
        layout: Layout<'_>,
        _: mouse::Cursor,
        _: &iced::Renderer,
        _: &mut dyn iced::advanced::Clipboard,
        _: &mut iced::advanced::Shell<'_, Message>,
        _: &Rectangle,
    ) {
        let st = tree.state.downcast_mut::<TooltipState>();
        let b = layout.bounds();
        match event {
            iced::Event::Mouse(mouse::Event::CursorMoved { position }) => {
                st.cursor = b.contains(*position).then_some(*position)
            }
            iced::Event::Mouse(mouse::Event::CursorLeft) => st.cursor = None,
            _ => {}
        }
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut iced::Renderer,
        theme: &iced::Theme,
        _: &renderer::Style,
        layout: Layout<'_>,
        _: mouse::Cursor,
        _: &Rectangle,
    ) {
        let bounds = layout.bounds();
        let state = self.state.borrow();
        renderer.fill_quad(
            Quad {
                bounds,
                border: Default::default(),
                shadow: Default::default(),
                snap: true,
            },
            Background::Color(state.style.background),
        );
        if let Some(p) = state.visual_params(bounds) {
            renderer.draw_primitive(bounds, SpectrogramPrimitive::new(p));
        }
        state.clear_pending();
        drop(state);
        if let Some(c) = tree.state.downcast_ref::<TooltipState>().cursor
            && bounds.contains(c)
        {
            renderer.with_layer(bounds, |r| self.draw_tooltip(r, theme, bounds, c));
        }
    }

    fn mouse_interaction(
        &self,
        _: &Tree,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        _: &Rectangle,
        _: &iced::Renderer,
    ) -> mouse::Interaction {
        if cursor.is_over(layout.bounds()) {
            mouse::Interaction::Crosshair
        } else {
            mouse::Interaction::default()
        }
    }
}

pub fn widget<'a, Message: 'a>(state: &'a RefCell<SpectrogramState>) -> Element<'a, Message> {
    Element::new(Spectrogram::new(state))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bin_mapping_log_profile() {
        let m = BinMapping::new(
            8,
            2048,
            DEFAULT_SAMPLE_RATE,
            FrequencyScale::Logarithmic,
            false,
        );
        assert_eq!(m.lower.len(), 8);
        assert!(m.lower[0] as f32 + m.weight[0] > m.lower[7] as f32 + m.weight[7]);
    }
}
