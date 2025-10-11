use crate::audio::meter_tap::MeterFormat;
use crate::dsp::loudness::{
    LoudnessConfig, LoudnessProcessor as CoreLoudnessProcessor, LoudnessSnapshot,
};
use crate::dsp::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::ui::render::lufs_meter::{
    ChannelVisual, FillVisual, GuideLine, LufsMeterPrimitive, VisualParams,
};
use crate::ui::theme;
use iced::advanced::Renderer as _;
use iced::advanced::graphics::text::Paragraph as RenderParagraph;
use iced::advanced::renderer::{self, Quad};
use iced::advanced::text::Paragraph as _;
use iced::advanced::text::{LineHeight, Renderer as TextRenderer, Shaping, Text, Wrapping};
use iced::advanced::widget::{Tree, tree};
use iced::advanced::{Layout, Widget, layout, mouse};
use iced::alignment::{Horizontal, Vertical};
use iced::font::Weight as FontWeight;
use iced::{
    Background, Border, Color, Element, Font, Length, Pixels, Point, Rectangle, Size, Theme,
};
use iced_wgpu::primitive::Renderer as _;
use std::cell::RefCell;
use std::time::Instant;

const CHANNELS: usize = 2;
const DEFAULT_MIN_LUFS: f32 = -60.0;
const DEFAULT_MAX_LUFS: f32 = 0.0;
const DEFAULT_HEIGHT: f32 = 300.0;
const DEFAULT_WIDTH: f32 = 140.0;
const DEFAULT_SHORT_TERM_WINDOW_SECS: f32 = 3.0;
const DEFAULT_RMS_FAST_WINDOW_SECS: f32 = 0.03;
const GUIDE_LABEL_CHAR_WIDTH: f32 = 7.0;
const GUIDE_LABEL_FONT_SIZE: f32 = 10.0;
const GUIDE_LABEL_HEIGHT: f32 = 22.0;
const GUIDE_LABEL_MIN_WIDTH: f32 = 52.0;
const GUIDE_LABEL_MAX_WIDTH: f32 = 116.0;
const GUIDE_LABEL_GAP: f32 = 7.0;
const GUIDE_LABEL_MARGIN: f32 = 8.0;
const GUIDE_LABEL_BAR_MARGIN: f32 = 3.0;
const GUIDE_LEVELS: [f32; 5] = [-6.0, -12.0, -18.0, -24.0, -36.0];
const GUIDE_LINE_LENGTH: f32 = 4.0;
const GUIDE_LINE_THICKNESS: f32 = 1.2;
const GUIDE_LINE_PADDING: f32 = 3.0;
const VALUE_SCALE_MIX: f32 = 0.6;
const METER_VERTICAL_PADDING: f32 = 0.0;
const CHANNEL_GAP_FRACTION: f32 = 0.1;
const CHANNEL_WIDTH_SCALE: f32 = 0.6;
const LIVE_VALUE_LABEL_GAP: f32 = GUIDE_LABEL_BAR_MARGIN;
const LIVE_VALUE_LABEL_MARGIN: f32 = 6.0;
const LIVE_VALUE_LABEL_HEIGHT: f32 = 20.0;
const LIVE_VALUE_LABEL_TEMPLATE: &str = "-99.9LUFS";
const LIVE_VALUE_LABEL_FONT_SIZE: f32 = 12.0;
const LIVE_VALUE_LABEL_CHAR_WIDTH: f32 = 7.0;
const LIVE_VALUE_LABEL_HORIZONTAL_PADDING: f32 = 2.0;
const LIVE_VALUE_LABEL_MIN_WIDTH: f32 = 21.0;
const LIVE_VALUE_LABEL_MAX_WIDTH: f32 = 128.0;

const fn clamp_width(value: f32, min: f32, max: f32) -> f32 {
    let mut v = if value < min { min } else { value };
    if v > max {
        v = max;
    }
    v
}

const LIVE_VALUE_LABEL_TEMPLATE_WIDTH: f32 = clamp_width(
    LIVE_VALUE_LABEL_TEMPLATE.len() as f32 * LIVE_VALUE_LABEL_CHAR_WIDTH
        + LIVE_VALUE_LABEL_HORIZONTAL_PADDING * 2.0,
    LIVE_VALUE_LABEL_MIN_WIDTH,
    LIVE_VALUE_LABEL_MAX_WIDTH,
);

const LIVE_VALUE_LABEL_RESERVE: f32 =
    LIVE_VALUE_LABEL_GAP + LIVE_VALUE_LABEL_MARGIN + LIVE_VALUE_LABEL_TEMPLATE_WIDTH;

fn guide_line_color() -> Color {
    let secondary = theme::text_secondary();
    let surface = theme::surface_color();
    theme::with_alpha(theme::mix_colors(secondary, surface, 0.35), 0.88)
}

fn guide_label_color() -> Color {
    theme::mix_colors(theme::text_color(), theme::surface_color(), 0.2)
}

fn live_label_background_color() -> Color {
    theme::with_alpha(
        theme::mix_colors(theme::surface_color(), theme::accent_primary(), 0.25),
        0.92,
    )
}

fn live_label_text_color() -> Color {
    theme::text_color()
}

/// UI wrapper around the shared loudness processor.
#[derive(Debug, Clone)]
pub struct LufsProcessor {
    inner: CoreLoudnessProcessor,
    channels: usize,
}

impl LufsProcessor {
    pub fn new(sample_rate: f32) -> Self {
        let config = LoudnessConfig {
            sample_rate,
            short_term_window: DEFAULT_SHORT_TERM_WINDOW_SECS,
            rms_fast_window: DEFAULT_RMS_FAST_WINDOW_SECS,
            floor_lufs: DEFAULT_MIN_LUFS,
        };
        Self {
            inner: CoreLoudnessProcessor::new(config),
            channels: CHANNELS,
        }
    }

    pub fn ingest(&mut self, samples: &[f32], format: MeterFormat) -> LoudnessSnapshot {
        if samples.is_empty() {
            return self.inner.snapshot().clone();
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

        match self.inner.process_block(&block) {
            ProcessorUpdate::Snapshot(snapshot) => snapshot,
            ProcessorUpdate::None => self.inner.snapshot().clone(),
        }
    }
}

/// View-model state consumed by the LUFS meter widget.
#[derive(Debug, Clone)]
pub struct LufsMeterState {
    short_term_lufs: [f32; CHANNELS],
    true_peak_db: [f32; CHANNELS],
    range: (f32, f32),
    style: VisualStyle,
}

impl LufsMeterState {
    pub fn new() -> Self {
        Self {
            short_term_lufs: [DEFAULT_MIN_LUFS; CHANNELS],
            true_peak_db: [DEFAULT_MIN_LUFS; CHANNELS],
            range: (DEFAULT_MIN_LUFS, DEFAULT_MAX_LUFS),
            style: VisualStyle::default(),
        }
    }

    pub fn apply_snapshot(&mut self, snapshot: &LoudnessSnapshot) {
        let floor = self.range.0;
        Self::copy_into(&mut self.short_term_lufs, &snapshot.short_term_lufs, floor);
        Self::copy_into(&mut self.true_peak_db, &snapshot.true_peak_db, floor);
    }

    fn copy_into(target: &mut [f32; CHANNELS], source: &[f32], floor: f32) {
        let copied = target.len().min(source.len());
        target[..copied].copy_from_slice(&source[..copied]);
        for value in &mut target[copied..] {
            *value = floor;
        }
    }

    pub fn range(&self) -> (f32, f32) {
        self.range
    }

    pub fn style(&self) -> &VisualStyle {
        &self.style
    }

    pub fn short_term_average(&self) -> f32 {
        self.short_term_lufs.iter().copied().sum::<f32>() / CHANNELS as f32
    }

    pub fn visual_params(&self, range: (f32, f32)) -> VisualParams {
        let (min, max) = range;
        let style = *self.style();
        let short_term = self.short_term_average();
        let peak_channel = self.peak_channel(style);
        let short_term_channel = Self::short_term_channel(style, short_term);
        let (guides, max_label_width) = Self::build_guides((min, max));
        let channels = vec![peak_channel, short_term_channel];

        let left_padding = GUIDE_LABEL_MARGIN
            + max_label_width
            + GUIDE_LABEL_GAP
            + GUIDE_LINE_LENGTH
            + GUIDE_LINE_PADDING;

        VisualParams {
            min_lufs: min,
            max_lufs: max,
            channels,
            channel_gap_fraction: CHANNEL_GAP_FRACTION,
            channel_width_scale: CHANNEL_WIDTH_SCALE,
            short_term_value: short_term,
            guides,
            left_padding,
            right_padding: LIVE_VALUE_LABEL_RESERVE,
            guide_padding: GUIDE_LINE_PADDING,
            value_scale_bias: VALUE_SCALE_MIX,
            vertical_padding: METER_VERTICAL_PADDING,
        }
    }

    fn peak_channel(&self, style: VisualStyle) -> ChannelVisual {
        let mut fills = Vec::with_capacity(CHANNELS);
        for (value, color) in self.true_peak_db.iter().zip([
            theme::color_to_rgba(style.raw_peak_left_fill),
            theme::color_to_rgba(style.raw_peak_right_fill),
        ]) {
            fills.push(FillVisual {
                value_lufs: *value,
                color,
            });
        }

        ChannelVisual {
            background_color: theme::color_to_rgba(style.background),
            fills,
        }
    }

    fn short_term_channel(style: VisualStyle, short_term: f32) -> ChannelVisual {
        ChannelVisual {
            background_color: theme::color_to_rgba(style.background),
            fills: vec![FillVisual {
                value_lufs: short_term,
                color: theme::color_to_rgba(style.short_term_fill),
            }],
        }
    }

    fn build_guides(range: (f32, f32)) -> (Vec<GuideLine>, f32) {
        let (min, max) = range;
        let mut guides = Vec::with_capacity(GUIDE_LEVELS.len());
        let mut max_label_width = 0.0f32;
        let guide_color = theme::color_to_rgba(guide_line_color());

        for level in GUIDE_LEVELS.iter().copied() {
            if level > max || level < min {
                continue;
            }

            let label = format!("{:.1}", level.abs());
            let label_width = guide_label_width(label.as_str());
            max_label_width = max_label_width.max(label_width);
            guides.push(GuideLine {
                value_lufs: level,
                color: guide_color,
                length: GUIDE_LINE_LENGTH,
                thickness: GUIDE_LINE_THICKNESS,
                label: Some(label),
                label_width,
            });
        }

        (guides, max_label_width)
    }
}

/// Palette for the LUFS meter.
#[derive(Debug, Clone, Copy)]
pub struct VisualStyle {
    pub background: Color,
    pub raw_peak_left_fill: Color,
    pub raw_peak_right_fill: Color,
    pub short_term_fill: Color,
}

impl Default for VisualStyle {
    fn default() -> Self {
        let surface = theme::surface_color();
        let elevated = theme::elevated_color();
        let hover = theme::hover_color();
        let text = theme::text_color();
        let accent = theme::accent_primary();
        let success = theme::accent_success();

        Self {
            background: surface,
            raw_peak_left_fill: theme::mix_colors(elevated, text, 0.18),
            raw_peak_right_fill: theme::mix_colors(hover, text, 0.12),
            short_term_fill: theme::mix_colors(accent, success, 0.35),
        }
    }
}

/// Declare a LUFS meter widget that renders using the iced_wgpu backend.
#[derive(Debug)]
pub struct LufsMeter<'a> {
    state: &'a LufsMeterState,
    explicit_range: Option<(f32, f32)>,
    height: f32,
    width: f32,
    fill_height: bool,
}

impl<'a> LufsMeter<'a> {
    pub fn new(state: &'a LufsMeterState) -> Self {
        Self {
            state,
            explicit_range: None,
            height: DEFAULT_HEIGHT,
            width: DEFAULT_WIDTH,
            fill_height: false,
        }
    }

    pub fn with_range(mut self, min: f32, max: f32) -> Self {
        self.explicit_range = Some((min, max));
        self
    }

    pub fn with_height(mut self, height: f32) -> Self {
        self.height = height.max(32.0);
        self
    }

    pub fn fill_height(mut self) -> Self {
        self.fill_height = true;
        self
    }

    pub fn with_width(mut self, width: f32) -> Self {
        self.width = width.max(0.0);
        self
    }

    fn active_range(&self) -> (f32, f32) {
        self.explicit_range.unwrap_or_else(|| self.state.range())
    }

    fn total_width(&self) -> f32 {
        self.width + LIVE_VALUE_LABEL_RESERVE
    }
}

impl<'a, Message> Widget<Message, Theme, iced::Renderer> for LufsMeter<'a> {
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<GuideLabelCache>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(GuideLabelCache::default())
    }

    fn size(&self) -> Size<Length> {
        let width = self.total_width();
        let height = if self.fill_height {
            Length::Fill
        } else {
            Length::Fixed(self.height)
        };

        Size::new(Length::Fixed(width), height)
    }

    fn layout(
        &self,
        _tree: &mut Tree,
        _renderer: &iced::Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        let width = self.total_width();
        let height = if self.fill_height {
            Length::Fill
        } else {
            Length::Fixed(self.height)
        };

        let fallback_height = self.height;

        let size = limits.resolve(
            Length::Fixed(width),
            height,
            Size::new(width, fallback_height),
        );

        layout::Node::new(size)
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut iced::Renderer,
        _theme: &Theme,
        _style: &renderer::Style,
        layout: Layout<'_>,
        _cursor: mouse::Cursor,
        _viewport: &Rectangle,
    ) {
        let bounds = layout.bounds();
        let (min, max) = self.active_range();
        let params = self.state.visual_params((min, max));
        renderer.draw_primitive(bounds, LufsMeterPrimitive::new(params.clone()));
        draw_guide_labels(tree, renderer, bounds, &params);
        draw_live_value_label(tree, renderer, bounds, &params);
    }

    fn children(&self) -> Vec<Tree> {
        Vec::new()
    }

    fn diff(&self, _tree: &mut Tree) {}
}

/// Convenience conversion into an [`iced::Element`].
pub fn widget_with_layout<'a, Message>(
    state: &'a LufsMeterState,
    preferred_width: f32,
    preferred_height: f32,
) -> Element<'a, Message>
where
    Message: 'a,
{
    let (min, max) = state.range();
    Element::new(
        LufsMeter::new(state)
            .with_range(min, max)
            .fill_height()
            .with_height(preferred_height)
            .with_width(preferred_width),
    )
}

#[derive(Default)]
struct GuideLabelCache {
    entries: RefCell<Vec<CachedParagraph>>,
    live_value: RefCell<Option<LiveLabelParagraph>>,
}

struct CachedParagraph {
    label: String,
    bounds: Size,
    paragraph: RenderParagraph,
}

impl CachedParagraph {
    fn new(label: &str, bounds: Size) -> Self {
        Self {
            label: label.to_owned(),
            bounds,
            paragraph: build_label_paragraph(label, bounds),
        }
    }

    fn ensure(&mut self, bounds: Size) {
        if !size_eq(self.bounds, bounds) {
            self.paragraph.resize(bounds);
            self.bounds = bounds;
        }
    }
}

struct LiveLabelParagraph {
    label: String,
    bounds: Size,
    paragraph: RenderParagraph,
}

impl LiveLabelParagraph {
    fn new(label: &str, bounds: Size) -> Self {
        Self {
            label: label.to_owned(),
            bounds,
            paragraph: build_live_label_paragraph(label, bounds),
        }
    }

    fn ensure(&mut self, label: &str, bounds: Size) {
        if self.label != label {
            *self = Self::new(label, bounds);
        } else if !size_eq(self.bounds, bounds) {
            self.paragraph.resize(bounds);
            self.bounds = bounds;
        }
    }
}

fn ensure_label_paragraph(entries: &mut Vec<CachedParagraph>, label: &str, bounds: Size) -> usize {
    if let Some((index, entry)) = entries
        .iter_mut()
        .enumerate()
        .find(|(_, entry)| entry.label == label)
    {
        entry.ensure(bounds);
        index
    } else {
        entries.push(CachedParagraph::new(label, bounds));
        entries.len() - 1
    }
}

fn prune_label_paragraphs(entries: &mut Vec<CachedParagraph>, active: &[&str]) {
    if active.is_empty() {
        entries.clear();
        return;
    }

    entries.retain(|entry| active.iter().any(|label| *label == entry.label));
}

fn build_label_paragraph(label: &str, bounds: Size) -> RenderParagraph {
    RenderParagraph::with_text(Text {
        content: label,
        bounds,
        size: Pixels(GUIDE_LABEL_FONT_SIZE),
        line_height: LineHeight::Relative(1.0),
        font: iced::Font::default(),
        horizontal_alignment: Horizontal::Left,
        vertical_alignment: Vertical::Center,
        shaping: Shaping::Advanced,
        wrapping: Wrapping::None,
    })
}

fn size_eq(a: Size, b: Size) -> bool {
    (a.width - b.width).abs() <= f32::EPSILON && (a.height - b.height).abs() <= f32::EPSILON
}

fn draw_guide_labels(
    tree: &Tree,
    renderer: &mut iced::Renderer,
    bounds: Rectangle,
    params: &VisualParams,
) {
    let cache = tree.state.downcast_ref::<GuideLabelCache>();
    let mut entries = cache.entries.borrow_mut();

    if params.guides.is_empty() {
        entries.clear();
        return;
    }

    let Some(context) = GuideRenderContext::new(bounds, params) else {
        entries.clear();
        return;
    };
    let label_height = GUIDE_LABEL_HEIGHT;
    let label_color = guide_label_color();
    let mut active_labels = Vec::with_capacity(params.guides.len());

    for guide in &params.guides {
        let Some(label) = guide.label.as_deref() else {
            continue;
        };

        let label_bounds = Size::new(guide.label_width, label_height);
        let index = ensure_label_paragraph(&mut entries, label, label_bounds);
        let paragraph = &entries[index].paragraph;

        let text_bounds = paragraph.min_bounds();
        let text_width = text_bounds.width.max(1.0);
        let center_y = context.center(params.meter_ratio(guide.value_lufs));
        let (clip_bounds, position) = context.label_clip(center_y, label_height, text_width);

        renderer.fill_paragraph(paragraph, position, label_color, clip_bounds);
        active_labels.push(label);
    }

    prune_label_paragraphs(&mut entries, &active_labels);
}

fn draw_live_value_label(
    tree: &Tree,
    renderer: &mut iced::Renderer,
    bounds: Rectangle,
    params: &VisualParams,
) {
    let cache = tree.state.downcast_ref::<GuideLabelCache>();
    let mut live_entry = cache.live_value.borrow_mut();

    if !params.short_term_value.is_finite() {
        live_entry.take();
        return;
    }

    let Some(context) = LiveLabelContext::new(bounds, params) else {
        live_entry.take();
        return;
    };

    let label = format!("{:.1}LUFS", params.short_term_value);
    let label_height = LIVE_VALUE_LABEL_HEIGHT.max(12.0);
    let available_width = context.available_width();

    let minimum_width = live_label_min_width(label.as_str());

    if available_width < minimum_width {
        live_entry.take();
        return;
    }

    let label_width = minimum_width.min(available_width);
    let label_bounds = Size::new(label_width, label_height);

    if let Some(entry) = live_entry.as_mut() {
        entry.ensure(label.as_str(), label_bounds);
    } else {
        *live_entry = Some(LiveLabelParagraph::new(label.as_str(), label_bounds));
    }

    let Some(entry) = live_entry.as_ref() else {
        return;
    };

    let text_bounds = entry.paragraph.min_bounds();
    let text_width = text_bounds.width.max(1.0);
    if text_width > label_width + f32::EPSILON {
        live_entry.take();
        return;
    }

    let (clip_bounds, position) = context.clip(label_height, label_width);

    let label_background = live_label_background_color();
    renderer.fill_quad(
        Quad {
            bounds: clip_bounds,
            border: Border::default(),
            shadow: Default::default(),
        },
        Background::Color(label_background),
    );

    renderer.fill_paragraph(
        &entry.paragraph,
        position,
        live_label_text_color(),
        clip_bounds,
    );
}

struct GuideRenderContext {
    y0: f32,
    y1: f32,
    content_height: f32,
    max_label_right: f32,
    min_left: f32,
}

impl GuideRenderContext {
    fn new(bounds: Rectangle, params: &VisualParams) -> Option<Self> {
        let (y0, y1) = params.vertical_bounds(&bounds)?;
        let content_height = (y1 - y0).max(1.0);
        let (meter_start, _) = params.meter_horizontal_bounds(&bounds)?;
        let max_right = (meter_start - GUIDE_LABEL_BAR_MARGIN).max(bounds.x);
        let anchor_x = meter_start - params.guide_padding;
        let max_label_right = (anchor_x - GUIDE_LABEL_GAP).min(max_right);
        let min_left = bounds.x + GUIDE_LABEL_MARGIN;

        if max_label_right <= min_left {
            return None;
        }

        Some(Self {
            y0,
            y1,
            content_height,
            max_label_right,
            min_left,
        })
    }

    fn center(&self, ratio: f32) -> f32 {
        self.y1 - self.content_height * ratio
    }

    fn label_clip(&self, center_y: f32, label_height: f32, text_width: f32) -> (Rectangle, Point) {
        let available_bottom = (self.y1 - label_height).max(self.y0);
        let label_top = (center_y - label_height * 0.5).clamp(self.y0, available_bottom);

        let mut label_left = (self.max_label_right - text_width).max(self.min_left);
        let mut label_right = label_left + text_width;
        if label_right > self.max_label_right {
            label_right = self.max_label_right;
            label_left = (label_right - text_width).max(self.min_left);
        }

        let clip_bounds = Rectangle {
            x: label_left,
            y: label_top,
            width: label_right - label_left,
            height: label_height,
        };
        let position = Point::new(label_left, clip_bounds.y + clip_bounds.height * 0.5);

        (clip_bounds, position)
    }
}

struct LiveLabelContext {
    label_left: f32,
    label_right_limit: f32,
    center_y: f32,
    y0: f32,
    y1: f32,
}

impl LiveLabelContext {
    fn new(bounds: Rectangle, params: &VisualParams) -> Option<Self> {
        let (y0, y1) = params.vertical_bounds(&bounds)?;
        let (_, meter_right) = params.meter_horizontal_bounds(&bounds)?;
        let ratio = params.meter_ratio(params.short_term_value);
        if !ratio.is_finite() {
            return None;
        }

        let content_height = (y1 - y0).max(1.0);
        let center_y = y1 - content_height * ratio;
        let label_left = meter_right + LIVE_VALUE_LABEL_GAP;
        let label_right_limit = bounds.x + bounds.width.max(0.0) - LIVE_VALUE_LABEL_MARGIN;

        if label_right_limit <= label_left {
            return None;
        }

        Some(Self {
            label_left,
            label_right_limit,
            center_y,
            y0,
            y1,
        })
    }

    fn available_width(&self) -> f32 {
        (self.label_right_limit - self.label_left).max(0.0)
    }

    fn clip(&self, label_height: f32, label_width: f32) -> (Rectangle, Point) {
        let available_height = (self.y1 - label_height).max(self.y0);
        let label_top = (self.center_y - label_height * 0.5).clamp(self.y0, available_height);

        let clip_bounds = Rectangle {
            x: self.label_left,
            y: label_top,
            width: label_width,
            height: label_height,
        };
        let position = Point::new(
            self.label_left + label_width * 0.5,
            clip_bounds.y + clip_bounds.height * 0.5,
        );

        (clip_bounds, position)
    }
}

fn guide_label_width(label: &str) -> f32 {
    clamp_width(
        label.len() as f32 * GUIDE_LABEL_CHAR_WIDTH,
        GUIDE_LABEL_MIN_WIDTH,
        GUIDE_LABEL_MAX_WIDTH,
    )
}

fn build_live_label_paragraph(label: &str, bounds: Size) -> RenderParagraph {
    RenderParagraph::with_text(Text {
        content: label,
        bounds,
        size: Pixels(LIVE_VALUE_LABEL_FONT_SIZE),
        line_height: LineHeight::Relative(1.0),
        font: Font {
            weight: FontWeight::Bold,
            ..Font::DEFAULT
        },
        horizontal_alignment: Horizontal::Center,
        vertical_alignment: Vertical::Center,
        shaping: Shaping::Advanced,
        wrapping: Wrapping::None,
    })
}

fn live_label_min_width(label: &str) -> f32 {
    clamp_width(
        label.len() as f32 * LIVE_VALUE_LABEL_CHAR_WIDTH
            + LIVE_VALUE_LABEL_HORIZONTAL_PADDING * 2.0,
        LIVE_VALUE_LABEL_MIN_WIDTH,
        LIVE_VALUE_LABEL_MAX_WIDTH,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_aggregates_channels() {
        let mut state = LufsMeterState::new();
        state.apply_snapshot(&LoudnessSnapshot {
            short_term_lufs: vec![-12.0, -6.0],
            rms_fast_db: vec![-15.0, -9.0],
            true_peak_db: vec![-1.0, -3.0],
        });

        assert!((state.short_term_average() + 9.0).abs() < f32::EPSILON);
        assert_eq!(state.true_peak_db[0], -1.0);
        assert_eq!(state.true_peak_db[1], -3.0);

        let params = state.visual_params(state.range());
        assert_eq!(params.channels.len(), 2);
        assert_eq!(params.channels[0].fills.len(), 2);
        assert_eq!(params.channels[1].fills.len(), 1);
        let left_peak = params.channels[0].fills[0].value_lufs;
        let right_peak = params.channels[0].fills[1].value_lufs;
        let short_term = params.channels[1].fills[0].value_lufs;
        assert!((left_peak + 1.0).abs() < f32::EPSILON);
        assert!((right_peak + 3.0).abs() < f32::EPSILON);
        assert!((short_term + 9.0).abs() < f32::EPSILON);
        assert!((params.short_term_value + 9.0).abs() < f32::EPSILON);
        assert!(!params.guides.is_empty());
        let labels: Vec<_> = params
            .guides
            .iter()
            .filter_map(|guide| guide.label.as_ref().cloned())
            .collect();
        assert!(labels.iter().any(|label| label.contains("6.0")));
        assert!(labels.iter().all(|label| !label.contains("LUFS")));
    }
}
