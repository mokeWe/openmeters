//! UI wrapper around the loudness DSP processor and renderer.

use crate::audio::meter_tap::MeterFormat;
use crate::dsp::loudness::{
    LoudnessConfig, LoudnessProcessor as CoreLoudnessProcessor, LoudnessSnapshot,
};
use crate::dsp::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::ui::render::loudness::{
    ChannelVisual, FillVisual, GuideLine, LoudnessMeterPrimitive, VisualParams,
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
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::fmt;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum MeterMode {
    #[default]
    LufsShortTerm,
    LufsMomentary,
    RmsFast,
    RmsSlow,
    TruePeak,
}

impl MeterMode {
    pub const ALL: &'static [MeterMode] = &[
        MeterMode::LufsShortTerm,
        MeterMode::LufsMomentary,
        MeterMode::RmsFast,
        MeterMode::RmsSlow,
        MeterMode::TruePeak,
    ];

    pub fn unit_label(self) -> &'static str {
        match self {
            MeterMode::LufsShortTerm | MeterMode::LufsMomentary => "LUFS",
            MeterMode::RmsFast | MeterMode::RmsSlow | MeterMode::TruePeak => "dB",
        }
    }
}

impl fmt::Display for MeterMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MeterMode::LufsShortTerm => f.write_str("LUFS Short-term"),
            MeterMode::LufsMomentary => f.write_str("LUFS Momentary"),
            MeterMode::RmsFast => f.write_str("RMS Fast"),
            MeterMode::RmsSlow => f.write_str("RMS Slow"),
            MeterMode::TruePeak => f.write_str("True Peak"),
        }
    }
}

const CHANNELS: usize = 2;
const DEFAULT_RANGE: (f32, f32) = (-60.0, 0.0);
const DEFAULT_HEIGHT: f32 = 300.0;
const DEFAULT_WIDTH: f32 = 140.0;
const GUIDE_LEVELS: [f32; 5] = [-6.0, -12.0, -18.0, -24.0, -36.0];
const VALUE_SCALE_MIX: f32 = 0.6;
const METER_VERTICAL_PADDING: f32 = 0.0;
const CHANNEL_GAP_FRACTION: f32 = 0.1;
const CHANNEL_WIDTH_SCALE: f32 = 0.6;

const GUIDE_LABEL_FONT_SIZE: f32 = 10.0;
const GUIDE_LABEL_HEIGHT: f32 = 22.0;
const GUIDE_LABEL_CHAR_WIDTH: f32 = 7.0;
const GUIDE_LABEL_MIN_WIDTH: f32 = 52.0;
const GUIDE_LABEL_MAX_WIDTH: f32 = 116.0;
const GUIDE_LABEL_GAP: f32 = 7.0;
const GUIDE_LABEL_MARGIN: f32 = 8.0;
const GUIDE_LABEL_BAR_MARGIN: f32 = 3.0;
const GUIDE_LINE_LENGTH: f32 = 4.0;
const GUIDE_LINE_THICKNESS: f32 = 1.0;
const GUIDE_LINE_PADDING: f32 = 3.0;

const LIVE_VALUE_LABEL_FONT_SIZE: f32 = 12.0;
const LIVE_VALUE_LABEL_HEIGHT: f32 = 20.0;
const LIVE_VALUE_LABEL_CHAR_WIDTH: f32 = 7.0;
const LIVE_VALUE_LABEL_TEMPLATE: &str = "-99.9 LUFS";
const LIVE_VALUE_LABEL_MIN_WIDTH: f32 = 21.0;
const LIVE_VALUE_LABEL_MAX_WIDTH: f32 = 128.0;
const LIVE_VALUE_LABEL_HORIZONTAL_PADDING: f32 = 2.0;
const LIVE_VALUE_LABEL_GAP: f32 = GUIDE_LABEL_BAR_MARGIN;
const LIVE_VALUE_LABEL_MARGIN: f32 = 6.0;

const fn clamp_width(value: f32, min: f32, max: f32) -> f32 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

const LIVE_VALUE_LABEL_WIDTH: f32 = clamp_width(
    LIVE_VALUE_LABEL_TEMPLATE.len() as f32 * LIVE_VALUE_LABEL_CHAR_WIDTH
        + LIVE_VALUE_LABEL_HORIZONTAL_PADDING * 2.0,
    LIVE_VALUE_LABEL_MIN_WIDTH,
    LIVE_VALUE_LABEL_MAX_WIDTH,
);

const LIVE_VALUE_LABEL_RESERVE: f32 =
    LIVE_VALUE_LABEL_GAP + LIVE_VALUE_LABEL_MARGIN + LIVE_VALUE_LABEL_WIDTH;

fn guide_line_color(theme: &Theme) -> Color {
    let palette = theme.extended_palette();
    theme::with_alpha(
        theme::mix_colors(
            palette.secondary.weak.text,
            palette.background.weak.color,
            0.35,
        ),
        0.88,
    )
}

fn guide_label_color(theme: &Theme) -> Color {
    let palette = theme.extended_palette();
    theme::mix_colors(
        palette.background.base.text,
        palette.background.weak.color,
        0.2,
    )
}

fn live_label_background(theme: &Theme) -> Color {
    let palette = theme.extended_palette();
    theme::with_alpha(
        theme::mix_colors(
            palette.background.weak.color,
            palette.primary.base.color,
            0.25,
        ),
        0.92,
    )
}

/// UI wrapper around the shared loudness processor.
#[derive(Debug, Clone)]
pub struct LoudnessMeterProcessor {
    inner: CoreLoudnessProcessor,
    channels: usize,
}

impl LoudnessMeterProcessor {
    pub fn new(sample_rate: f32) -> Self {
        let config = LoudnessConfig {
            sample_rate,
            ..Default::default()
        };
        Self {
            inner: CoreLoudnessProcessor::new(config),
            channels: CHANNELS,
        }
    }

    pub fn ingest(&mut self, samples: &[f32], format: MeterFormat) -> LoudnessSnapshot {
        if samples.is_empty() {
            return *self.inner.snapshot();
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
            ProcessorUpdate::None => *self.inner.snapshot(),
        }
    }
}

/// View-model state consumed by the loudness widget.
#[derive(Debug, Clone)]
pub struct LoudnessMeterState {
    short_term_loudness: [f32; CHANNELS],
    momentary_loudness: [f32; CHANNELS],
    rms_fast_db: [f32; CHANNELS],
    rms_slow_db: [f32; CHANNELS],
    true_peak_db: [f32; CHANNELS],
    range: (f32, f32),
    left_mode: MeterMode,
    right_mode: MeterMode,
}

impl LoudnessMeterState {
    pub fn new() -> Self {
        Self {
            short_term_loudness: [DEFAULT_RANGE.0; CHANNELS],
            momentary_loudness: [DEFAULT_RANGE.0; CHANNELS],
            rms_fast_db: [DEFAULT_RANGE.0; CHANNELS],
            rms_slow_db: [DEFAULT_RANGE.0; CHANNELS],
            true_peak_db: [DEFAULT_RANGE.0; CHANNELS],
            range: DEFAULT_RANGE,
            left_mode: MeterMode::TruePeak,
            right_mode: MeterMode::LufsShortTerm,
        }
    }

    pub fn apply_snapshot(&mut self, snapshot: &LoudnessSnapshot) {
        let floor = self.range.0;
        Self::copy_into(
            &mut self.short_term_loudness,
            &snapshot.short_term_loudness,
            floor,
        );
        Self::copy_into(
            &mut self.momentary_loudness,
            &snapshot.momentary_loudness,
            floor,
        );
        Self::copy_into(&mut self.rms_fast_db, &snapshot.rms_fast_db, floor);
        Self::copy_into(&mut self.rms_slow_db, &snapshot.rms_slow_db, floor);
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

    pub fn set_modes(&mut self, left: MeterMode, right: MeterMode) {
        self.left_mode = left;
        self.right_mode = right;
    }

    pub fn left_mode(&self) -> MeterMode {
        self.left_mode
    }

    pub fn right_mode(&self) -> MeterMode {
        self.right_mode
    }

    #[cfg(test)]
    pub fn short_term_average(&self) -> f32 {
        self.short_term_loudness.iter().copied().sum::<f32>() / CHANNELS as f32
    }

    fn get_value_for_mode(&self, mode: MeterMode, channel: usize) -> f32 {
        match mode {
            MeterMode::LufsShortTerm => {
                self.short_term_loudness.iter().copied().sum::<f32>() / CHANNELS as f32
            }
            MeterMode::LufsMomentary => {
                self.momentary_loudness.iter().copied().sum::<f32>() / CHANNELS as f32
            }
            MeterMode::RmsFast => *self.rms_fast_db.get(channel).unwrap_or(&self.range.0),
            MeterMode::RmsSlow => *self.rms_slow_db.get(channel).unwrap_or(&self.range.0),
            MeterMode::TruePeak => *self.true_peak_db.get(channel).unwrap_or(&self.range.0),
        }
    }

    pub fn visual_params(&self, range: (f32, f32), theme: &Theme) -> VisualParams {
        let (min, max) = range;
        let palette = theme.extended_palette();
        let elevated = palette.background.strong.color;
        let text = palette.background.base.text;
        let accent = palette.primary.base.color;
        let success = palette.success.base.color;

        // Approximate hover as elevated for now
        let hover = elevated;

        let short_term_fill = theme::mix_colors(accent, success, 0.35);
        let raw_peak_left_fill = theme::mix_colors(elevated, text, 0.18);
        let raw_peak_right_fill = theme::mix_colors(hover, text, 0.12);

        let guide_color = theme::color_to_rgba(guide_line_color(theme));
        let bg_color = theme::color_to_rgba(palette.background.weak.color);

        let left_values: Vec<f32> = (0..CHANNELS)
            .map(|ch| self.get_value_for_mode(self.left_mode, ch))
            .collect();
        let right_value = self.get_value_for_mode(self.right_mode, 0);

        let left_fills: Vec<_> = left_values
            .iter()
            .zip([raw_peak_left_fill, raw_peak_right_fill])
            .map(|(&value, color)| FillVisual {
                value_loudness: value,
                color: theme::color_to_rgba(color),
            })
            .collect();

        let channels = vec![
            ChannelVisual {
                background_color: bg_color,
                fills: left_fills,
            },
            ChannelVisual {
                background_color: bg_color,
                fills: vec![FillVisual {
                    value_loudness: right_value,
                    color: theme::color_to_rgba(short_term_fill),
                }],
            },
        ];

        let mut max_label_width = 0.0f32;
        let guides: Vec<_> = GUIDE_LEVELS
            .iter()
            .filter(|&&level| level >= min && level <= max)
            .map(|&level| {
                let label = format!("{:.1}", level.abs());
                let label_width = guide_label_width(&label);
                max_label_width = max_label_width.max(label_width);
                GuideLine {
                    value_loudness: level,
                    color: guide_color,
                    length: GUIDE_LINE_LENGTH,
                    thickness: GUIDE_LINE_THICKNESS,
                    label: Some(label),
                    label_width,
                }
            })
            .collect();

        VisualParams {
            min_loudness: min,
            max_loudness: max,
            channels,
            channel_gap_fraction: CHANNEL_GAP_FRACTION,
            channel_width_scale: CHANNEL_WIDTH_SCALE,
            short_term_value: right_value,
            value_unit: self.right_mode.unit_label().to_string(),
            guides,
            left_padding: GUIDE_LABEL_MARGIN
                + max_label_width
                + GUIDE_LABEL_GAP
                + GUIDE_LINE_LENGTH
                + GUIDE_LINE_PADDING,
            right_padding: LIVE_VALUE_LABEL_RESERVE,
            guide_padding: GUIDE_LINE_PADDING,
            value_scale_bias: VALUE_SCALE_MIX,
            vertical_padding: METER_VERTICAL_PADDING,
        }
    }
}

#[derive(Debug)]
pub struct LoudnessMeter<'a> {
    state: &'a LoudnessMeterState,
    range: Option<(f32, f32)>,
    height: f32,
    width: f32,
    fill_height: bool,
}

impl<'a> LoudnessMeter<'a> {
    pub fn new(state: &'a LoudnessMeterState) -> Self {
        Self {
            state,
            range: None,
            height: DEFAULT_HEIGHT,
            width: DEFAULT_WIDTH,
            fill_height: false,
        }
    }

    pub fn with_range(mut self, min: f32, max: f32) -> Self {
        self.range = Some((min, max));
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
        self.range.unwrap_or_else(|| self.state.range())
    }

    fn total_width(&self) -> f32 {
        self.width + LIVE_VALUE_LABEL_RESERVE
    }
}

impl<'a, Message> Widget<Message, Theme, iced::Renderer> for LoudnessMeter<'a> {
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<GuideLabelCache>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(GuideLabelCache::default())
    }

    fn size(&self) -> Size<Length> {
        Size::new(
            Length::Fixed(self.total_width()),
            if self.fill_height {
                Length::Fill
            } else {
                Length::Fixed(self.height)
            },
        )
    }

    fn layout(
        &self,
        _tree: &mut Tree,
        _renderer: &iced::Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        layout::Node::new(limits.resolve(
            Length::Fixed(self.total_width()),
            if self.fill_height {
                Length::Fill
            } else {
                Length::Fixed(self.height)
            },
            Size::new(0.0, self.height),
        ))
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut iced::Renderer,
        theme: &Theme,
        _style: &renderer::Style,
        layout: Layout<'_>,
        _cursor: mouse::Cursor,
        _viewport: &Rectangle,
    ) {
        let bounds = layout.bounds();
        let (min, max) = self.active_range();
        let params = self.state.visual_params((min, max), theme);
        renderer.draw_primitive(bounds, LoudnessMeterPrimitive::new(params.clone()));
        draw_guide_labels(tree, renderer, bounds, &params, theme);
        draw_live_value_label(tree, renderer, bounds, &params, theme);
    }

    fn children(&self) -> Vec<Tree> {
        Vec::new()
    }

    fn diff(&self, _tree: &mut Tree) {}
}

/// Convenience conversion into an [`iced::Element`].
pub fn widget_with_layout<'a, Message>(
    state: &'a LoudnessMeterState,
    preferred_width: f32,
    preferred_height: f32,
) -> Element<'a, Message>
where
    Message: 'a,
{
    let (min, max) = state.range();
    Element::new(
        LoudnessMeter::new(state)
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
            label: label.to_string(),
            bounds,
            paragraph: build_paragraph(label, bounds, GUIDE_LABEL_FONT_SIZE, false),
        }
    }

    fn ensure(&mut self, bounds: Size) {
        if !size_eq(self.bounds, bounds) {
            self.bounds = bounds;
            self.paragraph = build_paragraph(&self.label, bounds, GUIDE_LABEL_FONT_SIZE, false);
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
            label: label.to_string(),
            bounds,
            paragraph: build_paragraph(label, bounds, LIVE_VALUE_LABEL_FONT_SIZE, true),
        }
    }

    fn ensure(&mut self, label: &str, bounds: Size) {
        if self.label != label || !size_eq(self.bounds, bounds) {
            self.label = label.to_string();
            self.bounds = bounds;
            self.paragraph = build_paragraph(label, bounds, LIVE_VALUE_LABEL_FONT_SIZE, true);
        }
    }
}
fn ensure_label_paragraph(entries: &mut Vec<CachedParagraph>, label: &str, bounds: Size) -> usize {
    if let Some(index) = entries.iter().position(|entry| entry.label == label) {
        entries[index].ensure(bounds);
        index
    } else {
        entries.push(CachedParagraph::new(label, bounds));
        entries.len() - 1
    }
}

fn prune_label_paragraphs(entries: &mut Vec<CachedParagraph>, active: &[&str]) {
    entries.retain(|entry| active.contains(&entry.label.as_str()));
}

fn build_paragraph(label: &str, bounds: Size, font_size: f32, bold: bool) -> RenderParagraph {
    RenderParagraph::with_text(Text {
        content: label,
        bounds,
        size: Pixels(font_size),
        line_height: LineHeight::Relative(1.0),
        font: if bold {
            Font {
                weight: FontWeight::Bold,
                ..Font::DEFAULT
            }
        } else {
            Font::default()
        },
        horizontal_alignment: if bold {
            Horizontal::Center
        } else {
            Horizontal::Left
        },
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
    theme: &Theme,
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

    let label_color = guide_label_color(theme);
    let mut active_labels = Vec::with_capacity(params.guides.len());

    for guide in &params.guides {
        let Some(ref label_text) = guide.label else {
            continue;
        };

        let ratio = params.clamp_ratio(guide.value_loudness);
        let center_y = context.center(ratio);
        let label_bounds = Size::new(guide.label_width, GUIDE_LABEL_HEIGHT);
        let index = ensure_label_paragraph(&mut entries, label_text, label_bounds);
        let text_bounds = entries[index].paragraph.min_bounds();
        let (clip_bounds, position) =
            context.label_clip(center_y, GUIDE_LABEL_HEIGHT, text_bounds.width);

        renderer.fill_paragraph(
            &entries[index].paragraph,
            position,
            label_color,
            clip_bounds,
        );
        active_labels.push(label_text.as_str());
    }

    prune_label_paragraphs(&mut entries, &active_labels);
}

fn draw_live_value_label(
    tree: &Tree,
    renderer: &mut iced::Renderer,
    bounds: Rectangle,
    params: &VisualParams,
    theme: &Theme,
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

    let label = format!("{:.1} {}", params.short_term_value, params.value_unit);
    let available_width = context.available_width();
    let minimum_width = live_label_min_width(&label);

    if available_width < minimum_width {
        live_entry.take();
        return;
    }

    let label_width = minimum_width.min(available_width);
    let label_bounds = Size::new(label_width, LIVE_VALUE_LABEL_HEIGHT);

    if let Some(entry) = live_entry.as_mut() {
        entry.ensure(&label, label_bounds);
    } else {
        *live_entry = Some(LiveLabelParagraph::new(&label, label_bounds));
    }

    let Some(entry) = live_entry.as_ref() else {
        return;
    };

    let text_width = entry.paragraph.min_bounds().width;
    if text_width > label_width + f32::EPSILON {
        live_entry.take();
        return;
    }

    let (clip_bounds, position) = context.clip(LIVE_VALUE_LABEL_HEIGHT, label_width);

    renderer.fill_quad(
        Quad {
            bounds: clip_bounds,
            border: Border::default(),
            shadow: Default::default(),
        },
        Background::Color(live_label_background(theme)),
    );

    renderer.fill_paragraph(
        &entry.paragraph,
        position,
        theme.extended_palette().background.base.text,
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
        let mut state = LoudnessMeterState::new();
        state.apply_snapshot(&LoudnessSnapshot {
            short_term_loudness: [-12.0, -6.0],
            momentary_loudness: [-10.0, -5.0],
            rms_fast_db: [-15.0, -9.0],
            rms_slow_db: [-14.0, -8.0],
            true_peak_db: [-1.0, -3.0],
        });

        assert!((state.short_term_average() + 9.0).abs() < f32::EPSILON);
        assert_eq!(state.true_peak_db[0], -1.0);
        assert_eq!(state.true_peak_db[1], -3.0);

        let theme = theme::theme(None);
        let params = state.visual_params(state.range(), &theme);
        assert_eq!(params.channels.len(), 2);
        assert_eq!(params.channels[0].fills.len(), 2);
        assert_eq!(params.channels[1].fills.len(), 1);
        let left_value_0 = params.channels[0].fills[0].value_loudness;
        let left_value_1 = params.channels[0].fills[1].value_loudness;
        let right_value = params.channels[1].fills[0].value_loudness;
        assert!((left_value_0 + 1.0).abs() < f32::EPSILON);
        assert!((left_value_1 + 3.0).abs() < f32::EPSILON);
        assert!((right_value + 9.0).abs() < f32::EPSILON);
        assert!((params.short_term_value + 9.0).abs() < f32::EPSILON);
        assert_eq!(params.value_unit, "LUFS");
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
