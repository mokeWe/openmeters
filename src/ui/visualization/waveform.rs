//! UI wrapper around the scrolling waveform DSP processor and renderer.

use crate::audio::meter_tap::MeterFormat;
use crate::dsp::waveform::{
    DEFAULT_COLUMN_CAPACITY, MAX_COLUMN_CAPACITY, WaveformConfig,
    WaveformProcessor as CoreWaveformProcessor, WaveformSnapshot,
};
use crate::dsp::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::ui::render::waveform::{WaveformParams, WaveformPrimitive};
use crate::ui::theme;
use iced::advanced::Renderer as _;
use iced::advanced::renderer::{self, Quad};
use iced::advanced::widget::{Tree, tree};
use iced::advanced::{Layout, Widget, layout, mouse};
use iced::{Background, Color, Element, Length, Rectangle, Size};
use iced_wgpu::primitive::Renderer as _;
use std::cell::Cell;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

const COLUMN_PIXEL_WIDTH: f32 = 2.0;
const DEFAULT_FILL_ALPHA: f32 = 1.0;
const DEFAULT_LINE_ALPHA: f32 = 1.0;
const DEFAULT_VERTICAL_PADDING: f32 = 8.0;
const DEFAULT_CHANNEL_GAP: f32 = 12.0;
const DEFAULT_AMPLITUDE_SCALE: f32 = 1.0;
const DEFAULT_STROKE_WIDTH: f32 = 1.0;

#[derive(Debug, Clone)]
pub struct WaveformProcessor {
    inner: CoreWaveformProcessor,
    channels: usize,
}

impl WaveformProcessor {
    pub fn new(sample_rate: f32) -> Self {
        let config = WaveformConfig {
            sample_rate,
            ..WaveformConfig::default()
        };
        Self {
            inner: CoreWaveformProcessor::new(config),
            channels: 2,
        }
    }

    pub fn ingest(&mut self, samples: &[f32], format: MeterFormat) -> WaveformSnapshot {
        if samples.is_empty() {
            return self.inner.snapshot().clone();
        }

        let channels = format.channels.max(1);
        self.channels = channels;

        let sample_rate = format.sample_rate.max(1.0);
        let mut config = self.inner.config();
        if (config.sample_rate - sample_rate).abs() > f32::EPSILON {
            config.sample_rate = sample_rate;
            self.inner.update_config(config);
        }

        let block = AudioBlock::new(samples, channels, sample_rate, Instant::now());

        match self.inner.process_block(&block) {
            ProcessorUpdate::Snapshot(snapshot) => snapshot,
            ProcessorUpdate::None => self.inner.snapshot().clone(),
        }
    }

    pub fn update_config(&mut self, config: WaveformConfig) {
        self.inner.update_config(config);
    }

    pub fn config(&self) -> WaveformConfig {
        self.inner.config()
    }
}

#[derive(Debug)]
struct PresentationData {
    columns: usize,
    column_width: f32,
    min_values: Vec<f32>,
    max_values: Vec<f32>,
    frequency: Vec<f32>,
    preview_min: Vec<f32>,
    preview_max: Vec<f32>,
    preview_frequency: f32,
    preview_progress: f32,
}

#[derive(Debug, Clone)]
pub struct WaveformState {
    snapshot: WaveformSnapshot,
    preview_min: Vec<f32>,
    preview_max: Vec<f32>,
    preview_frequency: f32,
    preview_progress: f32,
    style: WaveformStyle,
    desired_columns: Rc<Cell<usize>>,
    frequency_hint: f32,
    render_key: u64,
}

impl WaveformState {
    fn next_render_key() -> u64 {
        static NEXT_RENDER_KEY: AtomicU64 = AtomicU64::new(1);
        NEXT_RENDER_KEY.fetch_add(1, Ordering::Relaxed)
    }

    pub fn new() -> Self {
        Self {
            snapshot: WaveformSnapshot::default(),
            preview_min: Vec::new(),
            preview_max: Vec::new(),
            preview_frequency: 0.0,
            preview_progress: 0.0,
            style: WaveformStyle::default(),
            desired_columns: Rc::new(Cell::new(DEFAULT_COLUMN_CAPACITY)),
            frequency_hint: 0.0,
            render_key: Self::next_render_key(),
        }
    }

    pub fn apply_snapshot(&mut self, snapshot: WaveformSnapshot) {
        self.snapshot = snapshot;

        self.update_frequency_hint();
        self.update_preview_state();
    }

    pub fn visual(&self, bounds: Rectangle) -> Option<WaveformVisual> {
        let presentation = self.build_presentation(bounds.width)?;

        let channels = self.snapshot.channels.max(1);
        let mut colors = Vec::with_capacity(presentation.columns);
        for &value in &presentation.frequency {
            let color = self.style.color_for_frequency(value);
            colors.push(theme::color_to_rgba(color));
        }
        let preview_color = self
            .style
            .color_for_frequency(presentation.preview_frequency.min(1.0));

        let params = WaveformParams {
            bounds,
            channels,
            column_width: presentation.column_width,
            columns: presentation.columns,
            min_values: presentation.min_values,
            max_values: presentation.max_values,
            colors,
            preview_min: presentation.preview_min,
            preview_max: presentation.preview_max,
            preview_color: theme::color_to_rgba(preview_color),
            preview_progress: presentation.preview_progress,
            fill_alpha: self.style.fill_alpha,
            line_alpha: self.style.line_alpha,
            vertical_padding: self.style.vertical_padding,
            channel_gap: self.style.channel_gap,
            amplitude_scale: self.style.amplitude_scale,
            stroke_width: self.style.stroke_width,
            instance_key: self.render_key,
        };

        Some(WaveformVisual { primitive: params })
    }

    fn build_presentation(&self, width: f32) -> Option<PresentationData> {
        let channels = self.snapshot.channels.max(1);
        if width <= 0.0 {
            return None;
        }

        let mut required = (width / COLUMN_PIXEL_WIDTH).ceil() as usize;
        if required == 0 {
            required = 1;
        }
        self.desired_columns
            .set(required.clamp(1, MAX_COLUMN_CAPACITY));

        let columns = self.snapshot.columns;
        if columns == 0 {
            return None;
        }

        let mut visible = required;
        if visible == 0 {
            return None;
        }
        visible = visible.min(columns);

        let start = columns.saturating_sub(visible);
        let column_width = COLUMN_PIXEL_WIDTH;

        let mut min_values = vec![0.0; visible * channels];
        let mut max_values = vec![0.0; visible * channels];
        let min_source = self.snapshot.min_values.as_ref();
        let max_source = self.snapshot.max_values.as_ref();
        for channel in 0..channels {
            let src_base = channel * columns + start;
            let dest_base = channel * visible;
            min_values[dest_base..dest_base + visible]
                .copy_from_slice(&min_source[src_base..src_base + visible]);
            max_values[dest_base..dest_base + visible]
                .copy_from_slice(&max_source[src_base..src_base + visible]);
        }

        let mut frequency = vec![0.0; visible];
        let freq_source = self.snapshot.frequency_normalized.as_ref();
        frequency.copy_from_slice(&freq_source[start..start + visible]);

        let preview_active = self.preview_progress > 0.0;
        let preview_min = if preview_active {
            let mut values = self.preview_min.clone();
            if values.len() < channels {
                values.resize(channels, 0.0);
            }
            values
        } else {
            Vec::new()
        };
        let preview_max = if preview_active {
            let mut values = self.preview_max.clone();
            if values.len() < channels {
                values.resize(channels, 0.0);
            }
            values
        } else {
            Vec::new()
        };
        let preview_progress = if preview_active { 1.0 } else { 0.0 };

        Some(PresentationData {
            columns: visible,
            column_width,
            min_values,
            max_values,
            frequency,
            preview_min,
            preview_max,
            preview_frequency: self.preview_frequency,
            preview_progress,
        })
    }

    fn update_preview_state(&mut self) {
        let channels = self.snapshot.channels.max(1);
        ensure_len(&mut self.preview_min, channels);
        ensure_len(&mut self.preview_max, channels);

        let progress = self.snapshot.preview.progress.clamp(0.0, 1.0);
        if progress > 0.0
            && self.snapshot.preview.min_values.len() >= channels
            && self.snapshot.preview.max_values.len() >= channels
        {
            for channel in 0..channels {
                self.preview_min[channel] = self
                    .snapshot
                    .preview
                    .min_values
                    .get(channel)
                    .copied()
                    .unwrap_or(0.0);
                self.preview_max[channel] = self
                    .snapshot
                    .preview
                    .max_values
                    .get(channel)
                    .copied()
                    .unwrap_or(0.0);
            }
            self.preview_frequency = self.snapshot.preview.frequency_normalized;
            self.preview_progress = progress;
        } else {
            self.preview_min.fill(0.0);
            self.preview_max.fill(0.0);
            self.preview_frequency = self.frequency_hint;
            self.preview_progress = 0.0;
        }
    }

    fn update_frequency_hint(&mut self) {
        if let Some(freq) = self.snapshot.frequency_normalized.as_ref().last().copied() {
            self.frequency_hint = freq;
        } else {
            self.frequency_hint = self.snapshot.preview.frequency_normalized;
        }
    }

    #[allow(dead_code)]
    pub fn style_mut(&mut self) -> &mut WaveformStyle {
        &mut self.style
    }

    pub fn desired_columns(&self) -> usize {
        self.desired_columns.get()
    }
}

#[derive(Debug, Clone)]
pub struct WaveformStyle {
    pub background: Color,
    pub fill_alpha: f32,
    pub line_alpha: f32,
    pub vertical_padding: f32,
    pub channel_gap: f32,
    pub amplitude_scale: f32,
    pub stroke_width: f32,
    gradient: Vec<GradientStop>,
}

impl WaveformStyle {
    fn color_for_frequency(&self, value: f32) -> Color {
        if self.gradient.is_empty() {
            return theme::waveform_palette()
                .first()
                .copied()
                .unwrap_or_else(theme::accent_primary);
        }

        let clamped = value.clamp(0.0, 1.0);
        for window in self.gradient.windows(2) {
            let [start, end] = window else {
                continue;
            };
            if clamped <= end.position {
                let span = (end.position - start.position).max(f32::EPSILON);
                let alpha = (clamped - start.position).clamp(0.0, span) / span;
                return lerp_color(start.color, end.color, alpha);
            }
        }

        self.gradient
            .last()
            .map(|stop| stop.color)
            .unwrap_or_else(|| {
                theme::waveform_palette()
                    .last()
                    .copied()
                    .unwrap_or_else(theme::accent_primary)
            })
    }
}

impl Default for WaveformStyle {
    fn default() -> Self {
        let background = theme::with_alpha(theme::base_color(), 0.0);
        let palette = theme::waveform_palette();
        let gradient = if palette.len() <= 1 {
            palette
                .first()
                .copied()
                .map(|color| {
                    vec![GradientStop {
                        position: 0.0,
                        color,
                    }]
                })
                .unwrap_or_default()
        } else {
            let last_index = (palette.len() - 1) as f32;
            palette
                .iter()
                .enumerate()
                .map(|(index, color)| GradientStop {
                    position: index as f32 / last_index,
                    color: *color,
                })
                .collect()
        };

        Self {
            background,
            fill_alpha: DEFAULT_FILL_ALPHA,
            line_alpha: DEFAULT_LINE_ALPHA,
            vertical_padding: DEFAULT_VERTICAL_PADDING,
            channel_gap: DEFAULT_CHANNEL_GAP,
            amplitude_scale: DEFAULT_AMPLITUDE_SCALE,
            stroke_width: DEFAULT_STROKE_WIDTH,
            gradient,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct GradientStop {
    position: f32,
    color: Color,
}

#[derive(Debug, Clone)]
pub struct WaveformVisual {
    pub primitive: WaveformParams,
}

#[derive(Debug)]
pub struct Waveform<'a> {
    state: &'a WaveformState,
}

impl<'a> Waveform<'a> {
    pub fn new(state: &'a WaveformState) -> Self {
        Self { state }
    }
}

impl<'a, Message> Widget<Message, iced::Theme, iced::Renderer> for Waveform<'a> {
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
        renderer.fill_quad(
            Quad {
                bounds,
                border: Default::default(),
                shadow: Default::default(),
            },
            Background::Color(self.state.style.background),
        );

        if let Some(visual) = self.state.visual(bounds) {
            renderer.draw_primitive(bounds, WaveformPrimitive::new(visual.primitive));
        }
    }

    fn children(&self) -> Vec<Tree> {
        Vec::new()
    }

    fn diff(&self, _tree: &mut Tree) {}
}

pub fn widget<'a, Message>(state: &'a WaveformState) -> Element<'a, Message>
where
    Message: 'a,
{
    Element::new(Waveform::new(state))
}

fn ensure_len(vec: &mut Vec<f32>, len: usize) {
    if vec.len() != len {
        vec.resize(len, 0.0);
    }
}

fn lerp_color(a: Color, b: Color, alpha: f32) -> Color {
    Color::from_rgba(
        a.r + (b.r - a.r) * alpha,
        a.g + (b.g - a.g) * alpha,
        a.b + (b.b - a.b) * alpha,
        a.a + (b.a - a.a) * alpha,
    )
}
