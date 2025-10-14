//! UI wrapper around the scrolling waveform DSP processor and renderer.

use crate::audio::meter_tap::MeterFormat;
use crate::dsp::waveform::{
    WaveformConfig, WaveformProcessor as CoreWaveformProcessor, WaveformSnapshot,
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
use std::time::Instant;

const DEFAULT_LINE_ALPHA: f32 = 1.0;
const DEFAULT_VERTICAL_PADDING: f32 = 8.0;
const DEFAULT_CHANNEL_GAP: f32 = 12.0;
const DEFAULT_AMPLITUDE_SCALE: f32 = 0.9;
const DEFAULT_STROKE_WIDTH: f32 = 1.0;

#[derive(Debug, Clone)]
pub struct WaveformProcessor {
    inner: CoreWaveformProcessor,
    channels: usize,
}

impl WaveformProcessor {
    pub fn new(sample_rate: f32) -> Self {
        let mut config = WaveformConfig::default();
        config.sample_rate = sample_rate;
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

#[derive(Debug, Clone)]
pub struct WaveformState {
    snapshot: WaveformSnapshot,
    style: WaveformStyle,
    frequency_hint: f32,
}

impl WaveformState {
    pub fn new() -> Self {
        Self {
            snapshot: WaveformSnapshot::default(),
            style: WaveformStyle::default(),
            frequency_hint: 0.0,
        }
    }

    pub fn apply_snapshot(&mut self, snapshot: &WaveformSnapshot) {
        self.snapshot = snapshot.clone();

        self.update_frequency_hint();
    }

    pub fn visual(&self, bounds: Rectangle) -> Option<WaveformVisual> {
        if bounds.width <= 0.0 || bounds.height <= 0.0 {
            return None;
        }

        let channels = self.snapshot.channels.max(1);
        let frames = self.snapshot.frames;
        if frames < 2 {
            return None;
        }

        let expected_samples = frames.saturating_mul(channels);
        if self.snapshot.samples.len() < expected_samples {
            return None;
        }

        let mut samples = self.snapshot.samples.clone();
        samples.truncate(expected_samples);

        let mut colors = Vec::with_capacity(frames);
        let mut frequencies = self.snapshot.frequency_normalized.iter().copied();
        for _ in 0..frames {
            let frequency = frequencies.next().unwrap_or(self.frequency_hint);
            let color = self.style.color_for_frequency(frequency);
            colors.push(theme::color_to_rgba(color));
        }

        let params = WaveformParams {
            bounds,
            channels,
            frames,
            samples,
            colors,
            line_alpha: self.style.line_alpha,
            vertical_padding: self.style.vertical_padding,
            channel_gap: self.style.channel_gap,
            amplitude_scale: self.style.amplitude_scale,
            stroke_width: self.style.stroke_width,
        };

        Some(WaveformVisual { primitive: params })
    }

    fn update_frequency_hint(&mut self) {
        if let Some(freq) = self.snapshot.frequency_normalized.last().copied() {
            self.frequency_hint = freq;
        }
    }

    #[allow(dead_code)]
    pub fn style_mut(&mut self) -> &mut WaveformStyle {
        &mut self.style
    }
}

#[derive(Debug, Clone)]
pub struct WaveformStyle {
    pub background: Color,
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

fn lerp_color(a: Color, b: Color, alpha: f32) -> Color {
    Color::from_rgba(
        a.r + (b.r - a.r) * alpha,
        a.g + (b.g - a.g) * alpha,
        a.b + (b.b - a.b) * alpha,
        a.a + (b.a - a.a) * alpha,
    )
}
