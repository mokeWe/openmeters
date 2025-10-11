//! UI wrapper around the oscilloscope DSP processor and renderer.

use crate::audio::meter_tap::MeterFormat;
use crate::dsp::oscilloscope::{
    OscilloscopeConfig, OscilloscopeProcessor as CoreOscilloscopeProcessor, OscilloscopeSnapshot,
};
use crate::dsp::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::ui::render::oscilloscope::{OscilloscopeParams, OscilloscopePrimitive};
use crate::ui::theme;
use iced::advanced::Renderer as _;
use iced::advanced::renderer::{self, Quad};
use iced::advanced::widget::{Tree, tree};
use iced::advanced::{Layout, Widget, layout, mouse};
use iced::{Background, Color, Element, Length, Rectangle, Size};
use iced_wgpu::primitive::Renderer as _;
use std::time::Instant;

const DEFAULT_CHANNELS: usize = 2;
const DEFAULT_LINE_ALPHA: f32 = 0.92;
const DEFAULT_FADE_BASE: f32 = 0.25;
const DEFAULT_VERTICAL_PADDING: f32 = 8.0;
const DEFAULT_CHANNEL_GAP: f32 = 12.0;
const DEFAULT_AMPLITUDE_SCALE: f32 = 0.9;
const DEFAULT_STROKE_WIDTH: f32 = 2.0;

#[derive(Debug, Clone)]
pub struct OscilloscopeProcessor {
    inner: CoreOscilloscopeProcessor,
    channels: usize,
}

impl OscilloscopeProcessor {
    pub fn new(sample_rate: f32) -> Self {
        let config = OscilloscopeConfig {
            sample_rate,
            ..Default::default()
        };
        Self {
            inner: CoreOscilloscopeProcessor::new(config),
            channels: DEFAULT_CHANNELS,
        }
    }

    pub fn ingest(&mut self, samples: &[f32], format: MeterFormat) -> OscilloscopeSnapshot {
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

    #[allow(dead_code)]
    pub fn update_config(&mut self, config: OscilloscopeConfig) {
        self.inner.update_config(config);
    }

    pub fn config(&self) -> OscilloscopeConfig {
        self.inner.config()
    }
}

#[derive(Debug, Clone)]
pub struct OscilloscopeState {
    snapshot: OscilloscopeSnapshot,
    style: OscilloscopeStyle,
}

impl OscilloscopeState {
    pub fn new() -> Self {
        Self {
            snapshot: OscilloscopeSnapshot::default(),
            style: OscilloscopeStyle::default(),
        }
    }

    pub fn apply_snapshot(&mut self, snapshot: &OscilloscopeSnapshot) {
        self.snapshot = snapshot.clone();
    }

    #[allow(dead_code)]
    pub fn style_mut(&mut self) -> &mut OscilloscopeStyle {
        &mut self.style
    }

    pub fn visual_params(&self, bounds: Rectangle) -> OscilloscopeParams {
        let channels = self.snapshot.channels.max(1);
        let samples_per_channel = self.snapshot.samples_per_channel.max(1);

        let mut channel_palette = if channels <= self.style.channel_colors.len() {
            self.style.channel_colors[..channels].to_vec()
        } else {
            self.style
                .channel_colors
                .repeat(channels.div_ceil(self.style.channel_colors.len()))
        };
        channel_palette.truncate(channels);

        let colors = channel_palette
            .into_iter()
            .map(theme::color_to_rgba)
            .collect();

        OscilloscopeParams {
            bounds,
            channels,
            samples_per_channel,
            samples: self.snapshot.samples.clone(),
            colors,
            line_alpha: self.style.line_alpha,
            fade_alpha: (1.0 - self.snapshot.persistence.clamp(0.0, 1.0)) * DEFAULT_FADE_BASE,
            vertical_padding: self.style.vertical_padding,
            channel_gap: self.style.channel_gap,
            amplitude_scale: self.style.amplitude_scale,
            stroke_width: self.style.stroke_width,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OscilloscopeStyle {
    pub background: Color,
    pub channel_colors: Vec<Color>,
    pub line_alpha: f32,
    pub vertical_padding: f32,
    pub channel_gap: f32,
    pub amplitude_scale: f32,
    pub stroke_width: f32,
}

impl Default for OscilloscopeStyle {
    fn default() -> Self {
        let background = theme::with_alpha(theme::base_color(), 0.0);
        let primary = theme::accent_primary();
        let success = theme::accent_success();
        let danger = theme::accent_danger();
        let text = theme::text_color();
        let hover = theme::hover_color();

        let channel_colors = vec![
            theme::mix_colors(primary, text, 0.35),
            theme::mix_colors(success, text, 0.25),
            theme::mix_colors(danger, text, 0.20),
            theme::mix_colors(primary, hover, 0.55),
        ];

        Self {
            background,
            channel_colors,
            line_alpha: DEFAULT_LINE_ALPHA,
            vertical_padding: DEFAULT_VERTICAL_PADDING,
            channel_gap: DEFAULT_CHANNEL_GAP,
            amplitude_scale: DEFAULT_AMPLITUDE_SCALE,
            stroke_width: DEFAULT_STROKE_WIDTH,
        }
    }
}

#[derive(Debug)]
pub struct Oscilloscope<'a> {
    state: &'a OscilloscopeState,
}

impl<'a> Oscilloscope<'a> {
    pub fn new(state: &'a OscilloscopeState) -> Self {
        Self { state }
    }
}

impl<'a, Message> Widget<Message, iced::Theme, iced::Renderer> for Oscilloscope<'a> {
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
        let params = self.state.visual_params(bounds);
        if params.fade_alpha > f32::EPSILON {
            renderer.fill_quad(
                Quad {
                    bounds,
                    border: Default::default(),
                    shadow: Default::default(),
                },
                Background::Color(background),
            );
        }
        renderer.draw_primitive(bounds, OscilloscopePrimitive::new(params));
    }

    fn children(&self) -> Vec<Tree> {
        Vec::new()
    }

    fn diff(&self, _tree: &mut Tree) {}
}

pub fn widget<'a, Message>(state: &'a OscilloscopeState) -> Element<'a, Message>
where
    Message: 'a,
{
    Element::new(Oscilloscope::new(state))
}
