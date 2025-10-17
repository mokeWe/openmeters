//! UI wrapper around the oscilloscope DSP processor and renderer.

use crate::audio::meter_tap::MeterFormat;
use crate::dsp::oscilloscope::{
    DisplayMode, OscilloscopeConfig, OscilloscopeProcessor as CoreOscilloscopeProcessor,
    OscilloscopeSnapshot,
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

#[derive(Debug, Clone)]
pub struct OscilloscopeProcessor {
    inner: CoreOscilloscopeProcessor,
}

impl OscilloscopeProcessor {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            inner: CoreOscilloscopeProcessor::new(OscilloscopeConfig {
                sample_rate,
                ..Default::default()
            }),
        }
    }

    pub fn ingest(&mut self, samples: &[f32], format: MeterFormat) -> OscilloscopeSnapshot {
        if samples.is_empty() {
            return self.inner.snapshot().clone();
        }

        let sample_rate = format.sample_rate.max(1.0);
        let mut config = self.inner.config();
        if (config.sample_rate - sample_rate).abs() > f32::EPSILON {
            config.sample_rate = sample_rate;
            self.update_config(config);
        }

        let block = AudioBlock::new(samples, format.channels.max(1), sample_rate, Instant::now());

        match self.inner.process_block(&block) {
            ProcessorUpdate::Snapshot(snapshot) => snapshot,
            ProcessorUpdate::None => self.inner.snapshot().clone(),
        }
    }

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
    display_samples: Vec<f32>,
    fade_alpha: f32,
}

impl OscilloscopeState {
    pub fn new() -> Self {
        Self {
            snapshot: OscilloscopeSnapshot::default(),
            style: OscilloscopeStyle::default(),
            display_samples: Vec::new(),
            fade_alpha: 1.0,
        }
    }

    pub fn apply_snapshot(&mut self, snapshot: &OscilloscopeSnapshot) {
        if snapshot.samples.is_empty() {
            self.snapshot = snapshot.clone();
            self.display_samples.clear();
            self.fade_alpha = 1.0;
            return;
        }

        let mut persistence = snapshot.persistence.clamp(0.0, 0.98);
        if snapshot.display_mode == DisplayMode::XY {
            // XY persistence that is too aggressive makes the trace unreadable.
            persistence = persistence.min(0.9);
        }

        if self.display_samples.len() != snapshot.samples.len() {
            self.display_samples = snapshot.samples.clone();
        } else {
            let fresh = 1.0 - persistence;
            for (current, incoming) in self
                .display_samples
                .iter_mut()
                .zip(snapshot.samples.iter())
            {
                *current = (*current * persistence) + (*incoming * fresh);
            }
        }

        self.fade_alpha = (1.0 - persistence).sqrt().clamp(0.05, 1.0);

        self.snapshot = snapshot.clone();
        self.snapshot.samples.clear();
        self.snapshot
            .samples
            .extend_from_slice(&self.display_samples);
    }

    fn visual_params(&self, bounds: Rectangle) -> OscilloscopeParams {
        let channels = self.snapshot.channels.max(1);
        let colors = self
            .style
            .channel_colors
            .iter()
            .cycle()
            .take(channels)
            .map(|c| theme::color_to_rgba(*c))
            .collect();

        OscilloscopeParams {
            bounds,
            channels,
            samples_per_channel: self.snapshot.samples_per_channel,
            samples: self.snapshot.samples.clone(),
            colors,
            line_alpha: self.style.line_alpha,
            fade_alpha: self.fade_alpha,
            vertical_padding: self.style.vertical_padding,
            channel_gap: self.style.channel_gap,
            amplitude_scale: self.style.amplitude_scale,
            stroke_width: self.style.stroke_width,
            display_mode: self.snapshot.display_mode,
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
        let primary = theme::accent_primary();
        let success = theme::accent_success();
        let danger = theme::accent_danger();
        let text = theme::text_color();
        let hover = theme::hover_color();

        Self {
            background: theme::base_color(),
            channel_colors: vec![
                theme::mix_colors(primary, text, 0.35),
                theme::mix_colors(success, text, 0.25),
                theme::mix_colors(danger, text, 0.20),
                theme::mix_colors(primary, hover, 0.55),
            ],
            line_alpha: 0.92,
            vertical_padding: 8.0,
            channel_gap: 12.0,
            amplitude_scale: 0.9,
            stroke_width: 1.0,
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
        let params = self.state.visual_params(bounds);
        let background = self.state.style.background;

        if params.fade_alpha >= 0.999 {
            let mut clear_color = background;
            clear_color.a = 1.0;
            renderer.fill_quad(
                Quad {
                    bounds,
                    border: Default::default(),
                    shadow: Default::default(),
                },
                Background::Color(clear_color),
            );
        } else if params.fade_alpha > f32::EPSILON {
            let mut fade_color = background;
            fade_color.a = params.fade_alpha.clamp(0.0, 1.0);
            renderer.fill_quad(
                Quad {
                    bounds,
                    border: Default::default(),
                    shadow: Default::default(),
                },
                Background::Color(fade_color),
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
