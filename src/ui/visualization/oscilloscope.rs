//! UI wrapper around the oscilloscope DSP processor and renderer.

use crate::audio::meter_tap::MeterFormat;
use crate::dsp::oscilloscope::{
    OscilloscopeConfig, OscilloscopeProcessor as CoreOscilloscopeProcessor, OscilloscopeSnapshot,
};
use crate::dsp::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::ui::render::oscilloscope::{OscilloscopeParams, OscilloscopePrimitive};
use crate::ui::settings::OscilloscopeSettings;
use crate::ui::theme;
use iced::advanced::Renderer as _;
use iced::advanced::renderer::{self, Quad};
use iced::advanced::widget::{Tree, tree};
use iced::advanced::{Layout, Widget, layout, mouse};
use iced::{Background, Color, Element, Length, Rectangle, Size};
use iced_wgpu::primitive::Renderer as _;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

static NEXT_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

fn next_instance_id() -> u64 {
    NEXT_INSTANCE_ID.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DisplayMode {
    LR,
    XY,
}

impl Default for DisplayMode {
    fn default() -> Self {
        Self::LR
    }
}

impl DisplayMode {
    pub const ALL: [Self; 2] = [Self::LR, Self::XY];

    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LR => "LR",
            Self::XY => "XY",
        }
    }
}

impl fmt::Display for DisplayMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

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
    persistence: f32,
    display_mode: DisplayMode,
    instance_id: u64,
}

impl OscilloscopeState {
    pub fn new() -> Self {
        Self {
            snapshot: OscilloscopeSnapshot::default(),
            style: OscilloscopeStyle::default(),
            display_samples: Vec::new(),
            fade_alpha: 1.0,
            persistence: 0.85,
            display_mode: DisplayMode::default(),
            instance_id: next_instance_id(),
        }
    }

    pub fn update_view_settings(&mut self, settings: &OscilloscopeSettings) {
        self.persistence = settings.persistence.clamp(0.0, 1.0);
        self.display_mode = settings.display_mode;
        self.recompute_fade_alpha();
    }

    pub fn apply_snapshot(&mut self, snapshot: &OscilloscopeSnapshot) {
        if snapshot.samples.is_empty() {
            self.snapshot = snapshot.clone();
            self.display_samples.clear();
            self.fade_alpha = 1.0_f32;
            return;
        }

        let total_samples = snapshot.samples.len();
        if self.display_samples.len() != total_samples {
            self.display_samples.resize(total_samples, 0.0_f32);
        }

        let persistence = self.effective_persistence_for(snapshot.channels);
        if persistence <= f32::EPSILON {
            self.display_samples.copy_from_slice(&snapshot.samples);
        } else {
            let fresh = 1.0_f32 - persistence;
            for (current, incoming) in self.display_samples.iter_mut().zip(snapshot.samples.iter())
            {
                *current = (*current * persistence) + (*incoming * fresh);
            }
        }

        self.snapshot = snapshot.clone();
        self.snapshot.samples.clear();
        self.snapshot
            .samples
            .extend_from_slice(&self.display_samples);
        self.recompute_fade_alpha();
    }

    pub fn view_settings(&self) -> (f32, DisplayMode) {
        (self.persistence, self.display_mode)
    }

    fn effective_display_mode(&self) -> DisplayMode {
        if self.display_mode == DisplayMode::XY && self.snapshot.channels >= 2 {
            DisplayMode::XY
        } else {
            DisplayMode::LR
        }
    }

    fn effective_persistence_for(&self, channels: usize) -> f32 {
        let mut value = self.persistence.clamp(0.0, 0.98);
        if self.display_mode == DisplayMode::XY && channels >= 2 {
            value = value.min(0.9);
        }
        value
    }

    fn recompute_fade_alpha(&mut self) {
        if self.snapshot.samples.is_empty() {
            self.fade_alpha = 1.0_f32;
            return;
        }

        let persistence = self.effective_persistence_for(self.snapshot.channels);
        let fade = (1.0_f32 - persistence).max(0.0_f32);
        self.fade_alpha = fade.sqrt().clamp(0.05_f32, 1.0_f32);
    }

    fn visual_params(&self, bounds: Rectangle) -> OscilloscopeParams {
        let mode = self.effective_display_mode();
        let channels = match mode {
            DisplayMode::XY => 2,
            DisplayMode::LR => self.snapshot.channels.max(1),
        };
        let colors = self
            .style
            .channel_colors
            .iter()
            .cycle()
            .take(channels)
            .map(|c| theme::color_to_rgba(*c))
            .collect();

        let samples = self.build_render_samples(mode);

        OscilloscopeParams {
            instance_id: self.instance_id,
            bounds,
            channels,
            samples_per_channel: self.snapshot.samples_per_channel,
            samples,
            colors,
            line_alpha: self.style.line_alpha,
            fade_alpha: self.fade_alpha,
            vertical_padding: self.style.vertical_padding,
            channel_gap: self.style.channel_gap,
            amplitude_scale: self.style.amplitude_scale,
            stroke_width: self.style.stroke_width,
            display_mode: mode,
        }
    }

    fn build_render_samples(&self, mode: DisplayMode) -> Vec<f32> {
        match mode {
            DisplayMode::LR => self.snapshot.samples.clone(),
            DisplayMode::XY => {
                let per_channel = self.snapshot.samples_per_channel;
                if per_channel == 0 || self.snapshot.channels < 2 {
                    return Vec::new();
                }

                let required = per_channel.saturating_mul(2);
                if self.snapshot.samples.len() < required {
                    return Vec::new();
                }

                let left = &self.snapshot.samples[..per_channel];
                let right = &self.snapshot.samples[per_channel..per_channel * 2];
                let len = left.len().min(right.len());
                let mut output = Vec::with_capacity(len * 2);
                for index in 0..len {
                    output.push(left[index]);
                    output.push(right[index]);
                }
                output
            }
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
