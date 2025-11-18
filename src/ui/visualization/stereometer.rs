//! UI wrapper around the stereometer DSP processor and renderer.

use crate::audio::meter_tap::MeterFormat;
use crate::dsp::stereometer::{
    StereometerConfig, StereometerProcessor as CoreStereometerProcessor, StereometerSnapshot,
};
use crate::dsp::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::ui::render::stereometer::{StereometerParams, StereometerPrimitive};
use crate::ui::settings::StereometerSettings;
use crate::ui::theme;
use iced::advanced::Renderer as _;
use iced::advanced::renderer::{self, Quad};
use iced::advanced::widget::{Tree, tree};
use iced::advanced::{Layout, Widget, layout, mouse};
use iced::{Background, Color, Element, Length, Rectangle, Size};
use iced_wgpu::primitive::Renderer as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

static NEXT_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

fn next_instance_id() -> u64 {
    NEXT_INSTANCE_ID.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug, Clone)]
pub struct StereometerProcessor {
    inner: CoreStereometerProcessor,
}

impl StereometerProcessor {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            inner: CoreStereometerProcessor::new(StereometerConfig {
                sample_rate,
                ..Default::default()
            }),
        }
    }

    pub fn ingest(&mut self, samples: &[f32], format: MeterFormat) -> StereometerSnapshot {
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

    pub fn update_config(&mut self, config: StereometerConfig) {
        self.inner.update_config(config);
    }

    pub fn config(&self) -> StereometerConfig {
        self.inner.config()
    }
}

#[derive(Debug, Clone)]
pub struct StereometerState {
    snapshot: StereometerSnapshot,
    style: StereometerStyle,
    display_points: Vec<(f32, f32)>,
    fade_alpha: f32,
    persistence: f32,
    instance_id: u64,
}

impl StereometerState {
    pub fn new() -> Self {
        Self {
            snapshot: StereometerSnapshot::default(),
            style: StereometerStyle::default(),
            display_points: Vec::new(),
            fade_alpha: 1.0,
            persistence: 0.85,
            instance_id: next_instance_id(),
        }
    }

    pub fn update_view_settings(&mut self, settings: &StereometerSettings) {
        self.persistence = settings.persistence.clamp(0.0, 0.9);
        self.recompute_fade_alpha();
    }

    pub fn set_palette(&mut self, palette: &[Color]) {
        self.style.trace_color = palette.first().copied().unwrap_or(theme::accent_primary());
    }

    pub fn palette(&self) -> &[Color] {
        std::slice::from_ref(&self.style.trace_color)
    }

    pub fn apply_snapshot(&mut self, snapshot: &StereometerSnapshot) {
        if snapshot.xy_points.is_empty() {
            self.snapshot = snapshot.clone();
            self.display_points.clear();
            self.fade_alpha = 1.0_f32;
            return;
        }

        let total_points = snapshot.xy_points.len();
        if self.display_points.len() != total_points {
            self.display_points.resize(total_points, (0.0, 0.0));
        }

        let persistence = self.persistence.clamp(0.0, 0.9);
        if persistence <= f32::EPSILON {
            self.display_points.copy_from_slice(&snapshot.xy_points);
        } else {
            let fresh = 1.0_f32 - persistence;
            for (current, incoming) in self
                .display_points
                .iter_mut()
                .zip(snapshot.xy_points.iter())
            {
                current.0 = (current.0 * persistence) + (incoming.0 * fresh);
                current.1 = (current.1 * persistence) + (incoming.1 * fresh);
            }
        }

        self.snapshot = snapshot.clone();
        self.snapshot.xy_points.clear();
        self.snapshot
            .xy_points
            .extend_from_slice(&self.display_points);
        self.recompute_fade_alpha();
    }

    pub fn view_settings(&self) -> f32 {
        self.persistence
    }

    fn recompute_fade_alpha(&mut self) {
        if self.snapshot.xy_points.is_empty() {
            self.fade_alpha = 1.0_f32;
            return;
        }

        let persistence = self.persistence.clamp(0.0, 0.9);
        let fade = (1.0_f32 - persistence).max(0.0_f32);
        self.fade_alpha = fade.sqrt().clamp(0.05_f32, 1.0_f32);
    }

    fn visual_params(&self, bounds: Rectangle) -> StereometerParams {
        let color = theme::color_to_rgba(self.style.trace_color);

        StereometerParams {
            instance_id: self.instance_id,
            bounds,
            xy_points: self.snapshot.xy_points.clone(),
            color,
            line_alpha: self.style.line_alpha,
            fade_alpha: self.fade_alpha,
            amplitude_scale: self.style.amplitude_scale,
            stroke_width: self.style.stroke_width,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StereometerStyle {
    pub background: Color,
    pub trace_color: Color,
    pub line_alpha: f32,
    pub amplitude_scale: f32,
    pub stroke_width: f32,
}

impl Default for StereometerStyle {
    fn default() -> Self {
        Self {
            background: theme::base_color(),
            trace_color: theme::DEFAULT_STEREOMETER_PALETTE[0],
            line_alpha: 0.92,
            amplitude_scale: 0.9,
            stroke_width: 1.0,
        }
    }
}

#[derive(Debug)]
pub struct Stereometer<'a> {
    state: &'a StereometerState,
}

impl<'a> Stereometer<'a> {
    pub fn new(state: &'a StereometerState) -> Self {
        Self { state }
    }
}

impl<'a, Message> Widget<Message, iced::Theme, iced::Renderer> for Stereometer<'a> {
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

        renderer.draw_primitive(bounds, StereometerPrimitive::new(params));
    }

    fn children(&self) -> Vec<Tree> {
        Vec::new()
    }

    fn diff(&self, _tree: &mut Tree) {}
}

pub fn widget<'a, Message>(state: &'a StereometerState) -> Element<'a, Message>
where
    Message: 'a,
{
    Element::new(Stereometer::new(state))
}
