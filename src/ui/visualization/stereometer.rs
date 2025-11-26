//! UI wrapper around the stereometer DSP processor and renderer.

use crate::audio::meter_tap::MeterFormat;
use crate::dsp::stereometer::{
    StereometerConfig, StereometerProcessor as CoreStereometerProcessor, StereometerSnapshot,
};
use crate::dsp::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::ui::render::stereometer::{StereometerParams, StereometerPrimitive};
use crate::ui::settings::{StereometerMode, StereometerSettings};
use crate::ui::theme;
use iced::advanced::Renderer as _;
use iced::advanced::renderer::{self, Quad};
use iced::advanced::widget::{Tree, tree};
use iced::advanced::{Layout, Widget, layout, mouse};
use iced::{Background, Color, Element, Length, Rectangle, Size};
use iced_wgpu::primitive::Renderer as _;
use std::cell::RefCell;
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
    display_points: Vec<(f32, f32)>,
    trace_color: Color,
    grid_color: Color,
    persistence: f32,
    mode: StereometerMode,
    instance_id: u64,
}

impl StereometerState {
    pub fn new() -> Self {
        Self {
            snapshot: StereometerSnapshot::default(),
            display_points: Vec::new(),
            trace_color: theme::DEFAULT_STEREOMETER_PALETTE[0],
            grid_color: theme::DEFAULT_STEREOMETER_PALETTE[1],
            persistence: 0.0,
            mode: StereometerMode::default(),
            instance_id: next_instance_id(),
        }
    }

    pub fn update_view_settings(&mut self, settings: &StereometerSettings) {
        self.persistence = settings.persistence.clamp(0.0, 0.9);
        self.mode = settings.mode;
    }

    pub fn set_palette(&mut self, palette: &[Color]) {
        self.trace_color = palette
            .first()
            .copied()
            .unwrap_or(theme::DEFAULT_STEREOMETER_PALETTE[0]);
        self.grid_color = palette
            .get(1)
            .copied()
            .unwrap_or(theme::DEFAULT_STEREOMETER_PALETTE[1]);
    }

    pub fn palette(&self) -> [Color; 2] {
        [self.trace_color, self.grid_color]
    }

    pub fn apply_snapshot(&mut self, snapshot: &StereometerSnapshot) {
        if snapshot.xy_points.is_empty() {
            self.display_points.clear();
            self.snapshot = snapshot.clone();
            return;
        }

        if self.display_points.len() != snapshot.xy_points.len() {
            self.display_points
                .resize(snapshot.xy_points.len(), (0.0, 0.0));
        }

        if self.persistence <= f32::EPSILON {
            self.display_points.copy_from_slice(&snapshot.xy_points);
        } else {
            let fresh = 1.0 - self.persistence;
            for (current, incoming) in self.display_points.iter_mut().zip(&snapshot.xy_points) {
                current.0 = current.0 * self.persistence + incoming.0 * fresh;
                current.1 = current.1 * self.persistence + incoming.1 * fresh;
            }
        }

        self.snapshot.xy_points.clone_from(&self.display_points);
        self.snapshot.correlation = snapshot.correlation;
    }

    pub fn view_settings(&self) -> (f32, StereometerMode) {
        (self.persistence, self.mode)
    }

    fn visual_params(&self, bounds: Rectangle) -> Option<StereometerParams> {
        if self.snapshot.xy_points.len() < 2 {
            return None;
        }

        Some(StereometerParams {
            instance_id: self.instance_id,
            bounds,
            points: self.snapshot.xy_points.clone(),
            trace_color: theme::color_to_rgba(self.trace_color),
            grid_color: theme::color_to_rgba(self.grid_color),
            mode: self.mode,
        })
    }
}

#[derive(Debug)]
pub struct Stereometer<'a> {
    state: &'a RefCell<StereometerState>,
}

impl<'a> Stereometer<'a> {
    pub fn new(state: &'a RefCell<StereometerState>) -> Self {
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
        theme: &iced::Theme,
        _style: &renderer::Style,
        layout: Layout<'_>,
        _cursor: mouse::Cursor,
        _viewport: &Rectangle,
    ) {
        let bounds = layout.bounds();
        let Some(params) = self.state.borrow().visual_params(bounds) else {
            renderer.fill_quad(
                Quad {
                    bounds,
                    border: Default::default(),
                    shadow: Default::default(),
                },
                Background::Color(theme.extended_palette().background.base.color),
            );
            return;
        };

        renderer.draw_primitive(bounds, StereometerPrimitive::new(params));
    }

    fn children(&self) -> Vec<Tree> {
        Vec::new()
    }

    fn diff(&self, _tree: &mut Tree) {}
}

pub fn widget<'a, Message>(state: &'a RefCell<StereometerState>) -> Element<'a, Message>
where
    Message: 'a,
{
    Element::new(Stereometer::new(state))
}
