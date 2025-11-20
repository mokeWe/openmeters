//! UI wrapper around the oscilloscope DSP processor and renderer.

use crate::audio::meter_tap::MeterFormat;
use crate::dsp::oscilloscope::{
    OscilloscopeConfig, OscilloscopeProcessor as CoreOscilloscopeProcessor, OscilloscopeSnapshot,
};
use crate::dsp::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::ui::render::oscilloscope::{OscilloscopeParams, OscilloscopePrimitive};
use crate::ui::settings::{OscilloscopeChannelMode, OscilloscopeSettings};
use crate::ui::theme;
use iced::advanced::renderer;
use iced::advanced::widget::{Tree, tree};
use iced::advanced::{Layout, Widget, layout, mouse};
use iced::{Color, Element, Length, Rectangle, Size};
use iced_wgpu::primitive::Renderer as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

static NEXT_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

fn next_instance_id() -> u64 {
    NEXT_INSTANCE_ID.fetch_add(1, Ordering::Relaxed)
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
    latest_snapshot: OscilloscopeSnapshot,
    style: OscilloscopeStyle,
    display_samples: Vec<f32>,
    persistence: f32,
    channel_mode: OscilloscopeChannelMode,
    instance_id: u64,
}

impl OscilloscopeState {
    pub fn new() -> Self {
        Self {
            snapshot: OscilloscopeSnapshot::default(),
            latest_snapshot: OscilloscopeSnapshot::default(),
            style: OscilloscopeStyle::default(),
            display_samples: Vec::new(),
            persistence: 0.1,
            channel_mode: OscilloscopeChannelMode::Both,
            instance_id: next_instance_id(),
        }
    }

    pub fn update_view_settings(&mut self, settings: &OscilloscopeSettings) {
        self.persistence = settings.persistence.clamp(0.0, 1.0);
        if self.channel_mode != settings.channel_mode {
            self.channel_mode = settings.channel_mode;
            self.reproject_latest(false);
        }
    }

    pub fn set_palette(&mut self, palette: &[Color]) {
        self.style.channel_colors.clear();
        self.style.channel_colors.extend_from_slice(palette);
    }

    pub fn palette(&self) -> &[Color] {
        &self.style.channel_colors
    }

    pub fn apply_snapshot(&mut self, snapshot: &OscilloscopeSnapshot) {
        if snapshot.samples.is_empty() {
            self.snapshot = snapshot.clone();
            self.display_samples.clear();
            return;
        }

        self.latest_snapshot = snapshot.clone();
        self.project_snapshot(self.latest_snapshot.clone(), true);
    }

    pub fn view_settings(&self) -> f32 {
        self.persistence
    }

    pub fn channel_mode(&self) -> OscilloscopeChannelMode {
        self.channel_mode
    }

    fn effective_persistence(&self) -> f32 {
        self.persistence.clamp(0.0, 0.98)
    }

    fn reproject_latest(&mut self, blend: bool) {
        let latest = self.latest_snapshot.clone();
        if latest.samples.is_empty() {
            self.snapshot = latest;
            self.display_samples.clear();
            return;
        }
        self.project_snapshot(latest, blend);
    }

    fn project_snapshot(&mut self, source: OscilloscopeSnapshot, blend: bool) {
        let Some(projection) = self.project_samples(&source) else {
            self.snapshot = OscilloscopeSnapshot::default();
            self.display_samples.clear();
            return;
        };

        if self.display_samples.len() != projection.samples.len() {
            self.display_samples
                .resize(projection.samples.len(), 0.0_f32);
        }

        if blend {
            let persistence = self.effective_persistence();
            if persistence <= f32::EPSILON {
                self.display_samples.copy_from_slice(&projection.samples);
            } else {
                let fresh = 1.0_f32 - persistence;
                for (current, incoming) in self
                    .display_samples
                    .iter_mut()
                    .zip(projection.samples.iter())
                {
                    *current = (*current * persistence) + (*incoming * fresh);
                }
            }
        } else {
            self.display_samples.copy_from_slice(&projection.samples);
        }

        self.snapshot.channels = projection.channels;
        self.snapshot.samples_per_channel = projection.samples_per_channel;
        self.snapshot.samples.clear();
        self.snapshot
            .samples
            .extend_from_slice(&self.display_samples);
    }

    fn project_samples(&self, source: &OscilloscopeSnapshot) -> Option<Projection> {
        let channels = source.channels.max(1);
        let samples_per_channel = source.samples_per_channel;
        if samples_per_channel == 0 || source.samples.len() < samples_per_channel {
            return None;
        }

        let mut projection = Projection {
            channels,
            samples_per_channel,
            samples: Vec::new(),
        };

        match self.channel_mode {
            OscilloscopeChannelMode::Both => {
                projection.samples = source.samples.clone();
                projection.channels = channels;
            }
            OscilloscopeChannelMode::Left => {
                projection.channels = 1;
                let slice = source
                    .samples
                    .chunks(samples_per_channel)
                    .next()
                    .unwrap_or(&[]);
                projection.samples.extend_from_slice(slice);
            }
            OscilloscopeChannelMode::Right => {
                projection.channels = 1;
                let slice = source
                    .samples
                    .chunks(samples_per_channel)
                    .nth(1)
                    .or_else(|| source.samples.chunks(samples_per_channel).last())
                    .unwrap_or(&[]);
                projection.samples.extend_from_slice(slice);
            }
            OscilloscopeChannelMode::Mono => {
                projection.channels = 1;
                projection.samples.resize(samples_per_channel, 0.0);
                for channel_samples in source.samples.chunks(samples_per_channel).take(channels) {
                    for (idx, sample) in channel_samples.iter().enumerate() {
                        projection.samples[idx] += *sample;
                    }
                }
                let denom = channels as f32;
                if denom > 0.0 {
                    for sample in &mut projection.samples {
                        *sample /= denom;
                    }
                }
            }
        }

        if projection.samples.is_empty() {
            None
        } else {
            Some(projection)
        }
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
            instance_id: self.instance_id,
            bounds,
            channels,
            samples_per_channel: self.snapshot.samples_per_channel,
            samples: self.snapshot.samples.clone(),
            colors,
            line_alpha: self.style.line_alpha,
            vertical_padding: self.style.vertical_padding,
            channel_gap: self.style.channel_gap,
            amplitude_scale: self.style.amplitude_scale,
            stroke_width: self.style.stroke_width,
        }
    }
}

#[derive(Debug, Clone)]
struct Projection {
    channels: usize,
    samples_per_channel: usize,
    samples: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct OscilloscopeStyle {
    pub channel_colors: Vec<Color>,
    pub line_alpha: f32,
    pub vertical_padding: f32,
    pub channel_gap: f32,
    pub amplitude_scale: f32,
    pub stroke_width: f32,
}

impl Default for OscilloscopeStyle {
    fn default() -> Self {
        Self {
            channel_colors: theme::DEFAULT_OSCILLOSCOPE_PALETTE.to_vec(),
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
