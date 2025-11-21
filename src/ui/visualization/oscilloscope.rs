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
use std::cell::RefCell;
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
            self.inner.update_config(config);
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
        self.style.colors.clear();
        self.style.colors.extend_from_slice(palette);
    }

    pub fn palette(&self) -> &[Color] {
        &self.style.colors
    }

    pub fn apply_snapshot(&mut self, snapshot: &OscilloscopeSnapshot) {
        if snapshot.samples.is_empty() {
            self.snapshot = snapshot.clone();
            self.display_samples.clear();
            return;
        }
        self.latest_snapshot = snapshot.clone();
        self.project_snapshot(true);
    }

    pub fn channel_mode(&self) -> OscilloscopeChannelMode {
        self.channel_mode
    }

    pub fn persistence(&self) -> f32 {
        self.persistence
    }

    fn reproject_latest(&mut self, blend: bool) {
        if self.latest_snapshot.samples.is_empty() {
            self.snapshot = self.latest_snapshot.clone();
            self.display_samples.clear();
            return;
        }
        self.project_snapshot(blend);
    }

    fn project_snapshot(&mut self, blend: bool) {
        let Some((channels, samples_per_channel, samples)) =
            self.project_samples(&self.latest_snapshot)
        else {
            self.snapshot = OscilloscopeSnapshot::default();
            self.display_samples.clear();
            return;
        };

        self.display_samples.resize(samples.len(), 0.0);

        if blend {
            let persistence = self.persistence.clamp(0.0, 0.98);
            if persistence <= f32::EPSILON {
                self.display_samples.copy_from_slice(&samples);
            } else {
                let fresh = 1.0 - persistence;
                for (current, incoming) in self.display_samples.iter_mut().zip(&samples) {
                    *current = *current * persistence + incoming * fresh;
                }
            }
        } else {
            self.display_samples.copy_from_slice(&samples);
        }

        self.snapshot.channels = channels;
        self.snapshot.samples_per_channel = samples_per_channel;
        self.snapshot.samples.clone_from(&self.display_samples);
    }

    fn project_samples(&self, source: &OscilloscopeSnapshot) -> Option<(usize, usize, Vec<f32>)> {
        let channels = source.channels.max(1);
        let samples_per_channel = source.samples_per_channel;
        if samples_per_channel == 0 || source.samples.len() < samples_per_channel {
            return None;
        }

        match self.channel_mode {
            OscilloscopeChannelMode::Both => {
                Some((channels, samples_per_channel, source.samples.clone()))
            }
            OscilloscopeChannelMode::Left => {
                let samples = source.samples.chunks(samples_per_channel).next()?.to_vec();
                Some((1, samples_per_channel, samples))
            }
            OscilloscopeChannelMode::Right => {
                let samples = source
                    .samples
                    .chunks(samples_per_channel)
                    .nth(1)
                    .or_else(|| source.samples.chunks(samples_per_channel).last())?
                    .to_vec();
                Some((1, samples_per_channel, samples))
            }
            OscilloscopeChannelMode::Mono => {
                let mut samples = vec![0.0; samples_per_channel];
                for channel_samples in source.samples.chunks(samples_per_channel).take(channels) {
                    for (dest, src) in samples.iter_mut().zip(channel_samples) {
                        *dest += *src;
                    }
                }
                let scale = 1.0 / channels as f32;
                for sample in &mut samples {
                    *sample *= scale;
                }
                Some((1, samples_per_channel, samples))
            }
        }
    }

    fn visual_params(&self, bounds: Rectangle) -> OscilloscopeParams {
        let channels = self.snapshot.channels.max(1);
        let colors = self
            .style
            .colors
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
            fill_alpha: 0.15,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OscilloscopeStyle {
    pub colors: Vec<Color>,
}

impl Default for OscilloscopeStyle {
    fn default() -> Self {
        Self {
            colors: theme::DEFAULT_OSCILLOSCOPE_PALETTE.to_vec(),
        }
    }
}

#[derive(Debug)]
pub struct Oscilloscope<'a> {
    state: &'a RefCell<OscilloscopeState>,
}

impl<'a> Oscilloscope<'a> {
    pub fn new(state: &'a RefCell<OscilloscopeState>) -> Self {
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
        let params = self.state.borrow().visual_params(bounds);

        renderer.draw_primitive(bounds, OscilloscopePrimitive::new(params));
    }

    fn children(&self) -> Vec<Tree> {
        Vec::new()
    }

    fn diff(&self, _tree: &mut Tree) {}
}

pub fn widget<'a, Message>(state: &'a RefCell<OscilloscopeState>) -> Element<'a, Message>
where
    Message: 'a,
{
    Element::new(Oscilloscope::new(state))
}
