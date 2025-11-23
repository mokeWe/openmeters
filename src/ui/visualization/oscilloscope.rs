//! UI wrapper around the oscilloscope DSP processor and renderer.

use crate::audio::meter_tap::MeterFormat;
use crate::dsp::oscilloscope::{
    OscilloscopeConfig, OscilloscopeProcessor as CoreOscilloscopeProcessor, OscilloscopeSnapshot,
};
use crate::dsp::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::ui::render::oscilloscope::{OscilloscopeParams, OscilloscopePrimitive};
use crate::ui::settings::OscilloscopeChannelMode;
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
    pub fn new(config: OscilloscopeConfig) -> Self {
        Self {
            inner: CoreOscilloscopeProcessor::new(config),
        }
    }

    pub fn ingest(&mut self, samples: &[f32], format: MeterFormat) -> Option<OscilloscopeSnapshot> {
        if samples.is_empty() {
            return None;
        }

        let sample_rate = format.sample_rate.max(1.0);
        let mut config = self.inner.config();
        if (config.sample_rate - sample_rate).abs() > f32::EPSILON {
            config.sample_rate = sample_rate;
            self.inner.update_config(config);
        }

        let block = AudioBlock::new(samples, format.channels.max(1), sample_rate, Instant::now());

        match self.inner.process_block(&block) {
            ProcessorUpdate::Snapshot(snapshot) => Some(snapshot),
            ProcessorUpdate::None => None,
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
    persistence: f32,
    channel_mode: OscilloscopeChannelMode,
}

impl OscilloscopeState {
    pub fn new() -> Self {
        Self {
            snapshot: OscilloscopeSnapshot::default(),
            style: OscilloscopeStyle::default(),
            persistence: 0.0,
            channel_mode: OscilloscopeChannelMode::default(),
        }
    }

    pub fn update_view_settings(
        &mut self,
        persistence: f32,
        channel_mode: OscilloscopeChannelMode,
    ) {
        self.persistence = persistence.clamp(0.0, 1.0);
        let mode_changed = self.channel_mode != channel_mode;
        self.channel_mode = channel_mode;
        if mode_changed {
            self.snapshot = Self::project_channels(&self.snapshot, channel_mode);
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
        let projected = Self::project_channels(snapshot, self.channel_mode);

        if !projected.samples.is_empty()
            && !self.snapshot.samples.is_empty()
            && projected.samples.len() == self.snapshot.samples.len()
        {
            let persistence = self.persistence.clamp(0.0, 0.98);
            if persistence > f32::EPSILON {
                let fresh = 1.0 - persistence;
                for (current, incoming) in self.snapshot.samples.iter_mut().zip(&projected.samples)
                {
                    *current = *current * persistence + incoming * fresh;
                }
                return;
            }
        }

        self.snapshot = projected;
    }

    pub fn channel_mode(&self) -> OscilloscopeChannelMode {
        self.channel_mode
    }

    pub fn persistence(&self) -> f32 {
        self.persistence
    }

    fn project_channels(
        source: &OscilloscopeSnapshot,
        mode: OscilloscopeChannelMode,
    ) -> OscilloscopeSnapshot {
        let channels = source.channels.max(1);
        let spc = source.samples_per_channel;

        if spc == 0 || source.samples.len() < spc {
            return OscilloscopeSnapshot::default();
        }

        let (out_channels, samples) = match mode {
            OscilloscopeChannelMode::Both => (channels, source.samples.clone()),
            OscilloscopeChannelMode::Left => {
                let samples = source
                    .samples
                    .chunks(spc)
                    .next()
                    .map(|s| s.to_vec())
                    .unwrap_or_default();
                (1, samples)
            }
            OscilloscopeChannelMode::Right => {
                let samples = source
                    .samples
                    .chunks(spc)
                    .nth(1)
                    .or_else(|| source.samples.chunks(spc).last())
                    .map(|s| s.to_vec())
                    .unwrap_or_default();
                (1, samples)
            }
            OscilloscopeChannelMode::Mono => {
                let mut samples = vec![0.0; spc];
                let scale = 1.0 / channels as f32;
                for channel_samples in source.samples.chunks(spc).take(channels) {
                    for (dest, src) in samples.iter_mut().zip(channel_samples) {
                        *dest += *src * scale;
                    }
                }
                (1, samples)
            }
        };

        OscilloscopeSnapshot {
            channels: out_channels,
            samples_per_channel: spc,
            samples,
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
