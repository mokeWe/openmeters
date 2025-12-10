//! UI wrapper around the loudness DSP processor and renderer.

use crate::audio::meter_tap::MeterFormat;
use crate::dsp::loudness::{
    LoudnessConfig, LoudnessProcessor as CoreLoudnessProcessor, LoudnessSnapshot,
};
use crate::dsp::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::ui::render::loudness::{LoudnessMeterPrimitive, MeterBar, RenderParams};
use crate::ui::theme;
use iced::advanced::Renderer as _;
use iced::advanced::renderer::{self, Quad};
use iced::advanced::widget::{Tree, tree};
use iced::advanced::{Layout, Widget, layout, mouse, text};
use iced::alignment::{Horizontal, Vertical};
use iced::{Background, Border, Color, Element, Length, Point, Rectangle, Size, Theme};
use iced_wgpu::primitive::Renderer as _;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::Instant;

const CHANNELS: usize = 2;
const DEFAULT_RANGE: (f32, f32) = (-60.0, 4.0);
const DEFAULT_WIDTH: f32 = 100.0;
const GUIDE_LEVELS: [f32; 6] = [0.0, -6.0, -12.0, -18.0, -24.0, -36.0];
const LEFT_PADDING: f32 = 28.0;
const RIGHT_PADDING: f32 = 64.0;
const LABEL_FONT_SIZE: f32 = 10.0;
const VALUE_FONT_SIZE: f32 = 12.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum MeterMode {
    #[default]
    LufsShortTerm,
    LufsMomentary,
    RmsFast,
    RmsSlow,
    TruePeak,
}

impl MeterMode {
    pub const ALL: &'static [MeterMode] = &[
        MeterMode::LufsShortTerm,
        MeterMode::LufsMomentary,
        MeterMode::RmsFast,
        MeterMode::RmsSlow,
        MeterMode::TruePeak,
    ];

    pub fn unit_label(self) -> &'static str {
        match self {
            MeterMode::LufsShortTerm | MeterMode::LufsMomentary => "LUFS",
            MeterMode::RmsFast | MeterMode::RmsSlow | MeterMode::TruePeak => "dB",
        }
    }
}

impl fmt::Display for MeterMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MeterMode::LufsShortTerm => f.write_str("LUFS Short-term"),
            MeterMode::LufsMomentary => f.write_str("LUFS Momentary"),
            MeterMode::RmsFast => f.write_str("RMS Fast"),
            MeterMode::RmsSlow => f.write_str("RMS Slow"),
            MeterMode::TruePeak => f.write_str("True Peak"),
        }
    }
}

/// UI wrapper around the shared loudness processor.
#[derive(Debug, Clone)]
pub struct LoudnessMeterProcessor {
    inner: CoreLoudnessProcessor,
    channels: usize,
}

impl LoudnessMeterProcessor {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            inner: CoreLoudnessProcessor::new(LoudnessConfig {
                sample_rate,
                ..Default::default()
            }),
            channels: CHANNELS,
        }
    }

    pub fn ingest(&mut self, samples: &[f32], format: MeterFormat) -> LoudnessSnapshot {
        if samples.is_empty() {
            return *self.inner.snapshot();
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
            ProcessorUpdate::Snapshot(s) => s,
            ProcessorUpdate::None => *self.inner.snapshot(),
        }
    }
}

pub const LOUDNESS_PALETTE_SIZE: usize = 5;

/// View-model state consumed by the loudness widget.
#[derive(Debug, Clone)]
pub struct LoudnessMeterState {
    short_term_loudness: [f32; CHANNELS],
    momentary_loudness: [f32; CHANNELS],
    rms_fast_db: [f32; CHANNELS],
    rms_slow_db: [f32; CHANNELS],
    true_peak_db: [f32; CHANNELS],
    range: (f32, f32),
    left_mode: MeterMode,
    right_mode: MeterMode,
    palette: [Color; LOUDNESS_PALETTE_SIZE],
}

impl LoudnessMeterState {
    pub fn new() -> Self {
        Self {
            short_term_loudness: [DEFAULT_RANGE.0; CHANNELS],
            momentary_loudness: [DEFAULT_RANGE.0; CHANNELS],
            rms_fast_db: [DEFAULT_RANGE.0; CHANNELS],
            rms_slow_db: [DEFAULT_RANGE.0; CHANNELS],
            true_peak_db: [DEFAULT_RANGE.0; CHANNELS],
            range: DEFAULT_RANGE,
            left_mode: MeterMode::TruePeak,
            right_mode: MeterMode::LufsShortTerm,
            palette: theme::DEFAULT_LOUDNESS_PALETTE,
        }
    }

    pub fn apply_snapshot(&mut self, snapshot: &LoudnessSnapshot) {
        for (i, &v) in snapshot
            .short_term_loudness
            .iter()
            .take(CHANNELS)
            .enumerate()
        {
            self.short_term_loudness[i] = v;
        }
        for (i, &v) in snapshot
            .momentary_loudness
            .iter()
            .take(CHANNELS)
            .enumerate()
        {
            self.momentary_loudness[i] = v;
        }
        for (i, &v) in snapshot.rms_fast_db.iter().take(CHANNELS).enumerate() {
            self.rms_fast_db[i] = v;
        }
        for (i, &v) in snapshot.rms_slow_db.iter().take(CHANNELS).enumerate() {
            self.rms_slow_db[i] = v;
        }
        for (i, &v) in snapshot.true_peak_db.iter().take(CHANNELS).enumerate() {
            self.true_peak_db[i] = v;
        }
    }

    pub fn set_modes(&mut self, left: MeterMode, right: MeterMode) {
        self.left_mode = left;
        self.right_mode = right;
    }

    pub fn left_mode(&self) -> MeterMode {
        self.left_mode
    }

    pub fn right_mode(&self) -> MeterMode {
        self.right_mode
    }

    pub fn set_palette(&mut self, palette: &[Color; LOUDNESS_PALETTE_SIZE]) {
        self.palette = *palette;
    }

    pub fn palette(&self) -> &[Color; LOUDNESS_PALETTE_SIZE] {
        &self.palette
    }

    #[cfg(test)]
    pub fn short_term_average(&self) -> f32 {
        self.short_term_loudness.iter().sum::<f32>() / CHANNELS as f32
    }

    fn get_value(&self, mode: MeterMode, channel: usize) -> f32 {
        match mode {
            MeterMode::LufsShortTerm => {
                self.short_term_loudness.iter().sum::<f32>() / CHANNELS as f32
            }
            MeterMode::LufsMomentary => {
                self.momentary_loudness.iter().sum::<f32>() / CHANNELS as f32
            }
            MeterMode::RmsFast => self
                .rms_fast_db
                .get(channel)
                .copied()
                .unwrap_or(self.range.0),
            MeterMode::RmsSlow => self
                .rms_slow_db
                .get(channel)
                .copied()
                .unwrap_or(self.range.0),
            MeterMode::TruePeak => self
                .true_peak_db
                .get(channel)
                .copied()
                .unwrap_or(self.range.0),
        }
    }

    fn render_params(&self) -> RenderParams {
        let (min, max) = self.range;
        let guide_color = theme::color_to_rgba(self.palette[4]);
        let mut bg = self.palette[0];
        bg.a = 1.0;
        let bg_color = theme::color_to_rgba(bg);

        RenderParams {
            min_db: min,
            max_db: max,
            bars: vec![
                MeterBar {
                    bg_color,
                    fills: vec![
                        (
                            self.get_value(self.left_mode, 0),
                            theme::color_to_rgba(self.palette[1]),
                        ),
                        (
                            self.get_value(self.left_mode, 1),
                            theme::color_to_rgba(self.palette[2]),
                        ),
                    ],
                },
                MeterBar {
                    bg_color,
                    fills: vec![(
                        self.get_value(self.right_mode, 0),
                        theme::color_to_rgba(self.palette[3]),
                    )],
                },
            ],
            guides: GUIDE_LEVELS
                .iter()
                .filter(|&&l| l >= min && l <= max)
                .copied()
                .collect(),
            guide_color,
            threshold_db: Some(0.0),
            left_padding: LEFT_PADDING,
            right_padding: RIGHT_PADDING,
        }
    }
}

/// The loudness meter widget.
#[derive(Debug)]
pub struct LoudnessMeter<'a> {
    state: &'a LoudnessMeterState,
    width: f32,
}

impl<'a> LoudnessMeter<'a> {
    pub fn new(state: &'a LoudnessMeterState) -> Self {
        Self {
            state,
            width: DEFAULT_WIDTH,
        }
    }

    pub fn with_width(mut self, width: f32) -> Self {
        self.width = width.max(0.0);
        self
    }
}

impl<'a, Message> Widget<Message, Theme, iced::Renderer> for LoudnessMeter<'a> {
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<()>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(())
    }

    fn size(&self) -> Size<Length> {
        Size::new(Length::Fixed(self.width), Length::Fill)
    }

    fn layout(
        &mut self,
        _tree: &mut Tree,
        _renderer: &iced::Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        layout::Node::new(limits.resolve(Length::Fixed(self.width), Length::Fill, Size::ZERO))
    }

    fn draw(
        &self,
        _tree: &Tree,
        renderer: &mut iced::Renderer,
        theme: &Theme,
        _style: &renderer::Style,
        layout: Layout<'_>,
        _cursor: mouse::Cursor,
        _viewport: &Rectangle,
    ) {
        let bounds = layout.bounds();
        let params = self.state.render_params();

        renderer.draw_primitive(bounds, LoudnessMeterPrimitive::new(params.clone()));

        let palette = theme.extended_palette();
        let label_color = self.state.palette[4];

        if params.meter_bounds(&bounds).is_some() {
            let height = bounds.height;

            for &db in &params.guides {
                let ratio = params.db_to_ratio(db);
                let y = bounds.y + height * (1.0 - ratio);
                let label = format!("{:.0}", db.abs());

                text::Renderer::fill_text(
                    renderer,
                    iced::advanced::text::Text {
                        content: label,
                        bounds: Size::new(LEFT_PADDING, 20.0),
                        size: iced::Pixels(LABEL_FONT_SIZE),
                        line_height: iced::advanced::text::LineHeight::default(),
                        font: iced::Font::default(),
                        align_x: Horizontal::Right.into(),
                        align_y: Vertical::Center,
                        shaping: iced::advanced::text::Shaping::Basic,
                        wrapping: iced::advanced::text::Wrapping::None,
                    },
                    Point::new(bounds.x + LEFT_PADDING - 4.0, y),
                    label_color,
                    bounds,
                );
            }

            let value = self.state.get_value(self.state.right_mode, 0);
            let unit = self.state.right_mode.unit_label();
            let ratio = params.db_to_ratio(value);
            let y = bounds.y + height * (1.0 - ratio);
            let label = format!("{:.1} {}", value, unit);

            let (meter_x, bar_width, stride) = params.meter_bounds(&bounds).unwrap();
            let right_bar_end = meter_x + stride + bar_width;
            let label_x = right_bar_end + 4.0;
            let label_width = 68.0;
            let clamp_max = (bounds.y + bounds.height - 20.0).max(bounds.y);
            let label_rect = Rectangle {
                x: label_x,
                y: (y - 10.0).clamp(bounds.y, clamp_max),
                width: label_width,
                height: 20.0,
            };

            renderer.fill_quad(
                Quad {
                    bounds: label_rect,
                    border: Border::default(),
                    shadow: Default::default(),
                    snap: true,
                },
                Background::Color(Color {
                    a: 1.0,
                    ..self.state.palette[0]
                }),
            );

            text::Renderer::fill_text(
                renderer,
                iced::advanced::text::Text {
                    content: label,
                    bounds: Size::new(label_rect.width, label_rect.height),
                    size: iced::Pixels(VALUE_FONT_SIZE),
                    line_height: iced::advanced::text::LineHeight::default(),
                    font: iced::Font {
                        weight: iced::font::Weight::Bold,
                        ..Default::default()
                    },
                    align_x: Horizontal::Center.into(),
                    align_y: Vertical::Center,
                    shaping: iced::advanced::text::Shaping::Basic,
                    wrapping: iced::advanced::text::Wrapping::None,
                },
                Point::new(
                    label_rect.x + label_rect.width / 2.0,
                    label_rect.y + label_rect.height / 2.0,
                ),
                palette.background.base.text,
                bounds,
            );
        }
    }

    fn children(&self) -> Vec<Tree> {
        Vec::new()
    }

    fn diff(&self, _tree: &mut Tree) {}
}

/// Convenience conversion into an [`iced::Element`].
pub fn widget_with_layout<'a, Message>(
    state: &'a LoudnessMeterState,
    preferred_width: f32,
    _preferred_height: f32,
) -> Element<'a, Message>
where
    Message: 'a,
{
    Element::new(LoudnessMeter::new(state).with_width(preferred_width))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_aggregates_channels() {
        let mut state = LoudnessMeterState::new();
        state.apply_snapshot(&LoudnessSnapshot {
            short_term_loudness: [-12.0, -6.0],
            momentary_loudness: [-10.0, -5.0],
            rms_fast_db: [-15.0, -9.0],
            rms_slow_db: [-14.0, -8.0],
            true_peak_db: [-1.0, -3.0],
        });

        assert!((state.short_term_average() + 9.0).abs() < f32::EPSILON);
        assert_eq!(state.true_peak_db[0], -1.0);
        assert_eq!(state.true_peak_db[1], -3.0);

        let params = state.render_params();
        assert_eq!(params.bars.len(), 2);
        assert_eq!(params.bars[0].fills.len(), 2);
        assert_eq!(params.bars[1].fills.len(), 1);
    }
}
