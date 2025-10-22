//! UI wrapper around the scrolling waveform DSP processor and renderer.

use crate::audio::meter_tap::MeterFormat;
use crate::dsp::waveform::{
    DEFAULT_COLUMN_CAPACITY, MAX_COLUMN_CAPACITY, WaveformConfig,
    WaveformProcessor as CoreWaveformProcessor, WaveformSnapshot,
};
use crate::dsp::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::ui::render::waveform::{PreviewSample, WaveformParams, WaveformPrimitive};
use crate::ui::theme;
use iced::advanced::Renderer as _;
use iced::advanced::renderer::{self, Quad};
use iced::advanced::widget::{Tree, tree};
use iced::advanced::{Layout, Widget, layout, mouse};
use iced::{Background, Color, Element, Length, Rectangle, Size};
use iced_wgpu::primitive::Renderer as _;
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::Instant;

const COLUMN_PIXEL_WIDTH: f32 = 2.0;
const DEFAULT_FILL_ALPHA: f32 = 1.0;
const DEFAULT_LINE_ALPHA: f32 = 1.0;
const DEFAULT_VERTICAL_PADDING: f32 = 8.0;
const DEFAULT_CHANNEL_GAP: f32 = 12.0;
const DEFAULT_AMPLITUDE_SCALE: f32 = 1.0;
const DEFAULT_STROKE_WIDTH: f32 = 1.0;
#[derive(Debug, Clone)]
pub struct WaveformProcessor {
    inner: CoreWaveformProcessor,
    channels: usize,
}

impl WaveformProcessor {
    pub fn new(sample_rate: f32) -> Self {
        let config = WaveformConfig {
            sample_rate,
            ..WaveformConfig::default()
        };
        Self {
            inner: CoreWaveformProcessor::new(config),
            channels: 2,
        }
    }

    pub fn ingest(&mut self, samples: &[f32], format: MeterFormat) -> Option<WaveformSnapshot> {
        if samples.is_empty() {
            return None;
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
            ProcessorUpdate::Snapshot(snapshot) => Some(snapshot),
            ProcessorUpdate::None => None,
        }
    }

    pub fn update_config(&mut self, config: WaveformConfig) {
        self.inner.update_config(config);
    }

    pub fn config(&self) -> WaveformConfig {
        self.inner.config()
    }
}

#[derive(Debug, Default, Clone)]
struct WaveformRenderCache {
    samples: Arc<Vec<[f32; 2]>>,
    colors: Arc<Vec<[f32; 4]>>,
    preview_samples: Arc<Vec<PreviewSample>>,
}

#[derive(Debug, Clone)]
pub struct WaveformState {
    snapshot: WaveformSnapshot,
    style: WaveformStyle,
    desired_columns: Rc<Cell<usize>>,
    render_key: u64,
    render_cache: RefCell<WaveformRenderCache>,
}

impl WaveformState {
    fn next_render_key() -> u64 {
        static NEXT_RENDER_KEY: AtomicU64 = AtomicU64::new(1);
        NEXT_RENDER_KEY.fetch_add(1, Ordering::Relaxed)
    }

    pub fn new() -> Self {
        Self {
            snapshot: WaveformSnapshot::default(),
            style: WaveformStyle::default(),
            desired_columns: Rc::new(Cell::new(DEFAULT_COLUMN_CAPACITY)),
            render_key: Self::next_render_key(),
            render_cache: RefCell::new(WaveformRenderCache::default()),
        }
    }

    pub fn apply_snapshot(&mut self, snapshot: WaveformSnapshot) {
        self.snapshot = snapshot;
    }

    pub fn set_palette(&mut self, palette: &[Color]) {
        self.style.set_palette(palette);
        self.render_key = Self::next_render_key();
    }

    pub fn palette(&self) -> &[Color] {
        self.style.palette()
    }

    pub fn visual(&self, bounds: Rectangle) -> Option<WaveformParams> {
        self.prepare_render_params(bounds)
    }

    fn prepare_render_params(&self, bounds: Rectangle) -> Option<WaveformParams> {
        let channels = self.snapshot.channels.max(1);
        let width = bounds.width;
        if width <= 0.0 {
            return None;
        }

        let mut required = (width / COLUMN_PIXEL_WIDTH).ceil() as usize;
        if required == 0 {
            required = 1;
        }
        let required = required.clamp(1, MAX_COLUMN_CAPACITY);
        self.desired_columns.set(required);

        let columns = self.snapshot.columns;
        if columns == 0 {
            return None;
        }

        if self.snapshot.min_values.len() != columns * channels
            || self.snapshot.max_values.len() != columns * channels
            || self.snapshot.frequency_normalized.len() != columns * channels
        {
            return None;
        }

        let visible = required.min(columns);
        if visible == 0 {
            return None;
        }

        let start = columns - visible;
        let column_width = COLUMN_PIXEL_WIDTH;

        let mut cache = self.render_cache.borrow_mut();
        let WaveformRenderCache {
            samples, colors, ..
        } = &mut *cache;

        let (samples_arc, colors_arc) = {
            let samples_buffer = Arc::make_mut(samples);
            let colors_buffer = Arc::make_mut(colors);
            samples_buffer.clear();
            colors_buffer.clear();
            samples_buffer.reserve(visible * channels);
            colors_buffer.reserve(visible * channels);

            for channel in 0..channels {
                let base = channel * columns;
                let min_slice = &self.snapshot.min_values[base..base + columns];
                let max_slice = &self.snapshot.max_values[base..base + columns];
                let freq_slice = &self.snapshot.frequency_normalized[base..base + columns];

                for idx in start..columns {
                    let mut min_value = min_slice[idx];
                    let mut max_value = max_slice[idx];
                    if min_value > max_value {
                        std::mem::swap(&mut min_value, &mut max_value);
                    }
                    samples_buffer.push([min_value, max_value]);
                    let frequency = freq_slice[idx];
                    colors_buffer.push(theme::color_to_rgba(
                        self.style.color_for_frequency(frequency),
                    ));
                }
            }

            (samples.clone(), colors.clone())
        };

        let preview = &self.snapshot.preview;
        let preview_active = preview.progress > 0.0
            && preview.min_values.len() >= channels
            && preview.max_values.len() >= channels;
        let preview_progress = if preview_active {
            preview.progress.clamp(0.0, 1.0)
        } else {
            0.0
        };

        let preview_samples_arc = {
            let buffer = Arc::make_mut(&mut cache.preview_samples);
            buffer.clear();
            if preview_active {
                let frequency_hints: Vec<f32> = (0..channels)
                    .map(|channel| channel_frequency_hint(&self.snapshot, channel))
                    .collect();
                buffer.reserve(channels);
                for (channel, hint) in frequency_hints.iter().enumerate().take(channels) {
                    let mut min_value = preview.min_values[channel];
                    let mut max_value = preview.max_values[channel];
                    if min_value > max_value {
                        std::mem::swap(&mut min_value, &mut max_value);
                    }
                    min_value = min_value.clamp(-1.0, 1.0);
                    max_value = max_value.clamp(-1.0, 1.0);
                    let frequency = preview
                        .frequency_normalized
                        .get(channel)
                        .copied()
                        .filter(|f| f.is_finite() && *f > 0.0)
                        .unwrap_or(*hint);
                    let color = theme::color_to_rgba(self.style.color_for_frequency(frequency));
                    buffer.push(PreviewSample {
                        min: min_value,
                        max: max_value,
                        color,
                    });
                }
            }
            cache.preview_samples.clone()
        };

        Some(WaveformParams {
            bounds,
            channels,
            column_width,
            columns: visible,
            samples: samples_arc,
            colors: colors_arc,
            preview_samples: preview_samples_arc,
            preview_progress,
            fill_alpha: self.style.fill_alpha,
            line_alpha: self.style.line_alpha,
            vertical_padding: self.style.vertical_padding,
            channel_gap: self.style.channel_gap,
            amplitude_scale: self.style.amplitude_scale,
            stroke_width: self.style.stroke_width,
            instance_key: self.render_key,
        })
    }

    pub fn desired_columns(&self) -> usize {
        self.desired_columns.get()
    }
}

fn channel_frequency_hint(snapshot: &WaveformSnapshot, channel: usize) -> f32 {
    // Prefer the most recent historical frequency for the channel, falling back to the
    // current preview estimate when history is empty.
    let columns = snapshot.columns;
    if columns == 0 {
        return snapshot
            .preview
            .frequency_normalized
            .get(channel)
            .copied()
            .unwrap_or(0.0);
    }

    let base = channel * columns;
    if let Some(freq) = snapshot
        .frequency_normalized
        .get(base..base + columns)
        .and_then(|slice| {
            slice
                .iter()
                .rev()
                .copied()
                .find(|value| value.is_finite() && *value > 0.0)
        })
    {
        return freq;
    }

    snapshot
        .preview
        .frequency_normalized
        .get(channel)
        .copied()
        .unwrap_or(0.0)
}

#[derive(Debug, Clone)]
pub struct WaveformStyle {
    pub background: Color,
    pub fill_alpha: f32,
    pub line_alpha: f32,
    pub vertical_padding: f32,
    pub channel_gap: f32,
    pub amplitude_scale: f32,
    pub stroke_width: f32,
    palette: Vec<Color>,
    gradient: Vec<GradientStop>,
}

impl WaveformStyle {
    fn color_for_frequency(&self, value: f32) -> Color {
        if self.gradient.is_empty() {
            return self
                .palette
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
                return theme::mix_colors(start.color, end.color, alpha);
            }
        }

        self.palette
            .last()
            .copied()
            .unwrap_or_else(theme::accent_primary)
    }

    fn set_palette(&mut self, palette: &[Color]) {
        if self.palette.len() == palette.len()
            && self
                .palette
                .iter()
                .zip(palette)
                .all(|(a, b)| theme::colors_equal(*a, *b))
        {
            return;
        }

        self.palette = palette.to_vec();
        self.gradient = Self::build_gradient(&self.palette);
    }

    fn palette(&self) -> &[Color] {
        &self.palette
    }

    fn build_gradient(palette: &[Color]) -> Vec<GradientStop> {
        match palette.len() {
            0 => Vec::new(),
            1 => vec![GradientStop {
                position: 0.0,
                color: palette[0],
            }],
            len => {
                let last_index = (len - 1) as f32;
                palette
                    .iter()
                    .enumerate()
                    .map(|(index, &color)| GradientStop {
                        position: index as f32 / last_index,
                        color,
                    })
                    .collect()
            }
        }
    }
}

impl Default for WaveformStyle {
    fn default() -> Self {
        let background = theme::with_alpha(theme::base_color(), 0.0);
        let palette = theme::waveform_palette().to_vec();
        let gradient = Self::build_gradient(&palette);

        Self {
            background,
            fill_alpha: DEFAULT_FILL_ALPHA,
            line_alpha: DEFAULT_LINE_ALPHA,
            vertical_padding: DEFAULT_VERTICAL_PADDING,
            channel_gap: DEFAULT_CHANNEL_GAP,
            amplitude_scale: DEFAULT_AMPLITUDE_SCALE,
            stroke_width: DEFAULT_STROKE_WIDTH,
            palette,
            gradient,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct GradientStop {
    position: f32,
    color: Color,
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

        if let Some(params) = self.state.visual(bounds) {
            renderer.draw_primitive(bounds, WaveformPrimitive::new(params));
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
