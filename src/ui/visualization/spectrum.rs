use crate::dsp::spectrum::{
    SpectrumConfig, SpectrumProcessor as CoreSpectrumProcessor, SpectrumSnapshot,
};
use crate::dsp::{AudioBlock, AudioProcessor, ProcessorUpdate};
use crate::ui::render::spectrum::{SpectrumParams, SpectrumPrimitive};
use crate::ui::theme;
use iced::advanced::Renderer as _;
use iced::advanced::graphics::text::Paragraph as RenderParagraph;
use iced::advanced::renderer::{self, Quad};
use iced::advanced::text::{self, LineHeight, Paragraph as _, Renderer as _, Shaping, Wrapping};
use iced::advanced::widget::{Tree, tree};
use iced::advanced::{Layout, Widget, layout, mouse};
use iced::alignment::{Horizontal, Vertical};
use iced::{Background, Color, Element, Font, Length, Pixels, Point, Rectangle, Size};
use iced_wgpu::primitive::Renderer as _;
use std::cell::RefCell;
use std::time::Instant;

const DEFAULT_RESOLUTION: usize = 1024;
const MIN_FREQUENCY_HZ: f32 = 10.0;
const MAX_FREQUENCY_HZ: f32 = 20_000.0;
const EPSILON: f32 = 1.0e-6;
const GRID_LABEL_FONT_SIZE: f32 = 12.0;
const GRID_LABEL_WIDTH: f32 = 72.0;
const GRID_LABEL_HEIGHT: f32 = 18.0;
const GRID_LABEL_PADDING: f32 = 4.0;

/// High-level wrapper around the shared spectrum processor.
pub struct SpectrumProcessor {
    inner: CoreSpectrumProcessor,
    channels: usize,
}

impl std::fmt::Debug for SpectrumProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpectrumProcessor")
            .field("channels", &self.channels)
            .finish()
    }
}

impl SpectrumProcessor {
    pub fn new(sample_rate: f32) -> Self {
        let config = SpectrumConfig {
            sample_rate,
            ..Default::default()
        };
        Self {
            inner: CoreSpectrumProcessor::new(config),
            channels: 2,
        }
    }

    pub fn ingest(&mut self, samples: &[f32]) -> Option<SpectrumSnapshot> {
        if samples.is_empty() {
            return None;
        }

        let block = AudioBlock::new(
            samples,
            self.channels,
            self.inner.config().sample_rate,
            Instant::now(),
        );

        match self.inner.process_block(&block) {
            ProcessorUpdate::Snapshot(snapshot) => Some(snapshot),
            ProcessorUpdate::None => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SpectrumStyle {
    pub background: Color,
    pub line_color: Color,
    pub grid_major_color: Color,
    pub min_db: f32,
    pub max_db: f32,
    pub min_frequency: f32,
    pub max_frequency: f32,
    pub resolution: usize,
    pub line_thickness: f32,
    pub unweighted_line_color: Color,
    pub unweighted_line_thickness: f32,
    pub smoothing_radius: usize,
    pub smoothing_passes: usize,
    pub highlight_threshold: f32,
    pub highlight_color: Color,
}

impl Default for SpectrumStyle {
    fn default() -> Self {
        let background = Color::from_rgba(0.0, 0.0, 0.0, 0.0);
        let line_color = theme::mix_colors(theme::accent_primary(), theme::text_color(), 0.35);
        let major_grid = theme::with_alpha(theme::text_secondary(), 0.22);

        Self {
            background,
            line_color,
            grid_major_color: major_grid,
            min_db: -120.0,
            max_db: 0.0,
            min_frequency: MIN_FREQUENCY_HZ,
            max_frequency: MAX_FREQUENCY_HZ,
            resolution: DEFAULT_RESOLUTION,
            line_thickness: 2.4,
            unweighted_line_color: theme::with_alpha(line_color, 0.45),
            unweighted_line_thickness: 1.6,
            smoothing_radius: 2,
            smoothing_passes: 2,
            highlight_threshold: 0.65,
            highlight_color: theme::with_alpha(theme::accent_primary(), 0.9),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpectrumState {
    style: SpectrumStyle,
    weighted_points: Vec<[f32; 2]>,
    unweighted_points: Vec<[f32; 2]>,
    frequency_grid: Vec<GridLineSpec>,
    min_freq: f32,
    max_freq: f32,
    peak_frequency_hz: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct GridLineSpec {
    pub position: f32,
    pub frequency_hz: f32,
    pub color: Color,
    pub thickness: f32,
}

#[derive(Debug)]
pub struct SpectrumVisual {
    pub primitive: SpectrumParams,
    pub grid_lines: Vec<GridLineSpec>,
    pub background: Color,
}

impl SpectrumState {
    pub fn new() -> Self {
        Self {
            style: SpectrumStyle::default(),
            weighted_points: Vec::new(),
            unweighted_points: Vec::new(),
            frequency_grid: Vec::new(),
            min_freq: MIN_FREQUENCY_HZ,
            max_freq: MAX_FREQUENCY_HZ,
            peak_frequency_hz: None,
        }
    }

    #[allow(dead_code)]
    pub fn style_mut(&mut self) -> &mut SpectrumStyle {
        &mut self.style
    }

    #[allow(dead_code)]
    pub fn peak_frequency_hz(&self) -> Option<f32> {
        self.peak_frequency_hz
    }

    pub fn apply_snapshot(&mut self, snapshot: &SpectrumSnapshot) {
        if snapshot.frequency_bins.is_empty()
            || snapshot.magnitudes_db.is_empty()
            || snapshot.frequency_bins.len() != snapshot.magnitudes_db.len()
        {
            return;
        }

        let nyquist = snapshot
            .frequency_bins
            .last()
            .copied()
            .unwrap_or(self.style.max_frequency);

        let min_freq = self.style.min_frequency.max(EPSILON);
        let mut max_freq = self.style.max_frequency.min(nyquist);
        if max_freq <= min_freq {
            max_freq = nyquist.max(min_freq * 1.02);
        }

        self.min_freq = min_freq;
        self.max_freq = max_freq;
        self.peak_frequency_hz = snapshot.peak_frequency_hz;

        let resolution = self.style.resolution.max(32);
        if self.weighted_points.len() != resolution {
            self.weighted_points = Vec::with_capacity(resolution);
        }
        if self.unweighted_points.len() != resolution {
            self.unweighted_points = Vec::with_capacity(resolution);
        }
        self.weighted_points.clear();
        self.unweighted_points.clear();

        let log_min = min_freq.max(EPSILON).log10();
        let log_max = max_freq.max(min_freq * 1.01).log10();
        let denom = (log_max - log_min).max(EPSILON);

        for i in 0..resolution {
            let t = if resolution == 1 {
                0.0
            } else {
                i as f32 / (resolution - 1) as f32
            };
            let freq = 10.0f32.powf(log_min + denom * t);
            let magnitude_weighted =
                interpolate_magnitude(&snapshot.frequency_bins, &snapshot.magnitudes_db, freq);
            let magnitude_unweighted = interpolate_magnitude(
                &snapshot.frequency_bins,
                &snapshot.magnitudes_unweighted_db,
                freq,
            );
            let normalized_weighted =
                normalize_db(magnitude_weighted, self.style.min_db, self.style.max_db);
            let normalized_unweighted =
                normalize_db(magnitude_unweighted, self.style.min_db, self.style.max_db);
            self.weighted_points.push([t, normalized_weighted]);
            self.unweighted_points.push([t, normalized_unweighted]);
        }

        self.frequency_grid = build_frequency_grid(min_freq, max_freq, self.style.grid_major_color);
        if self.style.smoothing_radius > 0 && self.style.smoothing_passes > 0 {
            smooth_points(
                &mut self.weighted_points,
                self.style.smoothing_radius,
                self.style.smoothing_passes,
            );
            smooth_points(
                &mut self.unweighted_points,
                self.style.smoothing_radius,
                self.style.smoothing_passes,
            );
        }
    }

    pub fn visual(&self, bounds: Rectangle) -> Option<SpectrumVisual> {
        if self.weighted_points.len() < 2 {
            return None;
        }

        let normalized_points = self.weighted_points.clone();
        let unweighted_points = self.unweighted_points.clone();

        let grid_lines = self.frequency_grid.to_vec();

        let primitive = SpectrumParams {
            bounds,
            normalized_points,
            secondary_points: unweighted_points,
            line_color: theme::color_to_rgba(self.style.line_color),
            line_width: self.style.line_thickness,
            secondary_line_color: theme::color_to_rgba(self.style.unweighted_line_color),
            secondary_line_width: self.style.unweighted_line_thickness,
            highlight_threshold: self.style.highlight_threshold,
            highlight_color: theme::color_to_rgba(self.style.highlight_color),
        };

        Some(SpectrumVisual {
            primitive,
            grid_lines,
            background: self.style.background,
        })
    }
}

#[derive(Debug)]
pub struct Spectrum<'a> {
    state: &'a SpectrumState,
}

impl<'a> Spectrum<'a> {
    pub fn new(state: &'a SpectrumState) -> Self {
        Self { state }
    }
}

impl<'a, Message> Widget<Message, iced::Theme, iced::Renderer> for Spectrum<'a> {
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<FrequencyLabelCache>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(FrequencyLabelCache::default())
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
        tree: &Tree,
        renderer: &mut iced::Renderer,
        _theme: &iced::Theme,
        _style: &renderer::Style,
        layout: Layout<'_>,
        _cursor: mouse::Cursor,
        _viewport: &Rectangle,
    ) {
        let bounds = layout.bounds();
        let cache = tree.state.downcast_ref::<FrequencyLabelCache>();
        let mut cache_entries = cache.entries.borrow_mut();

        let Some(visual) = self.state.visual(bounds) else {
            cache_entries.clear();
            renderer.fill_quad(
                Quad {
                    bounds,
                    border: Default::default(),
                    shadow: Default::default(),
                },
                Background::Color(self.state.style.background),
            );
            return;
        };

        renderer.fill_quad(
            Quad {
                bounds,
                border: Default::default(),
                shadow: Default::default(),
            },
            Background::Color(visual.background),
        );

        for line in &visual.grid_lines {
            let x = bounds.x + bounds.width * line.position;
            let half = line.thickness * 0.5;
            let rect = Rectangle {
                x: x - half,
                y: bounds.y,
                width: line.thickness,
                height: bounds.height,
            };
            renderer.fill_quad(
                Quad {
                    bounds: rect,
                    border: Default::default(),
                    shadow: Default::default(),
                },
                Background::Color(line.color),
            );
        }

        renderer.draw_primitive(bounds, SpectrumPrimitive::new(visual.primitive));

        let label_color = theme::with_alpha(theme::text_secondary(), 0.85);
        let clip_bounds = Rectangle {
            x: bounds.x,
            y: bounds.y,
            width: bounds.width,
            height: GRID_LABEL_HEIGHT + GRID_LABEL_PADDING * 2.0,
        };

        let mut draw_commands: Vec<(usize, Point)> = Vec::with_capacity(visual.grid_lines.len());
        let mut active_labels: Vec<String> = Vec::with_capacity(visual.grid_lines.len());

        for line in &visual.grid_lines {
            let label = format_frequency_label(line.frequency_hz);
            if label.is_empty() {
                continue;
            }

            let label_bounds = Size::new(GRID_LABEL_WIDTH, GRID_LABEL_HEIGHT);
            let index = ensure_grid_label(&mut cache_entries, label.as_str(), label_bounds);
            let paragraph = &cache_entries[index].paragraph;
            let text_bounds = paragraph.min_bounds();
            if text_bounds.width <= 0.0 || clip_bounds.width <= 0.0 {
                continue;
            }

            let center_x = bounds.x + bounds.width * line.position;
            let mut x = center_x - text_bounds.width * 0.5;
            let max_x = bounds.x + bounds.width - text_bounds.width;
            if max_x < bounds.x {
                x = bounds.x;
            } else {
                x = x.clamp(bounds.x, max_x);
            }
            let y = bounds.y + GRID_LABEL_PADDING;
            let position = Point::new(x, y);
            draw_commands.push((index, position));
            active_labels.push(label);
        }

        for (index, position) in draw_commands {
            if let Some(entry) = cache_entries.get(index) {
                renderer.fill_paragraph(&entry.paragraph, position, label_color, clip_bounds);
            }
        }

        prune_grid_labels(&mut cache_entries, &active_labels);
    }

    fn children(&self) -> Vec<Tree> {
        Vec::new()
    }

    fn diff(&self, _tree: &mut Tree) {}
}

pub fn widget<'a, Message>(state: &'a SpectrumState) -> Element<'a, Message>
where
    Message: 'a,
{
    Element::new(Spectrum::new(state))
}

fn normalize_db(value: f32, min_db: f32, max_db: f32) -> f32 {
    if (max_db - min_db).abs() <= EPSILON {
        return 0.0;
    }
    ((value - min_db) / (max_db - min_db)).clamp(0.0, 1.0)
}

fn interpolate_magnitude(bins: &[f32], magnitudes: &[f32], target: f32) -> f32 {
    if bins.is_empty() || magnitudes.is_empty() {
        return 0.0;
    }

    if target <= bins[0] {
        return magnitudes[0];
    }
    if target >= *bins.last().unwrap() {
        return *magnitudes.last().unwrap();
    }

    match bins.binary_search_by(|probe| probe.partial_cmp(&target).unwrap()) {
        Ok(index) => magnitudes[index],
        Err(index) => {
            let upper = index.min(bins.len() - 1);
            let lower = upper.saturating_sub(1);
            let span = (bins[upper] - bins[lower]).max(EPSILON);
            let t = (target - bins[lower]) / span;
            magnitudes[lower] + (magnitudes[upper] - magnitudes[lower]) * t
        }
    }
}

fn build_frequency_grid(min_freq: f32, max_freq: f32, major_color: Color) -> Vec<GridLineSpec> {
    const STANDARD_FREQUENCIES: &[f32] = &[
        31.5, 63.0, 125.0, 250.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0, 16_000.0,
    ];

    let log_min = min_freq.max(EPSILON).log10();
    let log_max = max_freq.max(min_freq * 1.01).log10();
    let denom = (log_max - log_min).max(EPSILON);

    let mut lines: Vec<GridLineSpec> = STANDARD_FREQUENCIES
        .iter()
        .copied()
        .filter(|f| *f >= min_freq && *f <= max_freq)
        .map(|frequency| {
            let ratio = (frequency.log10() - log_min) / denom;
            GridLineSpec {
                position: ratio.clamp(0.0, 1.0),
                frequency_hz: frequency,
                color: major_color,
                thickness: 1.5,
            }
        })
        .collect();

    if lines.is_empty() {
        lines.push(GridLineSpec {
            position: 0.0,
            frequency_hz: min_freq,
            color: major_color,
            thickness: 1.5,
        });
        lines.push(GridLineSpec {
            position: 1.0,
            frequency_hz: max_freq,
            color: major_color,
            thickness: 1.5,
        });
    }

    lines
}

fn ensure_grid_label(entries: &mut Vec<GridLabelParagraph>, label: &str, bounds: Size) -> usize {
    if let Some((index, entry)) = entries
        .iter_mut()
        .enumerate()
        .find(|(_, entry)| entry.label == label)
    {
        entry.ensure(label, bounds);
        index
    } else {
        entries.push(GridLabelParagraph::new(label, bounds));
        entries.len() - 1
    }
}

fn prune_grid_labels(entries: &mut Vec<GridLabelParagraph>, active: &[String]) {
    if active.is_empty() {
        entries.clear();
    } else {
        entries.retain(|entry| active.iter().any(|label| label == &entry.label));
    }
}

#[derive(Default)]
struct FrequencyLabelCache {
    entries: RefCell<Vec<GridLabelParagraph>>,
}

struct GridLabelParagraph {
    label: String,
    bounds: Size,
    paragraph: RenderParagraph,
}

impl GridLabelParagraph {
    fn new(label: &str, bounds: Size) -> Self {
        Self {
            label: label.to_owned(),
            bounds,
            paragraph: build_grid_label_paragraph(label, bounds),
        }
    }

    fn ensure(&mut self, label: &str, bounds: Size) {
        if self.label != label {
            self.label = label.to_owned();
            self.paragraph = build_grid_label_paragraph(label, bounds);
            self.bounds = bounds;
        } else if !size_eq(self.bounds, bounds) {
            self.paragraph.resize(bounds);
            self.bounds = bounds;
        }
    }
}

fn build_grid_label_paragraph(label: &str, bounds: Size) -> RenderParagraph {
    RenderParagraph::with_text(text::Text {
        content: label,
        bounds,
        size: Pixels(GRID_LABEL_FONT_SIZE),
        line_height: LineHeight::Relative(1.0),
        font: Font::default(),
        horizontal_alignment: Horizontal::Center,
        vertical_alignment: Vertical::Top,
        shaping: Shaping::Advanced,
        wrapping: Wrapping::None,
    })
}

fn size_eq(a: Size, b: Size) -> bool {
    (a.width - b.width).abs() <= f32::EPSILON && (a.height - b.height).abs() <= f32::EPSILON
}

fn format_frequency_label(frequency_hz: f32) -> String {
    if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
        return String::new();
    }

    if frequency_hz >= 1_000.0 {
        let kilo = frequency_hz / 1_000.0;
        if (kilo - kilo.round()).abs() < 0.05 {
            format!("{:.0}k", kilo.round())
        } else {
            format!("{:.1}k", kilo)
        }
    } else if (frequency_hz - frequency_hz.round()).abs() < 0.05 {
        format!("{:.0}", frequency_hz.round())
    } else {
        format!("{:.1}", frequency_hz)
    }
}

fn smooth_points(points: &mut [[f32; 2]], radius: usize, passes: usize) {
    if radius == 0 || passes == 0 || points.len() < 3 {
        return;
    }

    let len = points.len();
    let mut scratch = vec![0.0f32; len];

    for _ in 0..passes {
        for (dst, point) in scratch.iter_mut().zip(points.iter()) {
            *dst = point[1];
        }

        for (i, point) in points.iter_mut().enumerate().take(len) {
            let start = i.saturating_sub(radius);
            let end = (i + radius).min(len - 1);
            let mut weight_sum = 0.0;
            let mut value_sum = 0.0;

            for (j, &scratch_value) in scratch.iter().enumerate().take(end + 1).skip(start) {
                let distance = i.abs_diff(j);
                let weight = (radius - distance + 1) as f32;
                value_sum += scratch_value * weight;
                weight_sum += weight;
            }

            if weight_sum > 0.0 {
                point[1] = value_sum / weight_sum;
            }
        }
    }
}
