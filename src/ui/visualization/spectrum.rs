use crate::audio::meter_tap::MeterFormat;
use crate::dsp::spectrogram::{FrequencyScale, hz_to_mel, mel_to_hz};
use crate::dsp::spectrum::{
    SpectrumConfig, SpectrumProcessor as CoreSpectrumProcessor, SpectrumSnapshot,
};
use crate::dsp::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::ui::render::spectrum::{SpectrumParams, SpectrumPrimitive};
use crate::ui::theme;
use crate::util::audio::lerp;
use crate::util::audio::musical::MusicalNote;
use iced::advanced::Renderer as _;
use iced::advanced::graphics::text::Paragraph as RenderParagraph;
use iced::advanced::renderer::{self, Quad};
use iced::advanced::text::{self, Paragraph as _, Renderer as _};
use iced::advanced::widget::{Tree, tree};
use iced::advanced::{Layout, Widget, layout, mouse};
use iced::{Background, Color, Element, Length, Point, Rectangle, Size};
use iced_wgpu::primitive::Renderer as _;
use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const DEFAULT_RESOLUTION: usize = 1024;
const MIN_FREQUENCY_HZ: f32 = 10.0;
const MAX_FREQUENCY_HZ: f32 = 20_000.0;
const EPSILON: f32 = 1e-6;
const LABEL_TEXT_SIZE: f32 = 12.0;
const LABEL_PADDING: f32 = 4.0;
const LABEL_OFFSET: f32 = 8.0;
const PEAK_LABEL_ACTIVITY_THRESHOLD: f32 = 0.08;
const PEAK_LABEL_FADE_IN_LERP: f32 = 0.35;
const PEAK_LABEL_FADE_OUT_LERP: f32 = 0.12;
const PEAK_LABEL_MIN_OPACITY: f32 = 0.01;
const GRID_TEXT_SIZE: f32 = 10.0;
const GRID_LABEL_MARGIN: f32 = 6.0;
const GRID_LINE_ALPHA: f32 = 0.25;
const GRID_LABEL_ALPHA: f32 = 0.75;
const GRID_LINE_THICKNESS: f32 = 1.0;
const GRID_MIN_LABEL_SPACING: f32 = 36.0;
const GRID_LABEL_GAP_PADDING: f32 = 4.0;

const GRID_FREQUENCY_TABLE: &[(f32, u8)] = &[
    (10.0, 0),
    (20.0, 2),
    (31.5, 3),
    (40.0, 2),
    (50.0, 2),
    (63.0, 3),
    (80.0, 2),
    (100.0, 1),
    (125.0, 2),
    (160.0, 2),
    (200.0, 1),
    (250.0, 2),
    (315.0, 3),
    (400.0, 2),
    (500.0, 1),
    (630.0, 2),
    (800.0, 2),
    (1_000.0, 0),
    (1_250.0, 2),
    (1_600.0, 2),
    (2_000.0, 1),
    (2_500.0, 2),
    (3_150.0, 3),
    (4_000.0, 1),
    (5_000.0, 2),
    (6_300.0, 3),
    (8_000.0, 1),
    (10_000.0, 0),
    (16_000.0, 1),
];

static NEXT_POINT_SERIES_ID: AtomicUsize = AtomicUsize::new(1);

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

    pub fn ingest(&mut self, samples: &[f32], format: MeterFormat) -> Option<SpectrumSnapshot> {
        if samples.is_empty() {
            return None;
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
            ProcessorUpdate::Snapshot(snapshot) => Some(snapshot),
            ProcessorUpdate::None => None,
        }
    }

    pub fn update_config(&mut self, config: SpectrumConfig) {
        self.inner.update_config(config);
    }

    pub fn config(&self) -> SpectrumConfig {
        self.inner.config()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SpectrumStyle {
    pub min_db: f32,
    pub max_db: f32,
    pub min_frequency: f32,
    pub max_frequency: f32,
    pub resolution: usize,
    pub line_thickness: f32,
    pub unweighted_line_thickness: f32,
    pub smoothing_radius: usize,
    pub smoothing_passes: usize,
    pub highlight_threshold: f32,
    pub spectrum_palette: [Color; 5],
    pub frequency_scale: FrequencyScale,
    pub reverse_frequency: bool,
    pub show_grid: bool,
    pub show_peak_label: bool,
}

impl Default for SpectrumStyle {
    fn default() -> Self {
        Self {
            min_db: -80.0,
            max_db: 0.0,
            min_frequency: MIN_FREQUENCY_HZ,
            max_frequency: MAX_FREQUENCY_HZ,
            resolution: DEFAULT_RESOLUTION,
            line_thickness: 1.0,
            unweighted_line_thickness: 1.6,
            smoothing_radius: 0,
            smoothing_passes: 0,
            highlight_threshold: 0.45,
            spectrum_palette: theme::DEFAULT_SPECTRUM_PALETTE,
            frequency_scale: FrequencyScale::Logarithmic,
            reverse_frequency: false,
            show_grid: true,
            show_peak_label: true,
        }
    }
}

fn build_peak_label(
    snapshot: &SpectrumSnapshot,
    scale: &ScaleContext,
    style: &SpectrumStyle,
) -> Option<(String, f32, f32)> {
    let freq = snapshot.peak_frequency_hz?;
    if !freq.is_finite() || freq <= 0.0 {
        return None;
    }

    let mut normalized_x = scale.normalized_position_of(style.frequency_scale, freq);
    if style.reverse_frequency {
        normalized_x = 1.0 - normalized_x;
    }

    let magnitude = interpolate_magnitude(&snapshot.frequency_bins, &snapshot.magnitudes_db, freq);
    let normalized_y = normalize_db(magnitude, style.min_db, style.max_db);
    if normalized_y < PEAK_LABEL_ACTIVITY_THRESHOLD {
        return None;
    }

    let text = match MusicalNote::from_frequency(freq) {
        Some(note) => format!("{:.1} Hz | {}", freq, note.format()),
        None => format!("{:.1} Hz", freq),
    };

    Some((
        text,
        normalized_x.clamp(0.0, 1.0),
        normalized_y.clamp(0.0, 1.0),
    ))
}

#[derive(Debug, Clone)]
pub struct SpectrumState {
    style: SpectrumStyle,
    points: PointSeries,
    peak_label: Option<PeakLabel>,
    grid_lines: Arc<Vec<GridLine>>,
}

#[derive(Debug)]
pub struct SpectrumVisual {
    pub primitive: SpectrumParams,
    pub background: Color,
    label: Option<PeakLabel>,
    grid_lines: Arc<Vec<GridLine>>,
}

#[derive(Debug, Clone)]
struct PeakLabel {
    text: String,
    normalized_x: f32,
    normalized_y: f32,
    opacity: f32,
}

impl PeakLabel {
    fn new(text: String, normalized_x: f32, normalized_y: f32) -> Self {
        Self {
            text,
            normalized_x,
            normalized_y,
            opacity: 0.0,
        }
    }

    fn update(&mut self, text: String, normalized_x: f32, normalized_y: f32) {
        self.text = text;
        self.normalized_x = normalized_x;
        self.normalized_y = normalized_y;
    }

    fn fade_toward(&mut self, target: f32, rate: f32) {
        self.opacity += (target - self.opacity) * rate;
    }
}

#[derive(Debug, Clone)]
struct GridLine {
    position: f32,
    label: String,
    importance: u8,
}

struct GridCandidate<'a> {
    line: &'a GridLine,
    screen_x: f32,
    clamped_x: f32,
    line_bounds: Rectangle,
    text_origin: Point,
    text_size: Size,
}

impl SpectrumState {
    pub fn new() -> Self {
        Self {
            style: SpectrumStyle::default(),
            points: PointSeries::new(),
            peak_label: None,
            grid_lines: Arc::new(Vec::new()),
        }
    }

    pub fn style(&self) -> &SpectrumStyle {
        &self.style
    }

    pub fn style_mut(&mut self) -> &mut SpectrumStyle {
        &mut self.style
    }

    pub fn update_show_grid(&mut self, show: bool) {
        self.style.show_grid = show;
        if !show {
            self.grid_lines = Arc::new(Vec::new());
        }
    }

    pub fn update_show_peak_label(&mut self, show: bool) {
        self.style.show_peak_label = show;
        if !show {
            self.peak_label = None;
        }
    }

    pub fn set_palette(&mut self, palette: &[Color]) {
        if palette.len() != self.style.spectrum_palette.len() {
            return;
        }

        if self
            .style
            .spectrum_palette
            .iter()
            .zip(palette)
            .all(|(a, b)| theme::colors_equal(*a, *b))
        {
            return;
        }

        self.style.spectrum_palette.copy_from_slice(palette);
    }

    pub fn palette(&self) -> [Color; 5] {
        self.style.spectrum_palette
    }

    pub fn apply_snapshot(&mut self, snapshot: &SpectrumSnapshot) {
        if snapshot.frequency_bins.is_empty()
            || snapshot.magnitudes_db.is_empty()
            || snapshot.frequency_bins.len() != snapshot.magnitudes_db.len()
        {
            self.update_peak_label_state(None);
            return;
        }

        let Some((scale, resolution)) = domain_parameters(&self.style, snapshot) else {
            self.update_peak_label_state(None);
            return;
        };

        self.points
            .update(&self.style, &scale, snapshot, resolution);

        self.grid_lines = if self.style.show_grid {
            build_grid_lines(&self.style, &scale)
        } else {
            Arc::new(Vec::new())
        };

        let peak_data = self
            .style
            .show_peak_label
            .then(|| build_peak_label(snapshot, &scale, &self.style))
            .flatten();

        self.update_peak_label_state(peak_data);
    }

    fn update_peak_label_state(&mut self, data: Option<(String, f32, f32)>) {
        if data.is_none() && self.peak_label.is_none() {
            return;
        }

        let (target_opacity, fade_rate) = if let Some((text, x, y)) = data {
            let label = self
                .peak_label
                .get_or_insert_with(|| PeakLabel::new(text.clone(), x, y));
            label.update(text, x, y);
            (1.0, PEAK_LABEL_FADE_IN_LERP)
        } else {
            (0.0, PEAK_LABEL_FADE_OUT_LERP)
        };

        if let Some(label) = self.peak_label.as_mut() {
            label.fade_toward(target_opacity, fade_rate);

            if target_opacity <= 0.0 && label.opacity <= PEAK_LABEL_MIN_OPACITY {
                self.peak_label = None;
            }
        }
    }

    pub fn visual(&self, bounds: Rectangle, theme: &iced::Theme) -> Option<SpectrumVisual> {
        if !self.points.is_ready() {
            return None;
        }

        let palette = theme.extended_palette();
        let line_color = theme::mix_colors(
            palette.primary.base.color,
            palette.background.base.text,
            0.35,
        );
        let secondary_line_color = theme::with_alpha(palette.secondary.weak.text, 0.3);

        Some(SpectrumVisual {
            primitive: SpectrumParams {
                bounds,
                normalized_points: Arc::clone(&self.points.weighted),
                secondary_points: Arc::clone(&self.points.unweighted),
                instance_key: self.points.instance_key,
                line_color: theme::color_to_rgba(line_color),
                line_width: self.style.line_thickness,
                secondary_line_color: theme::color_to_rgba(secondary_line_color),
                secondary_line_width: self.style.unweighted_line_thickness,
                highlight_threshold: self.style.highlight_threshold,
                spectrum_palette: self
                    .style
                    .spectrum_palette
                    .iter()
                    .map(|&c| theme::color_to_rgba(c))
                    .collect(),
            },
            background: palette.background.base.color,
            label: self
                .style
                .show_peak_label
                .then(|| self.peak_label.clone())
                .flatten(),
            grid_lines: Arc::clone(&self.grid_lines),
        })
    }
}

#[derive(Debug)]
pub struct Spectrum<'a> {
    state: &'a RefCell<SpectrumState>,
}

impl<'a> Spectrum<'a> {
    pub fn new(state: &'a RefCell<SpectrumState>) -> Self {
        Self { state }
    }
}

impl<'a, Message> Widget<Message, iced::Theme, iced::Renderer> for Spectrum<'a> {
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<()>()
    }

    fn state(&self) -> tree::State {
        tree::State::None
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

        let Some(visual) = self.state.borrow().visual(bounds, theme) else {
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

        renderer.fill_quad(
            Quad {
                bounds,
                border: Default::default(),
                shadow: Default::default(),
            },
            Background::Color(visual.background),
        );

        let grid_lines = visual.grid_lines.clone();
        if !grid_lines.is_empty() {
            renderer.with_layer(bounds, |renderer| {
                draw_grid_lines(renderer, theme, bounds, grid_lines.as_ref());
            });
        }

        renderer.draw_primitive(bounds, SpectrumPrimitive::new(visual.primitive));

        if let Some(label) = visual.label.as_ref() {
            renderer.with_layer(bounds, |renderer| {
                draw_peak_label(renderer, theme, bounds, label);
            });
        }
    }

    fn children(&self) -> Vec<Tree> {
        Vec::new()
    }

    fn diff(&self, _tree: &mut Tree) {}
}

pub fn widget<'a, Message>(state: &'a RefCell<SpectrumState>) -> Element<'a, Message>
where
    Message: 'a,
{
    Element::new(Spectrum::new(state))
}

#[derive(Debug, Clone)]
struct PointSeries {
    weighted: Arc<Vec<[f32; 2]>>,
    unweighted: Arc<Vec<[f32; 2]>>,
    smoothing_scratch: Vec<f32>,
    instance_key: usize,
}

impl PointSeries {
    fn new() -> Self {
        Self {
            weighted: Arc::new(Vec::new()),
            unweighted: Arc::new(Vec::new()),
            smoothing_scratch: Vec::new(),
            instance_key: NEXT_POINT_SERIES_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    fn is_ready(&self) -> bool {
        self.weighted.len() >= 2
    }

    fn update(
        &mut self,
        style: &SpectrumStyle,
        scale: &ScaleContext,
        snapshot: &SpectrumSnapshot,
        resolution: usize,
    ) {
        let weighted = Arc::make_mut(&mut self.weighted);
        let unweighted = Arc::make_mut(&mut self.unweighted);

        rebuild_point_sets(weighted, unweighted, resolution, style, scale, snapshot);

        apply_smoothing_if_needed(
            style,
            weighted.as_mut_slice(),
            unweighted.as_mut_slice(),
            &mut self.smoothing_scratch,
        );

        if style.reverse_frequency {
            reverse_and_reindex(weighted);
            reverse_and_reindex(unweighted);
        }
    }
}

fn domain_parameters(
    style: &SpectrumStyle,
    snapshot: &SpectrumSnapshot,
) -> Option<(ScaleContext, usize)> {
    let nyquist = snapshot
        .frequency_bins
        .last()
        .copied()
        .unwrap_or(style.max_frequency);

    let min_freq = style.min_frequency.max(EPSILON);
    let mut max_freq = style.max_frequency.min(nyquist);
    if max_freq <= min_freq {
        max_freq = nyquist.max(min_freq * 1.02);
    }

    (max_freq > min_freq).then(|| {
        (
            ScaleContext::new(min_freq, max_freq),
            style.resolution.max(32),
        )
    })
}

#[derive(Debug, Clone, Copy)]
struct ScaleContext {
    min_hz: f32,
    max_hz: f32,
    log_min: f32,
    log_range: f32,
    mel_min: f32,
    mel_range: f32,
}

impl ScaleContext {
    fn new(min_hz: f32, max_hz: f32) -> Self {
        let log_min = min_hz.max(EPSILON).log10();
        let log_max = max_hz.max(min_hz * 1.01).log10();
        let log_range = (log_max - log_min).max(EPSILON);

        let mel_min = hz_to_mel(min_hz);
        let mel_max = hz_to_mel(max_hz);
        let mel_range = (mel_max - mel_min).max(EPSILON);

        Self {
            min_hz,
            max_hz,
            log_min,
            log_range,
            mel_min,
            mel_range,
        }
    }

    fn frequency_for(&self, scale: FrequencyScale, t: f32) -> f32 {
        match scale {
            FrequencyScale::Linear => self.min_hz + (self.max_hz - self.min_hz) * t,
            FrequencyScale::Logarithmic => 10.0f32.powf(self.log_min + self.log_range * t),
            FrequencyScale::Mel => {
                let mel = self.mel_min + self.mel_range * t;
                mel_to_hz(mel)
            }
        }
    }

    fn normalized_position_of(&self, scale: FrequencyScale, freq: f32) -> f32 {
        let freq = freq.clamp(self.min_hz, self.max_hz);
        let t = match scale {
            FrequencyScale::Linear => {
                let span = (self.max_hz - self.min_hz).max(EPSILON);
                (freq - self.min_hz) / span
            }
            FrequencyScale::Logarithmic => {
                let freq_log = freq.max(EPSILON).log10();
                (freq_log - self.log_min) / self.log_range
            }
            FrequencyScale::Mel => {
                let mel = hz_to_mel(freq);
                (mel - self.mel_min) / self.mel_range
            }
        };
        t.clamp(0.0, 1.0)
    }
}

fn rebuild_point_sets(
    weighted: &mut Vec<[f32; 2]>,
    unweighted: &mut Vec<[f32; 2]>,
    resolution: usize,
    style: &SpectrumStyle,
    scale: &ScaleContext,
    snapshot: &SpectrumSnapshot,
) {
    weighted.clear();
    unweighted.clear();
    weighted.reserve(resolution);
    unweighted.reserve(resolution);

    for step in 0..resolution {
        let t = normalized_position(step, resolution);
        let freq = scale.frequency_for(style.frequency_scale, t);

        let magnitude_weighted =
            interpolate_magnitude(&snapshot.frequency_bins, &snapshot.magnitudes_db, freq);
        let magnitude_unweighted = interpolate_magnitude(
            &snapshot.frequency_bins,
            &snapshot.magnitudes_unweighted_db,
            freq,
        );

        let normalized_weighted = normalize_db(magnitude_weighted, style.min_db, style.max_db);
        let normalized_unweighted = normalize_db(magnitude_unweighted, style.min_db, style.max_db);

        weighted.push([t, normalized_weighted]);
        unweighted.push([t, normalized_unweighted]);
    }
}

fn apply_smoothing_if_needed(
    style: &SpectrumStyle,
    weighted: &mut [[f32; 2]],
    unweighted: &mut [[f32; 2]],
    scratch: &mut Vec<f32>,
) {
    if style.smoothing_radius == 0 || style.smoothing_passes == 0 {
        return;
    }

    smooth_points(
        weighted,
        style.smoothing_radius,
        style.smoothing_passes,
        scratch,
    );
    smooth_points(
        unweighted,
        style.smoothing_radius,
        style.smoothing_passes,
        scratch,
    );
}

fn reverse_and_reindex(points: &mut Vec<[f32; 2]>) {
    if points.is_empty() {
        return;
    }

    points.reverse();
    refresh_abscissa(points.as_mut_slice());
}

fn refresh_abscissa(points: &mut [[f32; 2]]) {
    let count = points.len();
    for (index, point) in points.iter_mut().enumerate() {
        point[0] = normalized_position(index, count);
    }
}

fn normalized_position(index: usize, count: usize) -> f32 {
    if count <= 1 {
        0.0
    } else {
        index as f32 / (count - 1) as f32
    }
}

fn normalize_db(value: f32, min_db: f32, max_db: f32) -> f32 {
    ((value - min_db) / (max_db - min_db).max(EPSILON)).clamp(0.0, 1.0)
}

fn interpolate_magnitude(bins: &[f32], magnitudes: &[f32], target: f32) -> f32 {
    if bins.is_empty() || target <= bins[0] {
        return magnitudes.first().copied().unwrap_or(0.0);
    }

    if let Some(&last_bin) = bins.last()
        && target >= last_bin
    {
        return magnitudes.last().copied().unwrap_or(0.0);
    }

    match bins.binary_search_by(|probe| {
        probe
            .partial_cmp(&target)
            .unwrap_or(std::cmp::Ordering::Less)
    }) {
        Ok(index) => magnitudes.get(index).copied().unwrap_or(0.0),
        Err(index) => {
            let upper = index.min(bins.len() - 1);
            let lower = upper.saturating_sub(1);
            let t = (target - bins[lower]) / (bins[upper] - bins[lower]).max(EPSILON);
            let lower_mag = magnitudes.get(lower).copied().unwrap_or(0.0);
            let upper_mag = magnitudes.get(upper).copied().unwrap_or(lower_mag);
            lerp(lower_mag, upper_mag, t)
        }
    }
}

fn smooth_points(points: &mut [[f32; 2]], radius: usize, passes: usize, scratch: &mut Vec<f32>) {
    if radius == 0 || passes == 0 || points.len() < 3 {
        return;
    }

    let len = points.len();
    scratch.resize(len, 0.0);

    for _ in 0..passes {
        for (dst, point) in scratch.iter_mut().zip(points.iter()) {
            *dst = point[1];
        }

        for (i, point) in points.iter_mut().enumerate() {
            let start = i.saturating_sub(radius);
            let end = (i + radius + 1).min(len);
            let mut sum = 0.0;
            let mut weight_sum = 0.0;

            for (j, &value) in scratch[start..end].iter().enumerate() {
                let distance = (start + j).abs_diff(i);
                let weight = (radius - distance + 1) as f32;
                sum += value * weight;
                weight_sum += weight;
            }

            point[1] = sum / weight_sum;
        }
    }
}

fn build_grid_lines(style: &SpectrumStyle, scale: &ScaleContext) -> Arc<Vec<GridLine>> {
    let min_hz = scale.min_hz;
    let max_hz = scale.max_hz;
    let mut lines = Vec::new();

    for &(freq, importance) in GRID_FREQUENCY_TABLE.iter() {
        if freq < min_hz || freq > max_hz {
            continue;
        }

        let mut position = scale.normalized_position_of(style.frequency_scale, freq);
        if style.reverse_frequency {
            position = 1.0 - position;
        }

        if !position.is_finite() {
            continue;
        }

        lines.push(GridLine {
            position: position.clamp(0.0, 1.0),
            label: format_frequency_label(freq),
            importance,
        });
    }

    Arc::new(lines)
}

fn draw_grid_lines(
    renderer: &mut iced::Renderer,
    theme: &iced::Theme,
    bounds: Rectangle,
    lines: &[GridLine],
) {
    if bounds.width <= 0.0 || bounds.height <= 0.0 {
        return;
    }

    let palette = theme.extended_palette();
    let line_color = theme::with_alpha(palette.background.base.text, GRID_LINE_ALPHA);
    let label_color = theme::with_alpha(palette.background.base.text, GRID_LABEL_ALPHA);

    let mut candidates = Vec::with_capacity(lines.len());
    for line in lines {
        let x = bounds.x + bounds.width * line.position;
        if !x.is_finite() || x < bounds.x - 1.0 || x > bounds.x + bounds.width + 1.0 {
            continue;
        }

        let text_size = RenderParagraph::with_text(text::Text {
            content: line.label.as_str(),
            bounds: Size::INFINITY,
            size: iced::Pixels(GRID_TEXT_SIZE),
            font: iced::Font::default(),
            horizontal_alignment: iced::alignment::Horizontal::Left,
            vertical_alignment: iced::alignment::Vertical::Top,
            line_height: text::LineHeight::default(),
            shaping: text::Shaping::Basic,
            wrapping: text::Wrapping::None,
        })
        .min_bounds();

        if text_size.width <= 0.0 || text_size.height <= 0.0 {
            continue;
        }

        let text_y = bounds.y + GRID_LABEL_MARGIN;
        let line_top = text_y + text_size.height + GRID_LABEL_MARGIN;
        let line_height = (bounds.y + bounds.height - line_top).max(0.0);

        let clamped_x = x.clamp(bounds.x, bounds.x + bounds.width);
        let line_x = (clamped_x - GRID_LINE_THICKNESS * 0.5)
            .clamp(bounds.x, bounds.x + bounds.width - GRID_LINE_THICKNESS);
        let line_bounds = Rectangle::new(
            Point::new(line_x, line_top),
            Size::new(GRID_LINE_THICKNESS, line_height),
        );

        let min_x = bounds.x + GRID_LABEL_MARGIN;
        let max_x = (bounds.x + bounds.width - GRID_LABEL_MARGIN - text_size.width).max(min_x);
        let text_x = (x - text_size.width * 0.5).clamp(min_x, max_x);
        let text_origin = Point::new(text_x, text_y);

        candidates.push(GridCandidate {
            line,
            screen_x: x,
            clamped_x,
            line_bounds,
            text_origin,
            text_size,
        });
    }

    let accepted_indices = select_grid_label_indices(&candidates);
    for index in accepted_indices {
        let candidate = &candidates[index];

        if candidate.line_bounds.height > 0.0 {
            renderer.fill_quad(
                Quad {
                    bounds: candidate.line_bounds,
                    border: Default::default(),
                    shadow: Default::default(),
                },
                Background::Color(line_color),
            );
        }

        renderer.fill_text(
            text::Text {
                content: candidate.line.label.clone(),
                bounds: candidate.text_size,
                size: iced::Pixels(GRID_TEXT_SIZE),
                font: iced::Font::default(),
                horizontal_alignment: iced::alignment::Horizontal::Left,
                vertical_alignment: iced::alignment::Vertical::Top,
                line_height: text::LineHeight::default(),
                shaping: text::Shaping::Basic,
                wrapping: text::Wrapping::None,
            },
            candidate.text_origin,
            label_color,
            Rectangle::new(candidate.text_origin, candidate.text_size),
        );
    }
}

fn select_grid_label_indices(candidates: &[GridCandidate<'_>]) -> Vec<usize> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let max_tier = candidates
        .iter()
        .map(|c| c.line.importance)
        .max()
        .unwrap_or(0);

    let mut accepted = Vec::new();

    for tier in 0..=max_tier {
        for (index, candidate) in candidates.iter().enumerate() {
            if candidate.line.importance != tier {
                continue;
            }

            if !accepted
                .iter()
                .any(|&other| grid_candidates_overlap(candidate, &candidates[other]))
            {
                accepted.push(index);
            }
        }
    }

    if accepted.is_empty()
        && let Some((index, _)) = candidates
            .iter()
            .enumerate()
            .min_by_key(|(_, c)| (c.line.importance, c.screen_x as i32))
    {
        accepted.push(index);
    }

    accepted.sort_unstable_by(|a, b| {
        candidates[*a]
            .screen_x
            .partial_cmp(&candidates[*b].screen_x)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    accepted
}

fn grid_candidates_overlap(a: &GridCandidate<'_>, b: &GridCandidate<'_>) -> bool {
    if (a.clamped_x - b.clamped_x).abs() < GRID_MIN_LABEL_SPACING {
        return true;
    }

    let a_left = a.text_origin.x;
    let a_right = a_left + a.text_size.width;
    let b_left = b.text_origin.x;
    let b_right = b_left + b.text_size.width;

    a_left <= b_right + GRID_LABEL_GAP_PADDING && b_left <= a_right + GRID_LABEL_GAP_PADDING
}

fn format_frequency_label(freq: f32) -> String {
    if freq >= 10_000.0 {
        format!("{:.0} kHz", freq / 1_000.0)
    } else if freq >= 1_000.0 {
        format!("{:.1} kHz", freq / 1_000.0)
    } else if freq >= 100.0 {
        format!("{:.0} Hz", freq)
    } else if freq >= 10.0 {
        format!("{:.1} Hz", freq)
    } else {
        format!("{:.2} Hz", freq)
    }
}

fn draw_peak_label(
    renderer: &mut iced::Renderer,
    theme: &iced::Theme,
    bounds: Rectangle,
    label: &PeakLabel,
) {
    if label.opacity <= PEAK_LABEL_MIN_OPACITY
        || bounds.width <= LABEL_PADDING * 2.0
        || bounds.height <= LABEL_PADDING * 2.0
    {
        return;
    }

    let text_size = RenderParagraph::with_text(text::Text {
        content: label.text.as_str(),
        bounds: Size::INFINITY,
        size: iced::Pixels(LABEL_TEXT_SIZE),
        font: iced::Font::default(),
        horizontal_alignment: iced::alignment::Horizontal::Left,
        vertical_alignment: iced::alignment::Vertical::Top,
        line_height: text::LineHeight::default(),
        shaping: text::Shaping::Basic,
        wrapping: text::Wrapping::None,
    })
    .min_bounds();

    if text_size.width <= 0.0 || text_size.height <= 0.0 {
        return;
    }

    let anchor_x = bounds.x + bounds.width * label.normalized_x.clamp(0.0, 1.0);
    let anchor_y = bounds.y + bounds.height * (1.0 - label.normalized_y.clamp(0.0, 1.0));

    let min_x = bounds.x + LABEL_PADDING;
    let max_x = (bounds.x + bounds.width - LABEL_PADDING - text_size.width).max(min_x);
    let text_x = (anchor_x - text_size.width * 0.5).clamp(min_x, max_x);

    let min_y = bounds.y + LABEL_PADDING;
    let max_y = (bounds.y + bounds.height - LABEL_PADDING - text_size.height).max(min_y);
    let text_y = (anchor_y - LABEL_OFFSET - text_size.height).clamp(min_y, max_y);

    let text_origin = Point::new(text_x, text_y);
    let background_bounds = Rectangle::new(
        Point::new(text_x - LABEL_PADDING, text_y - LABEL_PADDING),
        Size::new(
            text_size.width + LABEL_PADDING * 2.0,
            text_size.height + LABEL_PADDING * 2.0,
        ),
    );

    let mut border = theme::sharp_border();
    border.color = theme::with_alpha(border.color, label.opacity);

    let palette = theme.extended_palette();

    renderer.fill_quad(
        Quad {
            bounds: background_bounds,
            border,
            shadow: Default::default(),
        },
        Background::Color(theme::with_alpha(
            palette.background.strong.color,
            label.opacity,
        )),
    );

    renderer.fill_text(
        text::Text {
            content: label.text.clone(),
            bounds: text_size,
            size: iced::Pixels(LABEL_TEXT_SIZE),
            font: iced::Font::default(),
            horizontal_alignment: iced::alignment::Horizontal::Left,
            vertical_alignment: iced::alignment::Vertical::Top,
            line_height: text::LineHeight::default(),
            shaping: text::Shaping::Basic,
            wrapping: text::Wrapping::None,
        },
        text_origin,
        theme::with_alpha(palette.background.base.text, label.opacity),
        Rectangle::new(text_origin, text_size),
    );
}
