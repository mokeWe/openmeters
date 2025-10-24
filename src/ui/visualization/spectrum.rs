use crate::audio::meter_tap::MeterFormat;
use crate::dsp::spectrogram::{FrequencyScale, hz_to_mel, mel_to_hz};
use crate::dsp::spectrum::{
    SpectrumConfig, SpectrumProcessor as CoreSpectrumProcessor, SpectrumSnapshot,
};
use crate::dsp::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::ui::render::spectrum::{SpectrumParams, SpectrumPrimitive};
use crate::ui::theme;
use iced::advanced::Renderer as _;
use iced::advanced::renderer::{self, Quad};
use iced::advanced::widget::{Tree, tree};
use iced::advanced::{Layout, Widget, layout, mouse};
use iced::{Background, Color, Element, Length, Rectangle, Size};
use iced_wgpu::primitive::Renderer as _;
use std::sync::Arc;
use std::time::Instant;

const DEFAULT_RESOLUTION: usize = 1024;
const MIN_FREQUENCY_HZ: f32 = 10.0;
const MAX_FREQUENCY_HZ: f32 = 20_000.0;
const EPSILON: f32 = 1.0e-6;

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
    pub background: Color,
    pub line_color: Color,
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
    pub frequency_scale: FrequencyScale,
    pub reverse_frequency: bool,
}

impl Default for SpectrumStyle {
    fn default() -> Self {
        let background = Color::from_rgba(0.0, 0.0, 0.0, 0.0);
        let line_color = theme::mix_colors(theme::accent_primary(), theme::text_color(), 0.35);

        Self {
            background,
            line_color,
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
            frequency_scale: FrequencyScale::Logarithmic,
            reverse_frequency: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpectrumState {
    style: SpectrumStyle,
    points: PointSeries,
}

#[derive(Debug)]
pub struct SpectrumVisual {
    pub primitive: SpectrumParams,
    pub background: Color,
}

impl SpectrumState {
    pub fn new() -> Self {
        Self {
            style: SpectrumStyle::default(),
            points: PointSeries::new(),
        }
    }

    pub fn style_mut(&mut self) -> &mut SpectrumStyle {
        &mut self.style
    }

    pub fn apply_snapshot(&mut self, snapshot: &SpectrumSnapshot) {
        if snapshot.frequency_bins.is_empty()
            || snapshot.magnitudes_db.is_empty()
            || snapshot.frequency_bins.len() != snapshot.magnitudes_db.len()
        {
            return;
        }

        let Some((scale, resolution)) = domain_parameters(&self.style, snapshot) else {
            return;
        };

        self.points
            .update(&self.style, &scale, snapshot, resolution);
    }

    pub fn visual(&self, bounds: Rectangle) -> Option<SpectrumVisual> {
        if !self.points.is_ready() {
            return None;
        }

        let normalized_points = self.points.weighted();
        let unweighted_points = self.points.unweighted();
        let instance_key = points_key(&normalized_points, &unweighted_points);

        let primitive = SpectrumParams {
            bounds,
            normalized_points,
            secondary_points: unweighted_points,
            instance_key,
            line_color: theme::color_to_rgba(self.style.line_color),
            line_width: self.style.line_thickness,
            secondary_line_color: theme::color_to_rgba(self.style.unweighted_line_color),
            secondary_line_width: self.style.unweighted_line_thickness,
            highlight_threshold: self.style.highlight_threshold,
            highlight_color: theme::color_to_rgba(self.style.highlight_color),
        };

        Some(SpectrumVisual {
            primitive,
            background: self.style.background,
        })
    }
}

fn points_key(primary: &Arc<Vec<[f32; 2]>>, secondary: &Arc<Vec<[f32; 2]>>) -> usize {
    let primary_ptr = Arc::as_ptr(primary) as usize;
    let secondary_ptr = Arc::as_ptr(secondary) as usize;
    primary_ptr ^ secondary_ptr.wrapping_mul(31)
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
        _theme: &iced::Theme,
        _style: &renderer::Style,
        layout: Layout<'_>,
        _cursor: mouse::Cursor,
        _viewport: &Rectangle,
    ) {
        let bounds = layout.bounds();

        let Some(visual) = self.state.visual(bounds) else {
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

        renderer.draw_primitive(bounds, SpectrumPrimitive::new(visual.primitive));
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

#[derive(Debug, Clone)]
struct PointSeries {
    weighted: Arc<Vec<[f32; 2]>>,
    unweighted: Arc<Vec<[f32; 2]>>,
    smoothing_scratch: Vec<f32>,
}

impl PointSeries {
    fn new() -> Self {
        Self {
            weighted: Arc::new(Vec::new()),
            unweighted: Arc::new(Vec::new()),
            smoothing_scratch: Vec::new(),
        }
    }

    fn weighted(&self) -> Arc<Vec<[f32; 2]>> {
        Arc::clone(&self.weighted)
    }

    fn unweighted(&self) -> Arc<Vec<[f32; 2]>> {
        Arc::clone(&self.unweighted)
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
    if (max_db - min_db).abs() <= EPSILON {
        return 0.0;
    }
    ((value - min_db) / (max_db - min_db)).clamp(0.0, 1.0)
}

fn interpolate_magnitude(bins: &[f32], magnitudes: &[f32], target: f32) -> f32 {
    use std::cmp::Ordering;

    if bins.is_empty() || magnitudes.is_empty() {
        return 0.0;
    }

    if target <= bins[0] {
        return magnitudes[0];
    }

    if let Some(&last_bin) = bins.last()
        && target >= last_bin
    {
        return magnitudes.last().copied().unwrap_or(0.0);
    }

    match bins.binary_search_by(|probe| probe.partial_cmp(&target).unwrap_or(Ordering::Less)) {
        Ok(index) => magnitudes.get(index).copied().unwrap_or(0.0),
        Err(index) => {
            let upper = index.min(bins.len() - 1);
            let lower = upper.saturating_sub(1);
            let span = (bins[upper] - bins[lower]).max(EPSILON);
            let t = (target - bins[lower]) / span;
            let lower_mag = magnitudes.get(lower).copied().unwrap_or(0.0);
            let upper_mag = magnitudes.get(upper).copied().unwrap_or(lower_mag);
            lower_mag + (upper_mag - lower_mag) * t
        }
    }
}

fn smooth_points(points: &mut [[f32; 2]], radius: usize, passes: usize, scratch: &mut Vec<f32>) {
    if radius == 0 || passes == 0 || points.len() < 3 {
        return;
    }

    let len = points.len();
    if scratch.len() != len {
        scratch.clear();
        scratch.resize(len, 0.0);
    }

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
