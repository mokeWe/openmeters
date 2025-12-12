use crate::audio::meter_tap::MeterFormat;
use crate::dsp::spectrogram::FrequencyScale;
use crate::dsp::spectrum::{
    SpectrumConfig, SpectrumProcessor as CoreSpectrumProcessor, SpectrumSnapshot,
};
use crate::dsp::{AudioBlock, AudioProcessor, ProcessorUpdate, Reconfigurable};
use crate::ui::render::spectrum::{SpectrumParams, SpectrumPrimitive};
use crate::ui::theme;
use crate::util::audio::musical::MusicalNote;
use crate::util::audio::{hz_to_mel, lerp, mel_to_hz};
use iced::advanced::graphics::text::Paragraph as RenderParagraph;
use iced::advanced::renderer::{self, Quad};
use iced::advanced::text::{self, Paragraph as _, Renderer as _};
use iced::advanced::widget::{Tree, tree};
use iced::advanced::{Layout, Renderer as _, Widget, layout, mouse};
use iced::{Background, Color, Element, Length, Point, Rectangle, Size};
use iced_wgpu::primitive::Renderer as _;
use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const EPSILON: f32 = 1e-6;
const GRID_FREQS: &[(f32, u8)] = &[
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

static NEXT_ID: AtomicUsize = AtomicUsize::new(1);

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
        Self {
            inner: CoreSpectrumProcessor::new(SpectrumConfig {
                sample_rate,
                ..Default::default()
            }),
            channels: 2,
        }
    }

    pub fn ingest(&mut self, samples: &[f32], format: MeterFormat) -> Option<SpectrumSnapshot> {
        if samples.is_empty() {
            return None;
        }
        self.channels = format.channels.max(1);
        let sr = format.sample_rate.max(1.0);
        let mut cfg = self.inner.config();
        if (cfg.sample_rate - sr).abs() > f32::EPSILON {
            cfg.sample_rate = sr;
            self.inner.update_config(cfg);
        }
        match self
            .inner
            .process_block(&AudioBlock::new(samples, self.channels, sr, Instant::now()))
        {
            ProcessorUpdate::Snapshot(s) => Some(s),
            ProcessorUpdate::None => None,
        }
    }

    pub fn update_config(&mut self, c: SpectrumConfig) {
        self.inner.update_config(c);
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
            min_db: -120.0,
            max_db: 0.0,
            min_frequency: 10.0,
            max_frequency: 20_000.0,
            resolution: 1024,
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

#[derive(Debug, Clone)]
pub struct SpectrumState {
    style: SpectrumStyle,
    weighted: Arc<Vec<[f32; 2]>>,
    unweighted: Arc<Vec<[f32; 2]>>,
    instance_key: usize,
    peak: Option<(String, f32, f32, f32)>, // text, x, y, opacity
    grid: Arc<Vec<(f32, String, u8)>>,     // pos, label, importance
    scratch: Vec<f32>,
}

impl SpectrumState {
    pub fn new() -> Self {
        Self {
            style: SpectrumStyle::default(),
            weighted: Arc::new(Vec::new()),
            unweighted: Arc::new(Vec::new()),
            instance_key: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            peak: None,
            grid: Arc::new(Vec::new()),
            scratch: Vec::new(),
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
            self.grid = Arc::new(Vec::new());
        }
    }

    pub fn update_show_peak_label(&mut self, show: bool) {
        self.style.show_peak_label = show;
        if !show {
            self.peak = None;
        }
    }

    pub fn set_palette(&mut self, palette: &[Color]) {
        if palette.len() == 5
            && !self
                .style
                .spectrum_palette
                .iter()
                .zip(palette)
                .all(|(a, b)| theme::colors_equal(*a, *b))
        {
            self.style.spectrum_palette.copy_from_slice(palette);
        }
    }

    pub fn palette(&self) -> [Color; 5] {
        self.style.spectrum_palette
    }

    pub fn apply_snapshot(&mut self, snap: &SpectrumSnapshot) {
        if snap.frequency_bins.is_empty()
            || snap.magnitudes_db.is_empty()
            || snap.frequency_bins.len() != snap.magnitudes_db.len()
        {
            self.fade_peak(None);
            return;
        }
        let nyq = snap
            .frequency_bins
            .last()
            .copied()
            .unwrap_or(self.style.max_frequency);
        let (min_f, mut max_f) = (
            self.style.min_frequency.max(EPSILON),
            self.style.max_frequency.min(nyq),
        );
        if max_f <= min_f {
            max_f = nyq.max(min_f * 1.02);
        }
        if max_f <= min_f {
            self.fade_peak(None);
            return;
        }

        let scale = Scale::new(min_f, max_f);
        let res = self.style.resolution.max(32);
        let w = Arc::make_mut(&mut self.weighted);
        let u = Arc::make_mut(&mut self.unweighted);
        build_points(&self.style, w, u, res, &scale, snap);

        if self.style.smoothing_radius > 0 && self.style.smoothing_passes > 0 {
            smooth(
                w,
                self.style.smoothing_radius,
                self.style.smoothing_passes,
                &mut self.scratch,
            );
            smooth(
                u,
                self.style.smoothing_radius,
                self.style.smoothing_passes,
                &mut self.scratch,
            );
        }
        if self.style.reverse_frequency {
            w.reverse();
            u.reverse();
            reindex(w);
            reindex(u);
        }

        self.grid = if self.style.show_grid {
            let mut v = Vec::new();
            for &(f, imp) in GRID_FREQS {
                if f < min_f || f > max_f {
                    continue;
                }
                let mut p = scale.pos_of(self.style.frequency_scale, f);
                if self.style.reverse_frequency {
                    p = 1.0 - p;
                }
                if p.is_finite() {
                    v.push((p.clamp(0.0, 1.0), fmt_freq(f), imp));
                }
            }
            Arc::new(v)
        } else {
            Arc::new(Vec::new())
        };

        let pk = self
            .style
            .show_peak_label
            .then(|| self.build_peak(snap, &scale))
            .flatten();
        self.fade_peak(pk);
    }

    fn build_peak(&self, snap: &SpectrumSnapshot, sc: &Scale) -> Option<(String, f32, f32)> {
        let f = snap
            .peak_frequency_hz
            .filter(|&f| f.is_finite() && f > 0.0)?;
        let mut x = sc.pos_of(self.style.frequency_scale, f);
        if self.style.reverse_frequency {
            x = 1.0 - x;
        }
        let m = interp(&snap.frequency_bins, &snap.magnitudes_db, f);
        let y = ((m - self.style.min_db) / (self.style.max_db - self.style.min_db).max(EPSILON))
            .clamp(0.0, 1.0);
        if y < 0.08 {
            return None;
        }
        let txt = MusicalNote::from_frequency(f).map_or_else(
            || format!("{:.1} Hz", f),
            |n| format!("{:.1} Hz | {}", f, n.format()),
        );
        Some((txt, x.clamp(0.0, 1.0), y))
    }

    fn fade_peak(&mut self, data: Option<(String, f32, f32)>) {
        match (data, &mut self.peak) {
            (Some((t, x, y)), Some(p)) => {
                *p = (t, x, y, (p.3 + (1.0 - p.3) * 0.35).min(1.0));
            }
            (Some((t, x, y)), None) => {
                self.peak = Some((t, x, y, 0.0));
            }
            (None, Some(p)) => {
                p.3 += (0.0 - p.3) * 0.12;
                if p.3 < 0.01 {
                    self.peak = None;
                }
            }
            (None, None) => {}
        }
    }

    fn visual(&self, bounds: Rectangle, theme: &iced::Theme) -> Option<SpectrumVisual> {
        if self.weighted.len() < 2 {
            return None;
        }
        let pal = theme.extended_palette();
        Some(SpectrumVisual {
            params: SpectrumParams {
                bounds,
                normalized_points: Arc::clone(&self.weighted),
                secondary_points: Arc::clone(&self.unweighted),
                instance_key: self.instance_key,
                line_color: theme::color_to_rgba(theme::mix_colors(
                    pal.primary.base.color,
                    pal.background.base.text,
                    0.35,
                )),
                line_width: self.style.line_thickness,
                secondary_line_color: theme::color_to_rgba(theme::with_alpha(
                    pal.secondary.weak.text,
                    0.3,
                )),
                secondary_line_width: self.style.unweighted_line_thickness,
                highlight_threshold: self.style.highlight_threshold,
                spectrum_palette: self
                    .style
                    .spectrum_palette
                    .iter()
                    .map(|&c| theme::color_to_rgba(c))
                    .collect(),
            },
            peak: self
                .style
                .show_peak_label
                .then(|| self.peak.clone())
                .flatten(),
            grid: Arc::clone(&self.grid),
        })
    }
}

#[derive(Debug)]
struct SpectrumVisual {
    params: SpectrumParams,
    peak: Option<(String, f32, f32, f32)>,
    grid: Arc<Vec<(f32, String, u8)>>,
}

#[derive(Debug)]
pub struct Spectrum<'a>(&'a RefCell<SpectrumState>);
impl<'a> Spectrum<'a> {
    pub fn new(state: &'a RefCell<SpectrumState>) -> Self {
        Self(state)
    }
}

impl<'a, M> Widget<M, iced::Theme, iced::Renderer> for Spectrum<'a> {
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<()>()
    }
    fn state(&self) -> tree::State {
        tree::State::None
    }
    fn size(&self) -> Size<Length> {
        Size::new(Length::Fill, Length::Fill)
    }
    fn children(&self) -> Vec<Tree> {
        Vec::new()
    }
    fn diff(&self, _: &mut Tree) {}
    fn layout(&mut self, _: &mut Tree, _: &iced::Renderer, lim: &layout::Limits) -> layout::Node {
        layout::Node::new(lim.resolve(Length::Fill, Length::Fill, Size::ZERO))
    }
    fn draw(
        &self,
        _: &Tree,
        r: &mut iced::Renderer,
        th: &iced::Theme,
        _: &renderer::Style,
        lay: Layout<'_>,
        _: mouse::Cursor,
        _: &Rectangle,
    ) {
        let b = lay.bounds();
        let Some(v) = self.0.borrow().visual(b, th) else {
            r.fill_quad(
                Quad {
                    bounds: b,
                    border: Default::default(),
                    shadow: Default::default(),
                    snap: true,
                },
                Background::Color(th.extended_palette().background.base.color),
            );
            return;
        };
        if !v.grid.is_empty() {
            r.with_layer(b, |r| draw_grid(r, th, b, &v.grid));
        }
        r.draw_primitive(b, SpectrumPrimitive::new(v.params));
        if let Some(pk) = &v.peak {
            r.with_layer(b, |r| draw_peak(r, th, b, pk));
        }
    }
}

pub fn widget<'a, M: 'a>(state: &'a RefCell<SpectrumState>) -> Element<'a, M> {
    Element::new(Spectrum::new(state))
}

// --- Helpers ---

#[derive(Clone, Copy)]
struct Scale {
    min: f32,
    max: f32,
    log_min: f32,
    log_range: f32,
    mel_min: f32,
    mel_range: f32,
}

impl Scale {
    fn new(min: f32, max: f32) -> Self {
        let log_min = min.max(EPSILON).log10();
        let log_range = (max.max(min * 1.01).log10() - log_min).max(EPSILON);
        let mel_min = hz_to_mel(min);
        Self {
            min,
            max,
            log_min,
            log_range,
            mel_min,
            mel_range: (hz_to_mel(max) - mel_min).max(EPSILON),
        }
    }
    fn freq_at(&self, s: FrequencyScale, t: f32) -> f32 {
        match s {
            FrequencyScale::Linear => self.min + (self.max - self.min) * t,
            FrequencyScale::Logarithmic => 10f32.powf(self.log_min + self.log_range * t),
            FrequencyScale::Mel => mel_to_hz(self.mel_min + self.mel_range * t),
        }
    }
    fn pos_of(&self, s: FrequencyScale, f: f32) -> f32 {
        let f = f.clamp(self.min, self.max);
        match s {
            FrequencyScale::Linear => (f - self.min) / (self.max - self.min).max(EPSILON),
            FrequencyScale::Logarithmic => (f.max(EPSILON).log10() - self.log_min) / self.log_range,
            FrequencyScale::Mel => (hz_to_mel(f) - self.mel_min) / self.mel_range,
        }
        .clamp(0.0, 1.0)
    }
}

fn interp(bins: &[f32], mags: &[f32], t: f32) -> f32 {
    if bins.is_empty() || t <= bins[0] {
        return mags.first().copied().unwrap_or(0.0);
    }
    if t >= *bins.last().unwrap() {
        return mags.last().copied().unwrap_or(0.0);
    }
    match bins.binary_search_by(|p| p.partial_cmp(&t).unwrap_or(std::cmp::Ordering::Less)) {
        Ok(i) => mags.get(i).copied().unwrap_or(0.0),
        Err(i) => {
            let (lo, hi) = (i.saturating_sub(1), i.min(bins.len() - 1));
            lerp(
                mags.get(lo).copied().unwrap_or(0.0),
                mags.get(hi).copied().unwrap_or(0.0),
                (t - bins[lo]) / (bins[hi] - bins[lo]).max(EPSILON),
            )
        }
    }
}

fn build_points(
    style: &SpectrumStyle,
    w: &mut Vec<[f32; 2]>,
    u: &mut Vec<[f32; 2]>,
    res: usize,
    sc: &Scale,
    snap: &SpectrumSnapshot,
) {
    w.clear();
    u.clear();
    w.reserve(res);
    u.reserve(res);
    let dr = (style.max_db - style.min_db).max(EPSILON);
    for i in 0..res {
        let t = if res > 1 {
            i as f32 / (res - 1) as f32
        } else {
            0.0
        };
        let f = sc.freq_at(style.frequency_scale, t);
        let mw = interp(&snap.frequency_bins, &snap.magnitudes_db, f);
        let mu = interp(&snap.frequency_bins, &snap.magnitudes_unweighted_db, f);
        w.push([t, ((mw - style.min_db) / dr).clamp(0.0, 1.0)]);
        u.push([t, ((mu - style.min_db) / dr).clamp(0.0, 1.0)]);
    }
}

fn smooth(pts: &mut [[f32; 2]], r: usize, passes: usize, scratch: &mut Vec<f32>) {
    if r == 0 || passes == 0 || pts.len() < 3 {
        return;
    }
    scratch.resize(pts.len(), 0.0);
    for _ in 0..passes {
        for (d, p) in scratch.iter_mut().zip(pts.iter()) {
            *d = p[1];
        }
        for (i, p) in pts.iter_mut().enumerate() {
            let (s, e) = (i.saturating_sub(r), (i + r + 1).min(scratch.len()));
            let (mut sum, mut wsum) = (0.0f32, 0.0f32);
            for (j, &v) in scratch[s..e].iter().enumerate() {
                let w = (r - (s + j).abs_diff(i) + 1) as f32;
                sum += v * w;
                wsum += w;
            }
            p[1] = sum / wsum;
        }
    }
}

fn reindex(pts: &mut [[f32; 2]]) {
    let n = pts.len();
    for (i, p) in pts.iter_mut().enumerate() {
        p[0] = if n > 1 {
            i as f32 / (n - 1) as f32
        } else {
            0.0
        };
    }
}

fn fmt_freq(f: f32) -> String {
    match f {
        f if f >= 10_000.0 => format!("{:.0} kHz", f / 1000.0),
        f if f >= 1_000.0 => format!("{:.1} kHz", f / 1000.0),
        f if f >= 100.0 => format!("{:.0} Hz", f),
        f if f >= 10.0 => format!("{:.1} Hz", f),
        _ => format!("{:.2} Hz", f),
    }
}

fn txt(s: &str, px: f32) -> (Size, text::Text<String>) {
    let t = text::Text {
        content: s.to_string(),
        bounds: Size::INFINITE,
        size: iced::Pixels(px),
        font: iced::Font::default(),
        align_x: iced::alignment::Horizontal::Left.into(),
        align_y: iced::alignment::Vertical::Top,
        line_height: text::LineHeight::default(),
        shaping: text::Shaping::Basic,
        wrapping: text::Wrapping::None,
    };
    let measure = text::Text {
        content: s,
        bounds: Size::INFINITE,
        size: iced::Pixels(px),
        font: iced::Font::default(),
        align_x: iced::alignment::Horizontal::Left.into(),
        align_y: iced::alignment::Vertical::Top,
        line_height: text::LineHeight::default(),
        shaping: text::Shaping::Basic,
        wrapping: text::Wrapping::None,
    };
    (RenderParagraph::with_text(measure).min_bounds(), t)
}

fn draw_grid(r: &mut iced::Renderer, th: &iced::Theme, b: Rectangle, lines: &[(f32, String, u8)]) {
    if b.width <= 0.0 || b.height <= 0.0 {
        return;
    }
    let pal = th.extended_palette();
    let (lc, tc) = (
        theme::with_alpha(pal.background.base.text, 0.25),
        theme::with_alpha(pal.background.base.text, 0.75),
    );
    let cands: Vec<_> = lines
        .iter()
        .filter_map(|(pos, lbl, imp)| {
            let x = b.x + b.width * pos;
            (x >= b.x - 1.0 && x <= b.x + b.width + 1.0)
                .then(|| {
                    let (sz, _) = txt(lbl, 10.0);
                    (sz.width > 0.0 && sz.height > 0.0).then_some((x, lbl, *imp, sz))
                })
                .flatten()
        })
        .collect();
    let mut acc = Vec::with_capacity(cands.len());
    let mut indices: Vec<usize> = (0..cands.len()).collect();
    indices.sort_by_key(|&i| cands[i].2);

    let bounds = |i| {
        let (x, _, _, sz): (f32, &String, u8, Size) = cands[i];
        let l =
            (x - sz.width * 0.5).clamp(b.x + 6.0, (b.x + b.width - 6.0 - sz.width).max(b.x + 6.0));
        (l, l + sz.width + 6.0)
    };

    for i in indices {
        let (l, r) = bounds(i);
        if !acc.iter().any(|&j| {
            let (ol, or) = bounds(j);
            l < or && r > ol
        }) {
            acc.push(i);
        }
    }
    for &i in &acc {
        let (x, lbl, _, sz) = &cands[i];
        let (ty, lt) = (b.y + 6.0, b.y + 6.0 + sz.height + 6.0);
        let lh = (b.y + b.height - lt).max(0.0);
        if lh > 0.0 {
            r.fill_quad(
                Quad {
                    bounds: Rectangle::new(
                        Point::new((x - 0.5).clamp(b.x, b.x + b.width - 1.0), lt),
                        Size::new(1.0, lh),
                    ),
                    border: Default::default(),
                    shadow: Default::default(),
                    snap: true,
                },
                Background::Color(lc),
            );
        }
        let tx =
            (x - sz.width * 0.5).clamp(b.x + 6.0, (b.x + b.width - 6.0 - sz.width).max(b.x + 6.0));
        let (_, mut t) = txt(lbl, 10.0);
        t.bounds = *sz;
        r.fill_text(
            t,
            Point::new(tx, ty),
            tc,
            Rectangle::new(Point::new(tx, ty), *sz),
        );
    }
}

fn draw_peak(r: &mut iced::Renderer, th: &iced::Theme, b: Rectangle, pk: &(String, f32, f32, f32)) {
    let (s, nx, ny, op) = pk;
    if *op < 0.01 || b.width < 8.0 || b.height < 8.0 {
        return;
    }
    let (sz, mut t) = txt(s, 12.0);
    if sz.width <= 0.0 || sz.height <= 0.0 {
        return;
    }
    let (ax, ay) = (
        b.x + b.width * nx.clamp(0.0, 1.0),
        b.y + b.height * (1.0 - ny.clamp(0.0, 1.0)),
    );
    let tx =
        (ax - sz.width * 0.5).clamp(b.x + 4.0, (b.x + b.width - 4.0 - sz.width).max(b.x + 4.0));
    let ty =
        (ay - 8.0 - sz.height).clamp(b.y + 4.0, (b.y + b.height - 4.0 - sz.height).max(b.y + 4.0));
    let bg = Rectangle::new(
        Point::new(tx - 4.0, ty - 4.0),
        Size::new(sz.width + 8.0, sz.height + 8.0),
    );
    let mut bdr = theme::sharp_border();
    bdr.color = theme::with_alpha(bdr.color, *op);
    let pal = th.extended_palette();
    r.fill_quad(
        Quad {
            bounds: bg,
            border: bdr,
            shadow: Default::default(),
            snap: true,
        },
        Background::Color(theme::with_alpha(pal.background.strong.color, *op)),
    );
    t.bounds = sz;
    r.fill_text(
        t,
        Point::new(tx, ty),
        theme::with_alpha(pal.background.base.text, *op),
        Rectangle::new(Point::new(tx, ty), sz),
    );
}
