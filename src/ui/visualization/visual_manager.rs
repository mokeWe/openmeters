//! Central owner of visual modules and their state.

use crate::audio::meter_tap::{self, MeterFormat};
use crate::dsp::ProcessorUpdate;
use crate::dsp::oscilloscope::OscilloscopeConfig;
use crate::dsp::waveform::{MAX_COLUMN_CAPACITY, MIN_COLUMN_CAPACITY};
use crate::ui::settings::{
    LoudnessSettings, ModuleSettings, OscilloscopeSettings, PaletteSettings, SpectrogramSettings,
    SpectrumSettings, StereometerSettings, VisualSettings, WaveformSettings,
};
use crate::ui::theme;
use crate::ui::visualization::loudness::{self, LoudnessMeterProcessor, LoudnessMeterState};
use crate::ui::visualization::oscilloscope::{self, OscilloscopeProcessor, OscilloscopeState};
use crate::ui::visualization::spectrogram::{self, SpectrogramProcessor, SpectrogramState};
use crate::ui::visualization::spectrum::{self, SpectrumProcessor, SpectrumState};
use crate::ui::visualization::stereometer::{self, StereometerProcessor, StereometerState};
use crate::ui::visualization::waveform::{
    self, WaveformProcessor as WaveformUiProcessor, WaveformState,
};
use crate::util::audio::DEFAULT_SAMPLE_RATE;
use iced::alignment::{Horizontal, Vertical};
use iced::widget::container;
use iced::{Element, Length};
use serde::{Deserialize, Serialize};
use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;
/// Identifiers for visual kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisualKind {
    Loudness,
    Oscilloscope,
    Spectrogram,
    Spectrum,
    Stereometer,
    Waveform,
}

impl VisualKind {
    pub const LOUDNESS: Self = Self::Loudness;
    pub const OSCILLOSCOPE: Self = Self::Oscilloscope;
    pub const SPECTROGRAM: Self = Self::Spectrogram;
    pub const SPECTRUM: Self = Self::Spectrum;
    pub const STEREOMETER: Self = Self::Stereometer;
    pub const WAVEFORM: Self = Self::Waveform;
}

/// Unique identifier for instantiated visual slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VisualId(u32);

impl VisualId {
    fn next(counter: &mut u32) -> Self {
        let current = *counter;
        *counter = counter.saturating_add(1);
        VisualId(current)
    }
}

/// Static metadata describing a visual module's preferred presentation.
#[derive(Debug, Clone, Copy)]
pub struct VisualMetadata {
    pub display_name: &'static str,
    pub preferred_width: f32,
    pub preferred_height: f32,
    pub fill_horizontal: bool,
    pub fill_vertical: bool,
    pub min_width: f32,
    pub max_width: f32,
}

impl VisualMetadata {
    const DEFAULT: Self = Self {
        display_name: "",
        preferred_width: 200.0,
        preferred_height: 200.0,
        fill_horizontal: true,
        fill_vertical: true,
        min_width: 100.0,
        max_width: f32::INFINITY,
    };
}

/// Aggregate snapshot of all visual slots.
#[derive(Debug, Clone)]
pub struct VisualSnapshot {
    pub slots: Vec<VisualSlotSnapshot>,
}

/// Snapshot representing a single visual slot.
#[derive(Debug, Clone)]
pub struct VisualSlotSnapshot {
    pub id: VisualId,
    pub kind: VisualKind,
    pub enabled: bool,
    pub metadata: VisualMetadata,
    pub content: VisualContent,
}

impl From<&VisualEntry> for VisualSlotSnapshot {
    fn from(entry: &VisualEntry) -> Self {
        Self {
            id: entry.id,
            kind: entry.kind,
            enabled: entry.enabled,
            metadata: entry.metadata,
            content: entry.cached_content.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum VisualContent {
    LoudnessMeter {
        state: LoudnessMeterState,
    },
    Oscilloscope {
        state: Rc<RefCell<OscilloscopeState>>,
    },
    Spectrogram {
        state: Rc<RefCell<SpectrogramState>>,
    },
    Spectrum {
        state: Rc<RefCell<SpectrumState>>,
    },
    Stereometer {
        state: Rc<RefCell<StereometerState>>,
    },
    Waveform {
        state: Rc<RefCell<WaveformState>>,
    },
}

impl VisualContent {
    /// Renders this visual content into an Element, framed according to metadata.
    pub fn render<M: 'static>(&self, metadata: VisualMetadata) -> Element<'_, M> {
        let inner: Element<'_, M> = match self {
            Self::LoudnessMeter { state } => container(loudness::widget_with_layout(
                state,
                metadata.preferred_width,
                metadata.preferred_height,
            ))
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(Horizontal::Center)
            .align_y(Vertical::Bottom)
            .into(),
            Self::Oscilloscope { state } => container(oscilloscope::widget(state))
                .width(Length::Fill)
                .height(Length::Fill)
                .center_x(Length::Fill)
                .into(),
            Self::Spectrogram { state } => container(spectrogram::widget(state))
                .width(Length::Fill)
                .height(Length::Fill)
                .center_x(Length::Fill)
                .into(),
            Self::Spectrum { state } => container(spectrum::widget(state))
                .width(Length::Fill)
                .height(Length::Fill)
                .center_x(Length::Fill)
                .into(),
            Self::Stereometer { state } => container(stereometer::widget(state))
                .width(Length::Fill)
                .height(Length::Fill)
                .center_x(Length::Fill)
                .into(),
            Self::Waveform { state } => container(waveform::widget(state))
                .width(Length::Fill)
                .height(Length::Fill)
                .center_x(Length::Fill)
                .into(),
        };

        let (w, h) = (
            if metadata.fill_horizontal {
                Length::Fill
            } else {
                Length::Fixed(metadata.preferred_width)
            },
            if metadata.fill_vertical {
                Length::Fill
            } else {
                Length::Fixed(metadata.preferred_height)
            },
        );

        container(
            container(inner)
                .width(w)
                .height(h)
                .align_x(Horizontal::Center)
                .align_y(Vertical::Center),
        )
        .width(Length::Fill)
        .height(Length::Fill)
        .align_x(Horizontal::Center)
        .align_y(Vertical::Center)
        .into()
    }
}

trait VisualModule {
    fn ingest(&mut self, samples: &[f32], format: MeterFormat);
    fn content(&self) -> VisualContent;
    fn apply_settings(&mut self, settings: &ModuleSettings);
    fn export_settings(&self) -> Option<ModuleSettings> {
        None
    }
}

struct VisualDescriptor {
    kind: VisualKind,
    metadata: VisualMetadata,
    build: fn() -> Box<dyn VisualModule>,
}

struct VisualEntry {
    id: VisualId,
    kind: VisualKind,
    enabled: bool,
    metadata: VisualMetadata,
    module: Box<dyn VisualModule>,
    cached_content: VisualContent,
}

impl VisualEntry {
    fn new(id: VisualId, descriptor: &VisualDescriptor) -> Self {
        let module = (descriptor.build)();
        let cached_content = module.content();
        Self {
            id,
            kind: descriptor.kind,
            enabled: false,
            metadata: descriptor.metadata,
            module,
            cached_content,
        }
    }

    fn apply_settings(&mut self, settings: &ModuleSettings) {
        if let Some(enabled) = settings.enabled {
            self.enabled = enabled;
        }
        self.module.apply_settings(settings);
        self.cached_content = self.module.content();
    }
}

const VISUAL_DESCRIPTORS: &[VisualDescriptor] = &[
    VisualDescriptor {
        kind: VisualKind::Loudness,
        metadata: VisualMetadata {
            display_name: "Loudness Meter",
            preferred_width: 140.0,
            preferred_height: 300.0,
            min_width: 80.0,
            max_width: 140.0,
            ..VisualMetadata::DEFAULT
        },
        build: build_module::<LoudnessVisual>,
    },
    VisualDescriptor {
        kind: VisualKind::Oscilloscope,
        metadata: VisualMetadata {
            display_name: "Oscilloscope",
            preferred_width: 150.0,
            preferred_height: 160.0,
            min_width: 100.0,
            ..VisualMetadata::DEFAULT
        },
        build: build_module::<OscilloscopeVisual>,
    },
    VisualDescriptor {
        kind: VisualKind::Waveform,
        metadata: VisualMetadata {
            display_name: "Waveform",
            preferred_width: 220.0,
            preferred_height: 180.0,
            min_width: 220.0,
            ..VisualMetadata::DEFAULT
        },
        build: build_module::<WaveformVisual>,
    },
    VisualDescriptor {
        kind: VisualKind::Spectrogram,
        metadata: VisualMetadata {
            display_name: "Spectrogram",
            preferred_width: 320.0,
            preferred_height: 220.0,
            min_width: 300.0,
            ..VisualMetadata::DEFAULT
        },
        build: build_module::<SpectrogramVisual>,
    },
    VisualDescriptor {
        kind: VisualKind::Spectrum,
        metadata: VisualMetadata {
            display_name: "Spectrum analyzer",
            preferred_width: 400.0,
            preferred_height: 180.0,
            min_width: 400.0,
            ..VisualMetadata::DEFAULT
        },
        build: build_module::<SpectrumVisual>,
    },
    VisualDescriptor {
        kind: VisualKind::Stereometer,
        metadata: VisualMetadata {
            display_name: "Stereometer",
            preferred_width: 150.0,
            preferred_height: 220.0,
            min_width: 100.0,
            ..VisualMetadata::DEFAULT
        },
        build: build_module::<StereometerVisual>,
    },
];

fn build_module<M>() -> Box<dyn VisualModule>
where
    M: VisualModule + Default + 'static,
{
    Box::new(M::default())
}

fn resolve_palette<const N: usize>(
    palette: &Option<PaletteSettings>,
    default: &[iced::Color; N],
) -> [iced::Color; N] {
    palette
        .as_ref()
        .and_then(|p| p.to_array::<N>())
        .unwrap_or(*default)
}

fn rc_cell<T>(value: T) -> Rc<RefCell<T>> {
    Rc::new(RefCell::new(value))
}

struct LoudnessVisual {
    processor: LoudnessMeterProcessor,
    state: LoudnessMeterState,
}

impl Default for LoudnessVisual {
    fn default() -> Self {
        Self {
            processor: LoudnessMeterProcessor::new(DEFAULT_SAMPLE_RATE),
            state: LoudnessMeterState::new(),
        }
    }
}

impl VisualModule for LoudnessVisual {
    fn ingest(&mut self, samples: &[f32], format: MeterFormat) {
        let snapshot = self.processor.ingest(samples, format);
        self.state.apply_snapshot(&snapshot);
    }

    fn content(&self) -> VisualContent {
        VisualContent::LoudnessMeter {
            state: self.state.clone(),
        }
    }

    fn apply_settings(&mut self, settings: &ModuleSettings) {
        if let Some(stored) = settings.config::<LoudnessSettings>() {
            self.state.set_modes(stored.left_mode, stored.right_mode);
            self.state.set_palette(&resolve_palette(
                &stored.palette,
                &theme::DEFAULT_LOUDNESS_PALETTE,
            ));
        }
    }

    fn export_settings(&self) -> Option<ModuleSettings> {
        let mut settings = LoudnessSettings::new(self.state.left_mode(), self.state.right_mode());
        settings.palette = PaletteSettings::maybe_from_colors(
            self.state.palette(),
            &theme::DEFAULT_LOUDNESS_PALETTE,
        );
        Some(ModuleSettings::with_config(&settings))
    }
}

struct OscilloscopeVisual {
    processor: OscilloscopeProcessor,
    state: Rc<RefCell<OscilloscopeState>>,
}

impl Default for OscilloscopeVisual {
    fn default() -> Self {
        Self {
            processor: OscilloscopeProcessor::new(OscilloscopeConfig {
                sample_rate: DEFAULT_SAMPLE_RATE,
                ..Default::default()
            }),
            state: rc_cell(OscilloscopeState::new()),
        }
    }
}

impl VisualModule for OscilloscopeVisual {
    fn ingest(&mut self, samples: &[f32], format: MeterFormat) {
        if let Some(snapshot) = self.processor.ingest(samples, format) {
            self.state.borrow_mut().apply_snapshot(&snapshot);
        }
    }

    fn content(&self) -> VisualContent {
        VisualContent::Oscilloscope {
            state: self.state.clone(),
        }
    }

    fn apply_settings(&mut self, settings: &ModuleSettings) {
        if let Some(stored) = settings.config::<OscilloscopeSettings>() {
            let mut config = self.processor.config();
            config.segment_duration = stored.segment_duration;
            config.trigger_mode = stored.trigger_mode;
            self.processor.update_config(config);

            let mut state = self.state.borrow_mut();
            state.update_view_settings(stored.persistence, stored.channel_mode);
            state.set_palette(&resolve_palette(
                &stored.palette,
                &theme::DEFAULT_OSCILLOSCOPE_PALETTE,
            ));
        }
    }

    fn export_settings(&self) -> Option<ModuleSettings> {
        let config = self.processor.config();
        let state = self.state.borrow();

        let settings = OscilloscopeSettings {
            segment_duration: config.segment_duration,
            trigger_mode: config.trigger_mode,
            persistence: state.persistence(),
            channel_mode: state.channel_mode(),
            palette: PaletteSettings::maybe_from_colors(
                state.palette(),
                &theme::DEFAULT_OSCILLOSCOPE_PALETTE,
            ),
        };

        Some(ModuleSettings::with_config(&settings))
    }
}

struct WaveformVisual {
    processor: WaveformUiProcessor,
    state: Rc<RefCell<WaveformState>>,
}

impl Default for WaveformVisual {
    fn default() -> Self {
        Self {
            processor: WaveformUiProcessor::new(DEFAULT_SAMPLE_RATE),
            state: rc_cell(WaveformState::new()),
        }
    }
}

impl VisualModule for WaveformVisual {
    fn ingest(&mut self, samples: &[f32], format: MeterFormat) {
        self.ensure_capacity();
        if let Some(snapshot) = self.processor.ingest(samples, format) {
            self.state.borrow_mut().apply_snapshot(snapshot);
        }
    }

    fn content(&self) -> VisualContent {
        VisualContent::Waveform {
            state: self.state.clone(),
        }
    }

    fn apply_settings(&mut self, settings: &ModuleSettings) {
        if let Some(stored) = settings.config::<WaveformSettings>() {
            let mut config = self.processor.config();
            stored.apply_to(&mut config);
            self.processor.update_config(config);
            self.ensure_capacity();
            self.state.borrow_mut().set_palette(&resolve_palette(
                &stored.palette,
                &theme::DEFAULT_WAVEFORM_PALETTE,
            ));
        }
    }

    fn export_settings(&self) -> Option<ModuleSettings> {
        let mut snapshot = WaveformSettings::from_config(&self.processor.config());
        snapshot.palette = PaletteSettings::maybe_from_colors(
            self.state.borrow().palette(),
            &theme::DEFAULT_WAVEFORM_PALETTE,
        );
        Some(ModuleSettings::with_config(&snapshot))
    }
}

impl WaveformVisual {
    fn ensure_capacity(&mut self) {
        let target = self
            .state
            .borrow()
            .desired_columns()
            .clamp(MIN_COLUMN_CAPACITY, MAX_COLUMN_CAPACITY);
        let mut config = self.processor.config();
        if config.max_columns != target {
            config.max_columns = target;
            self.processor.update_config(config);
        }
    }
}

struct SpectrogramVisual {
    processor: SpectrogramProcessor,
    state: Rc<RefCell<SpectrogramState>>,
}

impl Default for SpectrogramVisual {
    fn default() -> Self {
        Self {
            processor: SpectrogramProcessor::new(DEFAULT_SAMPLE_RATE),
            state: rc_cell(SpectrogramState::new()),
        }
    }
}

impl VisualModule for SpectrogramVisual {
    fn ingest(&mut self, samples: &[f32], format: MeterFormat) {
        if let ProcessorUpdate::Snapshot(update) = self.processor.ingest(samples, format) {
            self.state.borrow_mut().apply_update(&update);
        }
    }

    fn content(&self) -> VisualContent {
        VisualContent::Spectrogram {
            state: self.state.clone(),
        }
    }

    fn apply_settings(&mut self, settings: &ModuleSettings) {
        if let Some(stored) = settings.config::<SpectrogramSettings>() {
            let mut config = self.processor.config();
            stored.apply_to(&mut config);
            self.processor.update_config(config);
            self.state.borrow_mut().set_palette(resolve_palette(
                &stored.palette,
                &theme::DEFAULT_SPECTROGRAM_PALETTE,
            ));
        }
    }

    fn export_settings(&self) -> Option<ModuleSettings> {
        let mut snapshot = SpectrogramSettings::from_config(&self.processor.config());
        let palette = self.state.borrow().palette();
        snapshot.palette =
            PaletteSettings::maybe_from_colors(&palette, &theme::DEFAULT_SPECTROGRAM_PALETTE);
        Some(ModuleSettings::with_config(&snapshot))
    }
}

struct SpectrumVisual {
    processor: SpectrumProcessor,
    state: Rc<RefCell<SpectrumState>>,
}

impl Default for SpectrumVisual {
    fn default() -> Self {
        Self {
            processor: SpectrumProcessor::new(DEFAULT_SAMPLE_RATE),
            state: rc_cell(SpectrumState::new()),
        }
    }
}

impl VisualModule for SpectrumVisual {
    fn ingest(&mut self, samples: &[f32], format: MeterFormat) {
        if let Some(snapshot) = self.processor.ingest(samples, format) {
            self.state.borrow_mut().apply_snapshot(&snapshot);
        }
    }

    fn content(&self) -> VisualContent {
        VisualContent::Spectrum {
            state: self.state.clone(),
        }
    }

    fn apply_settings(&mut self, settings: &ModuleSettings) {
        if let Some(stored) = settings.config::<SpectrumSettings>() {
            let mut config = self.processor.config();
            stored.apply_to(&mut config);
            self.processor.update_config(config);

            let mut state = self.state.borrow_mut();
            state.set_palette(&resolve_palette(
                &stored.palette,
                &theme::DEFAULT_SPECTRUM_PALETTE,
            ));

            let updated = self.processor.config();
            let style = state.style_mut();
            style.frequency_scale = updated.frequency_scale;
            style.reverse_frequency = updated.reverse_frequency;
            style.smoothing_radius = stored.smoothing_radius;
            style.smoothing_passes = stored.smoothing_passes;
            state.update_show_grid(updated.show_grid);
            state.update_show_peak_label(updated.show_peak_label);
        }
    }

    fn export_settings(&self) -> Option<ModuleSettings> {
        let mut spectrum_settings = SpectrumSettings::from_config(&self.processor.config());
        let state = self.state.borrow();
        let palette = state.palette();
        spectrum_settings.palette =
            PaletteSettings::maybe_from_colors(&palette, &theme::DEFAULT_SPECTRUM_PALETTE);
        let style = state.style();
        spectrum_settings.smoothing_radius = style.smoothing_radius;
        spectrum_settings.smoothing_passes = style.smoothing_passes;
        Some(ModuleSettings::with_config(&spectrum_settings))
    }
}

struct StereometerVisual {
    processor: StereometerProcessor,
    state: Rc<RefCell<StereometerState>>,
}

impl Default for StereometerVisual {
    fn default() -> Self {
        Self {
            processor: StereometerProcessor::new(DEFAULT_SAMPLE_RATE),
            state: rc_cell(StereometerState::new()),
        }
    }
}

impl VisualModule for StereometerVisual {
    fn ingest(&mut self, samples: &[f32], format: MeterFormat) {
        let snapshot = self.processor.ingest(samples, format);
        self.state.borrow_mut().apply_snapshot(&snapshot);
    }

    fn content(&self) -> VisualContent {
        VisualContent::Stereometer {
            state: self.state.clone(),
        }
    }

    fn apply_settings(&mut self, settings: &ModuleSettings) {
        if let Some(stored) = settings.config::<StereometerSettings>() {
            let mut config = self.processor.config();
            stored.apply_to(&mut config);
            self.processor.update_config(config);

            let mut state = self.state.borrow_mut();
            state.update_view_settings(&stored);
            state.set_palette(&resolve_palette(
                &stored.palette,
                &theme::DEFAULT_STEREOMETER_PALETTE,
            ));
        }
    }

    fn export_settings(&self) -> Option<ModuleSettings> {
        let state = self.state.borrow();
        let (persistence, mode, scale, scale_range, rotation) = state.view_settings();
        let mut snapshot = StereometerSettings::from_config(&self.processor.config());
        snapshot.persistence = persistence;
        snapshot.mode = mode;
        snapshot.scale = scale;
        snapshot.scale_range = scale_range;
        snapshot.rotation = rotation;
        snapshot.palette = PaletteSettings::maybe_from_colors(
            &state.palette(),
            &theme::DEFAULT_STEREOMETER_PALETTE,
        );
        Some(ModuleSettings::with_config(&snapshot))
    }
}

pub struct VisualManager {
    entries: Vec<VisualEntry>,
    next_id: u32,
}

impl VisualManager {
    pub fn new() -> Self {
        let mut manager = Self {
            entries: Vec::with_capacity(VISUAL_DESCRIPTORS.len()),
            next_id: 1,
        };
        for descriptor in VISUAL_DESCRIPTORS {
            let id = VisualId::next(&mut manager.next_id);
            manager.entries.push(VisualEntry::new(id, descriptor));
        }
        manager
    }

    fn entry_mut_by_kind(&mut self, kind: VisualKind) -> Option<&mut VisualEntry> {
        self.entries.iter_mut().find(|entry| entry.kind == kind)
    }

    fn entry_by_kind(&self, kind: VisualKind) -> Option<&VisualEntry> {
        self.entries.iter().find(|entry| entry.kind == kind)
    }

    pub fn snapshot(&self) -> VisualSnapshot {
        VisualSnapshot {
            slots: self.entries.iter().map(VisualSlotSnapshot::from).collect(),
        }
    }

    pub fn module_settings(&self, kind: VisualKind) -> Option<ModuleSettings> {
        self.entry_by_kind(kind).map(|entry| {
            let mut snapshot = entry.module.export_settings().unwrap_or_default();
            if snapshot.enabled.is_none() {
                snapshot.enabled = Some(entry.enabled);
            }
            snapshot
        })
    }

    pub fn apply_module_settings(&mut self, kind: VisualKind, settings: &ModuleSettings) -> bool {
        if let Some(entry) = self.entry_mut_by_kind(kind) {
            entry.apply_settings(settings);
            true
        } else {
            false
        }
    }

    pub fn set_enabled_by_kind(&mut self, kind: VisualKind, enabled: bool) {
        if let Some(entry) = self.entry_mut_by_kind(kind)
            && entry.enabled != enabled
        {
            entry.enabled = enabled;
        }
    }

    pub fn apply_visual_settings(&mut self, settings: &VisualSettings) {
        for entry in &mut self.entries {
            if let Some(module_settings) = settings.modules.get(&entry.kind) {
                entry.apply_settings(module_settings);
            }
        }

        if !settings.order.is_empty() {
            let ordered_ids: Vec<_> = settings
                .order
                .iter()
                .filter_map(|kind| self.entry_by_kind(*kind).map(|e| e.id))
                .collect();
            if !ordered_ids.is_empty() {
                self.reorder(&ordered_ids);
            }
        }
    }

    pub fn reorder(&mut self, new_order: &[VisualId]) {
        for (position, id) in new_order.iter().enumerate() {
            if position >= self.entries.len() {
                break;
            }

            let Some(current_index) = self.entries.iter().position(|e| e.id == *id) else {
                continue;
            };

            if current_index != position {
                self.entries.swap(position, current_index);
            }
        }
    }

    /// Moves a visual to a specific index in the order.
    /// The index is clamped to valid bounds.
    pub fn restore_position(&mut self, visual_id: VisualId, target_index: usize) {
        let Some(current_index) = self.entries.iter().position(|e| e.id == visual_id) else {
            return;
        };

        let target = target_index.min(self.entries.len().saturating_sub(1));

        if current_index == target {
            return;
        }

        let entry = self.entries.remove(current_index);
        self.entries.insert(target, entry);
    }

    pub fn ingest_samples(&mut self, samples: &[f32]) {
        if samples.is_empty() {
            return;
        }

        let format = meter_tap::current_format();
        for entry in &mut self.entries {
            if entry.enabled {
                entry.module.ingest(samples, format);
                entry.cached_content = entry.module.content();
            }
        }
    }
}

#[derive(Clone)]
pub struct VisualManagerHandle {
    inner: Rc<RefCell<VisualManager>>,
}

impl VisualManagerHandle {
    pub fn new(manager: VisualManager) -> Self {
        Self {
            inner: rc_cell(manager),
        }
    }

    pub fn borrow(&self) -> Ref<'_, VisualManager> {
        self.inner.borrow()
    }

    pub fn borrow_mut(&self) -> RefMut<'_, VisualManager> {
        self.inner.borrow_mut()
    }

    pub fn snapshot(&self) -> VisualSnapshot {
        self.inner.borrow().snapshot()
    }
}

impl std::fmt::Debug for VisualManagerHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VisualManagerHandle")
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_reflects_descriptor_defaults() {
        let manager = VisualManager::new();
        let snapshot = manager.snapshot();

        assert_eq!(snapshot.slots.len(), VISUAL_DESCRIPTORS.len());

        for descriptor in VISUAL_DESCRIPTORS {
            let slot = snapshot
                .slots
                .iter()
                .find(|slot| slot.kind == descriptor.kind)
                .unwrap_or_else(|| {
                    panic!(
                        "{} slot missing from snapshot",
                        descriptor.metadata.display_name
                    )
                });

            assert!(
                !slot.enabled,
                "{} should be disabled by default",
                descriptor.metadata.display_name
            );
        }
    }
}
