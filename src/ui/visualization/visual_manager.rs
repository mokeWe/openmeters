//! central owner of visual modules and their state.
//!
//! the visual manager handles the following:
//! - instantiation of visual modules
//! - applying settings to mods
//! - exporting settings to `../settings.rs`
//! - ingesting audio samples and distributing them to enabled modules

use crate::audio::meter_tap::{self, MeterFormat};
use crate::dsp::ProcessorUpdate;
use crate::dsp::waveform::{MAX_COLUMN_CAPACITY, MIN_COLUMN_CAPACITY};
use crate::ui::render::spectrogram::SPECTROGRAM_PALETTE_SIZE;
use crate::ui::settings::{
    LoudnessSettings, ModuleSettings, OscilloscopeSettings, PaletteSettings, SpectrogramSettings,
    SpectrumSettings, VisualSettings, WaveformSettings,
};
use crate::ui::theme;
use crate::ui::visualization::loudness::{LoudnessMeterProcessor, LoudnessMeterState};
use crate::ui::visualization::oscilloscope::{OscilloscopeProcessor, OscilloscopeState};
use crate::ui::visualization::spectrogram::{SpectrogramProcessor, SpectrogramState};
use crate::ui::visualization::spectrum::{SpectrumProcessor, SpectrumState};
use crate::ui::visualization::waveform::{WaveformProcessor as WaveformUiProcessor, WaveformState};
use crate::util::audio::DEFAULT_SAMPLE_RATE;
use serde::de::{self, Deserializer};
use serde::ser::Serializer;
use serde::{Deserialize, Serialize};
use std::cell::{Ref, RefCell, RefMut};
use std::collections::HashMap;
use std::rc::Rc;

/// identifiers for visual kinds
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VisualKind(&'static str);

impl VisualKind {
    pub const LOUDNESS: Self = Self("loudness");
    pub const OSCILLOSCOPE: Self = Self("oscilloscope");
    pub const SPECTROGRAM: Self = Self("spectrogram");
    pub const SPECTRUM: Self = Self("spectrum");
    pub const WAVEFORM: Self = Self("waveform");
}

impl Serialize for VisualKind {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.0)
    }
}

impl<'de> Deserialize<'de> for VisualKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let token = String::deserialize(deserializer)?;
        match token.as_str() {
            "loudness" => Ok(Self::LOUDNESS),
            "oscilloscope" => Ok(Self::OSCILLOSCOPE),
            "spectrogram" => Ok(Self::SPECTROGRAM),
            "spectrum" => Ok(Self::SPECTRUM),
            "waveform" => Ok(Self::WAVEFORM),
            _ => Err(de::Error::unknown_variant(
                &token,
                &[
                    "loudness",
                    "oscilloscope",
                    "spectrogram",
                    "spectrum",
                    "waveform",
                ],
            )),
        }
    }
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

/// Layout hints surfaced to pane planners.
#[derive(Debug, Clone, Copy)]
pub struct VisualLayoutHint {
    pub preferred_width: f32,
    pub preferred_height: f32,
    pub fill_horizontal: bool,
    pub fill_vertical: bool,
    pub min_width: f32,
    pub max_width: f32,
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
    pub layout_hint: VisualLayoutHint,
    pub content: VisualContent,
}

/// what we need to render a visual
#[derive(Debug, Clone)]
pub enum VisualContent {
    LoudnessMeter { state: LoudnessMeterState },
    Oscilloscope { state: OscilloscopeState },
    Spectrogram { state: Box<SpectrogramState> },
    Spectrum { state: SpectrumState },
    Waveform { state: WaveformState },
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
    default_enabled: bool,
    build: fn() -> Box<dyn VisualModule>,
}

struct VisualSlot {
    id: VisualId,
    kind: VisualKind,
    enabled: bool,
    metadata: VisualMetadata,
}

impl VisualSlot {
    fn new(id: VisualId, descriptor: &VisualDescriptor) -> Self {
        Self {
            id,
            kind: descriptor.kind,
            enabled: descriptor.default_enabled,
            metadata: descriptor.metadata,
        }
    }

    fn layout_hint(&self) -> VisualLayoutHint {
        VisualLayoutHint {
            preferred_width: self.metadata.preferred_width,
            preferred_height: self.metadata.preferred_height,
            fill_horizontal: self.metadata.fill_horizontal,
            fill_vertical: self.metadata.fill_vertical,
            min_width: self.metadata.min_width,
            max_width: self.metadata.max_width,
        }
    }
}

struct VisualEntry {
    slot: VisualSlot,
    module: Box<dyn VisualModule>,
}

// describe all available visuals here
const VISUAL_DESCRIPTORS: &[VisualDescriptor] = &[
    VisualDescriptor {
        kind: VisualKind::LOUDNESS,
        metadata: VisualMetadata {
            display_name: "Loudness Meter",
            preferred_width: 200.0,
            preferred_height: 300.0,
            fill_horizontal: true,
            fill_vertical: true,
            min_width: 200.0,
            max_width: 300.0,
        },
        default_enabled: true,
        build: build_module::<LoudnessVisual>,
    },
    VisualDescriptor {
        kind: VisualKind::OSCILLOSCOPE,
        metadata: VisualMetadata {
            display_name: "Oscilloscope",
            preferred_width: 220.0,
            preferred_height: 160.0,
            fill_horizontal: true,
            fill_vertical: true,
            min_width: 160.0,
            max_width: f32::INFINITY,
        },
        default_enabled: false,
        build: build_module::<OscilloscopeVisual>,
    },
    VisualDescriptor {
        kind: VisualKind::WAVEFORM,
        metadata: VisualMetadata {
            display_name: "Waveform",
            preferred_width: 320.0,
            preferred_height: 180.0,
            fill_horizontal: true,
            fill_vertical: true,
            min_width: 220.0,
            max_width: f32::INFINITY,
        },
        default_enabled: true,
        build: build_module::<WaveformVisual>,
    },
    VisualDescriptor {
        kind: VisualKind::SPECTROGRAM,
        metadata: VisualMetadata {
            display_name: "Spectrogram",
            preferred_width: 320.0,
            preferred_height: 220.0,
            fill_horizontal: true,
            fill_vertical: true,
            min_width: 200.0,
            max_width: f32::INFINITY,
        },
        default_enabled: false,
        build: build_module::<SpectrogramVisual>,
    },
    VisualDescriptor {
        kind: VisualKind::SPECTRUM,
        metadata: VisualMetadata {
            display_name: "Spectrum analyzer",
            preferred_width: 320.0,
            preferred_height: 180.0,
            fill_horizontal: true,
            fill_vertical: true,
            min_width: 200.0,
            max_width: f32::INFINITY,
        },
        default_enabled: false,
        build: build_module::<SpectrumVisual>,
    },
];

const WAVEFORM_PALETTE_SIZE: usize = theme::DEFAULT_WAVEFORM_PALETTE.len();

fn build_module<M>() -> Box<dyn VisualModule>
where
    M: VisualModule + Default + 'static,
{
    Box::new(M::default())
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
        if let Some(loudness_settings) = settings.loudness() {
            self.state
                .set_modes(loudness_settings.left_mode, loudness_settings.right_mode);
        }
    }

    fn export_settings(&self) -> Option<ModuleSettings> {
        let mut module = ModuleSettings::default();
        module.set_loudness(LoudnessSettings::new(
            self.state.left_mode(),
            self.state.right_mode(),
        ));
        Some(module)
    }
}

struct OscilloscopeVisual {
    processor: OscilloscopeProcessor,
    state: OscilloscopeState,
}

impl Default for OscilloscopeVisual {
    fn default() -> Self {
        Self {
            processor: OscilloscopeProcessor::new(DEFAULT_SAMPLE_RATE),
            state: OscilloscopeState::new(),
        }
    }
}

impl VisualModule for OscilloscopeVisual {
    fn ingest(&mut self, samples: &[f32], format: MeterFormat) {
        let snapshot = self.processor.ingest(samples, format);
        self.state.apply_snapshot(&snapshot);
    }

    fn content(&self) -> VisualContent {
        VisualContent::Oscilloscope {
            state: self.state.clone(),
        }
    }

    fn apply_settings(&mut self, settings: &ModuleSettings) {
        if let Some(stored) = settings.oscilloscope() {
            let mut config = self.processor.config();
            stored.apply_to(&mut config);
            self.processor.update_config(config);
            self.state.update_view_settings(stored);
        }
    }

    fn export_settings(&self) -> Option<ModuleSettings> {
        let mut settings = ModuleSettings::default();
        let (persistence, mode) = self.state.view_settings();
        let snapshot = OscilloscopeSettings::from_config_with_view(
            &self.processor.config(),
            persistence,
            mode,
        );
        settings.set_oscilloscope(snapshot);
        Some(settings)
    }
}

struct WaveformVisual {
    processor: WaveformUiProcessor,
    state: WaveformState,
}

impl Default for WaveformVisual {
    fn default() -> Self {
        Self {
            processor: WaveformUiProcessor::new(DEFAULT_SAMPLE_RATE),
            state: WaveformState::new(),
        }
    }
}

impl VisualModule for WaveformVisual {
    fn ingest(&mut self, samples: &[f32], format: MeterFormat) {
        self.ensure_capacity();
        let snapshot = self.processor.ingest(samples, format);
        self.state.apply_snapshot(snapshot);
    }

    fn content(&self) -> VisualContent {
        VisualContent::Waveform {
            state: self.state.clone(),
        }
    }

    fn apply_settings(&mut self, settings: &ModuleSettings) {
        if let Some(stored) = settings.waveform() {
            let mut config = self.processor.config();
            stored.apply_to(&mut config);
            self.processor.update_config(config);
            self.ensure_capacity();

            if let Some(palette) = stored.palette_array::<WAVEFORM_PALETTE_SIZE>() {
                self.state.set_palette(&palette);
            } else {
                self.state.set_palette(&theme::DEFAULT_WAVEFORM_PALETTE);
            }
        }
    }

    fn export_settings(&self) -> Option<ModuleSettings> {
        let mut module = ModuleSettings::default();
        let mut snapshot = WaveformSettings::from_config(&self.processor.config());
        snapshot.palette = PaletteSettings::maybe_from_colors(
            self.state.palette(),
            &theme::DEFAULT_WAVEFORM_PALETTE,
        );
        module.set_waveform(snapshot);
        Some(module)
    }
}

impl WaveformVisual {
    fn ensure_capacity(&mut self) {
        let mut config = self.processor.config();
        let target = self.target_capacity();
        if config.max_columns != target {
            config.max_columns = target;
            self.processor.update_config(config);
        }
    }

    fn target_capacity(&self) -> usize {
        let desired = self.state.desired_columns();
        desired.clamp(MIN_COLUMN_CAPACITY, MAX_COLUMN_CAPACITY)
    }
}

struct SpectrogramVisual {
    processor: SpectrogramProcessor,
    state: Box<SpectrogramState>,
}

impl Default for SpectrogramVisual {
    fn default() -> Self {
        Self {
            processor: SpectrogramProcessor::new(DEFAULT_SAMPLE_RATE),
            state: Box::new(SpectrogramState::new()),
        }
    }
}

impl VisualModule for SpectrogramVisual {
    fn ingest(&mut self, samples: &[f32], format: MeterFormat) {
        if let ProcessorUpdate::Snapshot(update) = self.processor.ingest(samples, format) {
            self.state.apply_update(&update);
        }
    }

    fn content(&self) -> VisualContent {
        VisualContent::Spectrogram {
            state: self.state.clone(),
        }
    }

    fn apply_settings(&mut self, settings: &ModuleSettings) {
        if let Some(stored) = settings.spectrogram() {
            let mut config = self.processor.config();
            stored.apply_to(&mut config);
            self.processor.update_config(config);

            if let Some(palette) = stored.palette_array::<SPECTROGRAM_PALETTE_SIZE>() {
                self.state.set_palette(palette);
            } else {
                self.state.set_palette(theme::DEFAULT_SPECTROGRAM_PALETTE);
            }
        }
    }

    fn export_settings(&self) -> Option<ModuleSettings> {
        let mut settings = ModuleSettings::default();
        let mut snapshot = SpectrogramSettings::from_config(&self.processor.config());
        let palette = self.state.palette();
        snapshot.palette =
            PaletteSettings::maybe_from_colors(&palette, &theme::DEFAULT_SPECTROGRAM_PALETTE);
        settings.set_spectrogram(snapshot);
        Some(settings)
    }
}

struct SpectrumVisual {
    processor: SpectrumProcessor,
    state: SpectrumState,
}

impl Default for SpectrumVisual {
    fn default() -> Self {
        Self {
            processor: SpectrumProcessor::new(DEFAULT_SAMPLE_RATE),
            state: SpectrumState::new(),
        }
    }
}

impl VisualModule for SpectrumVisual {
    fn ingest(&mut self, samples: &[f32], format: MeterFormat) {
        if let Some(snapshot) = self.processor.ingest(samples, format) {
            self.state.apply_snapshot(&snapshot);
        }
    }

    fn content(&self) -> VisualContent {
        VisualContent::Spectrum {
            state: self.state.clone(),
        }
    }

    fn apply_settings(&mut self, settings: &ModuleSettings) {
        if let Some(stored) = settings.spectrum() {
            let mut config = self.processor.config();
            stored.apply_to(&mut config);
            self.processor.update_config(config);
            self.state.style_mut().frequency_scale = config.frequency_scale;
        }
    }

    fn export_settings(&self) -> Option<ModuleSettings> {
        let mut settings = ModuleSettings::default();
        let snapshot = SpectrumSettings::from_config(&self.processor.config());
        settings.set_spectrum(snapshot);
        Some(settings)
    }
}

/// owner of visual mods and their state
pub struct VisualManager {
    entries: Vec<VisualEntry>,
    id_index: HashMap<VisualId, usize>,
    next_id: u32,
}

impl VisualManager {
    pub fn new() -> Self {
        let mut manager = Self {
            entries: Vec::new(),
            id_index: HashMap::new(),
            next_id: 1,
        };
        manager.bootstrap_defaults();
        manager
    }

    fn bootstrap_defaults(&mut self) {
        self.entries.reserve(VISUAL_DESCRIPTORS.len());
        for descriptor in VISUAL_DESCRIPTORS {
            self.insert_entry(descriptor);
        }
    }

    fn insert_entry(&mut self, descriptor: &VisualDescriptor) {
        let id = VisualId::next(&mut self.next_id);
        let slot = VisualSlot::new(id, descriptor);
        let module = (descriptor.build)();

        let index = self.entries.len();
        self.id_index.insert(id, index);
        self.entries.push(VisualEntry { slot, module });
    }

    fn entry_mut_by_kind(&mut self, kind: VisualKind) -> Option<&mut VisualEntry> {
        self.entries
            .iter_mut()
            .find(|entry| entry.slot.kind == kind)
    }

    fn entry_by_kind(&self, kind: VisualKind) -> Option<&VisualEntry> {
        self.entries.iter().find(|entry| entry.slot.kind == kind)
    }

    pub fn snapshot(&self) -> VisualSnapshot {
        let mut slots = Vec::with_capacity(self.entries.len());
        for entry in &self.entries {
            slots.push(VisualSlotSnapshot {
                id: entry.slot.id,
                kind: entry.slot.kind,
                enabled: entry.slot.enabled,
                metadata: entry.slot.metadata,
                layout_hint: entry.slot.layout_hint(),
                content: entry.module.content(),
            });
        }

        VisualSnapshot { slots }
    }

    pub fn module_settings(&self, kind: VisualKind) -> Option<ModuleSettings> {
        self.entry_by_kind(kind).map(|entry| {
            let mut snapshot = entry.module.export_settings().unwrap_or_default();
            if snapshot.enabled.is_none() {
                snapshot.enabled = Some(entry.slot.enabled);
            }
            snapshot
        })
    }

    pub fn apply_module_settings(
        &mut self,
        kind: VisualKind,
        module_settings: &ModuleSettings,
    ) -> bool {
        if let Some(entry) = self.entry_mut_by_kind(kind) {
            if let Some(enabled) = module_settings.enabled {
                entry.slot.enabled = enabled;
            }
            entry.module.apply_settings(module_settings);
            true
        } else {
            false
        }
    }

    pub fn set_enabled_by_kind(&mut self, kind: VisualKind, enabled: bool) {
        if let Some(entry) = self.entry_mut_by_kind(kind)
            && entry.slot.enabled != enabled
        {
            entry.slot.enabled = enabled;
        }
    }

    pub fn apply_visual_settings(&mut self, settings: &VisualSettings) {
        for entry in &mut self.entries {
            if let Some(module_settings) = settings.modules.get(&entry.slot.kind) {
                if let Some(enabled) = module_settings.enabled {
                    entry.slot.enabled = enabled;
                }
                entry.module.apply_settings(module_settings);
            }
        }

        if !settings.order.is_empty() {
            let mut ordered_ids = Vec::with_capacity(settings.order.len());
            for kind in &settings.order {
                if let Some(entry) = self.entry_by_kind(*kind) {
                    ordered_ids.push(entry.slot.id);
                }
            }

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

            let Some(current_index) = self.id_index.get(id).copied() else {
                continue;
            };

            if current_index == position {
                continue;
            }

            self.entries.swap(position, current_index);

            let left_id = self.entries[position].slot.id;
            let right_id = self.entries[current_index].slot.id;
            self.id_index.insert(left_id, position);
            self.id_index.insert(right_id, current_index);
        }
    }

    pub fn ingest_samples(&mut self, samples: &[f32]) {
        if samples.is_empty() {
            return;
        }

        let format = meter_tap::current_format();
        for entry in &mut self.entries {
            if entry.slot.enabled {
                entry.module.ingest(samples, format);
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
            inner: Rc::new(RefCell::new(manager)),
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

    fn loudness_average(snapshot: &VisualSnapshot) -> f32 {
        snapshot
            .slots
            .iter()
            .find_map(|slot| match &slot.content {
                VisualContent::LoudnessMeter { state } => Some(state.short_term_average()),
                _ => None,
            })
            .expect("loudness meter missing from snapshot")
    }

    #[test]
    fn snapshot_includes_available_slots_enabled_by_default() {
        let manager = VisualManager::new();
        let snapshot = manager.snapshot();

        let mut loudness_enabled = false;
        let mut oscilloscope_enabled = false;
        let mut waveform_enabled = false;

        for slot in &snapshot.slots {
            if slot.kind == VisualKind::LOUDNESS {
                loudness_enabled = slot.enabled;
            }
            if slot.kind == VisualKind::OSCILLOSCOPE {
                oscilloscope_enabled = slot.enabled;
            }
            if slot.kind == VisualKind::WAVEFORM {
                waveform_enabled = slot.enabled;
            }
        }

        assert!(
            loudness_enabled,
            "Loudness meter should be enabled by default"
        );
        assert!(!oscilloscope_enabled, "Oscilloscope should start disabled");
        assert!(waveform_enabled, "Waveform should be enabled by default");
    }

    #[test]
    fn toggling_a_kind_updates_snapshot_state() {
        let mut manager = VisualManager::new();
        manager.set_enabled_by_kind(VisualKind::LOUDNESS, false);

        let snapshot = manager.snapshot();
        let loudness_slot = snapshot
            .slots
            .iter()
            .find(|slot| slot.kind == VisualKind::LOUDNESS)
            .expect("loudness slot not found");

        assert!(!loudness_slot.enabled);
    }

    #[test]
    fn reorder_changes_visual_order() {
        let mut manager = VisualManager::new();
        let snapshot = manager.snapshot();
        let mut desired_order: Vec<_> = snapshot.slots.iter().map(|slot| slot.id).collect();
        desired_order.reverse();

        manager.reorder(&desired_order);
        let snapshot_after = manager.snapshot();
        let reordered: Vec<_> = snapshot_after.slots.iter().map(|slot| slot.id).collect();

        assert_eq!(reordered, desired_order);
    }

    #[test]
    fn ingest_samples_updates_visual_content() {
        let mut manager = VisualManager::new();
        let baseline = manager.snapshot();
        let baseline_loudness = loudness_average(&baseline);

        let samples = vec![0.5_f32; 480];
        manager.ingest_samples(&samples);

        let snapshot = manager.snapshot();
        let updated_loudness = loudness_average(&snapshot);

        assert!(updated_loudness > baseline_loudness);
    }
}
