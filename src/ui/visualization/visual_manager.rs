use crate::dsp::ProcessorUpdate;
use crate::ui::visualization::lufs_meter::{LufsMeterState, LufsProcessor};
use crate::ui::visualization::oscilloscope::{OscilloscopeProcessor, OscilloscopeState};
use crate::ui::visualization::spectrogram::{SpectrogramProcessor, SpectrogramState};
use crate::util::audio::DEFAULT_SAMPLE_RATE;
use std::borrow::Cow;
use std::cell::{RefCell, RefMut};
use std::collections::HashMap;
use std::rc::Rc;

/// Unique identifier for a visual slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VisualId(u32);

impl VisualId {
    fn next(counter: &mut u32) -> Self {
        let id = *counter;
        *counter = counter.saturating_add(1);
        VisualId(id)
    }
}

/// Enumeration of the visual modules recognised by the UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VisualKind {
    LufsMeter,
    Oscilloscope,
    Spectrogram,
    Spectrum,
    Stereometer,
    Waveform,
}

impl VisualKind {
    pub const ALL: [VisualKind; 6] = [
        VisualKind::LufsMeter,
        VisualKind::Oscilloscope,
        VisualKind::Spectrogram,
        VisualKind::Spectrum,
        VisualKind::Stereometer,
        VisualKind::Waveform,
    ];

    pub fn metadata(self) -> VisualMetadata {
        match self {
            VisualKind::LufsMeter => VisualMetadata {
                display_name: "LUFS meter",
                available: true,
                default_enabled: true,
                preferred_width: 200.0,
                preferred_height: 300.0,
                fill_horizontal: true,
                fill_vertical: true,
                min_width: 200.0,
                max_width: 300.0,
            },
            VisualKind::Oscilloscope => VisualMetadata {
                display_name: "Oscilloscope",
                available: true,
                default_enabled: false,
                preferred_width: 220.0,
                preferred_height: 160.0,
                fill_horizontal: true,
                fill_vertical: true,
                min_width: 160.0,
                max_width: f32::INFINITY,
            },
            VisualKind::Spectrogram => VisualMetadata {
                display_name: "Spectrogram",
                available: true,
                default_enabled: false,
                preferred_width: 320.0,
                preferred_height: 220.0,
                fill_horizontal: true,
                fill_vertical: true,
                min_width: 200.0,
                max_width: f32::INFINITY,
            },
            VisualKind::Spectrum => VisualMetadata {
                display_name: "Spectrum analyzer",
                available: false,
                default_enabled: false,
                preferred_width: 320.0,
                preferred_height: 180.0,
                fill_horizontal: true,
                fill_vertical: true,
                min_width: 200.0,
                max_width: f32::INFINITY,
            },
            VisualKind::Stereometer => VisualMetadata {
                display_name: "Stereometer",
                available: false,
                default_enabled: false,
                preferred_width: 240.0,
                preferred_height: 240.0,
                fill_horizontal: false,
                fill_vertical: true,
                min_width: 240.0,
                max_width: 240.0,
            },
            VisualKind::Waveform => VisualMetadata {
                display_name: "Waveform",
                available: false,
                default_enabled: false,
                preferred_width: 320.0,
                preferred_height: 160.0,
                fill_horizontal: true,
                fill_vertical: true,
                min_width: 200.0,
                max_width: f32::INFINITY,
            },
        }
    }
}

/// Static metadata describing a visual module.
#[derive(Debug, Clone, Copy)]
pub struct VisualMetadata {
    pub display_name: &'static str,
    pub available: bool,
    pub default_enabled: bool,
    pub preferred_width: f32,
    pub preferred_height: f32,
    pub fill_horizontal: bool,
    pub fill_vertical: bool,
    pub min_width: f32,
    pub max_width: f32,
}

/// Lightweight layout hint for planning pane sizes.
#[derive(Debug, Clone, Copy)]
pub struct VisualLayoutHint {
    pub preferred_width: f32,
    pub preferred_height: f32,
    pub fill_horizontal: bool,
    pub fill_vertical: bool,
    pub min_width: f32,
    pub max_width: f32,
}

/// Snapshot returned to interested consumers summarising the current slots.
#[derive(Debug, Clone)]
pub struct VisualSnapshot {
    pub slots: Vec<VisualSlotSnapshot>,
}

/// Snapshot for a single visual slot.
#[derive(Debug, Clone)]
pub struct VisualSlotSnapshot {
    pub id: VisualId,
    pub kind: VisualKind,
    pub enabled: bool,
    pub metadata: VisualMetadata,
    pub layout_hint: VisualLayoutHint,
    pub content: VisualContent,
}

/// Snapshot of runtime data required to render a visual pane.
#[derive(Debug, Clone)]
pub enum VisualContent {
    LufsMeter { state: LufsMeterState },
    Oscilloscope { state: OscilloscopeState },
    Spectrogram { state: SpectrogramState },
    Placeholder { message: Cow<'static, str> },
}

#[derive(Debug)]
struct VisualEntry {
    slot: VisualSlot,
    runtime: VisualRuntime,
}

#[derive(Debug)]
struct VisualSlot {
    id: VisualId,
    kind: VisualKind,
    enabled: bool,
    metadata: VisualMetadata,
}

impl VisualSlot {
    fn new(id: VisualId, kind: VisualKind, metadata: VisualMetadata) -> Self {
        Self {
            id,
            kind,
            enabled: metadata.default_enabled,
            metadata,
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

#[derive(Debug)]
enum VisualRuntime {
    LufsMeter {
        processor: LufsProcessor,
        state: LufsMeterState,
    },
    Oscilloscope {
        processor: OscilloscopeProcessor,
        state: OscilloscopeState,
    },
    Spectrogram {
        processor: SpectrogramProcessor,
        state: SpectrogramState,
    },
    Placeholder,
}

const PLACEHOLDER_MESSAGE: &str = "Module not yet implemented.";

impl VisualRuntime {
    fn new(kind: VisualKind) -> Self {
        match kind {
            VisualKind::LufsMeter => VisualRuntime::LufsMeter {
                processor: LufsProcessor::new(DEFAULT_SAMPLE_RATE),
                state: LufsMeterState::new(),
            },
            VisualKind::Oscilloscope => VisualRuntime::Oscilloscope {
                processor: OscilloscopeProcessor::new(DEFAULT_SAMPLE_RATE),
                state: OscilloscopeState::new(),
            },
            VisualKind::Spectrogram => VisualRuntime::Spectrogram {
                processor: SpectrogramProcessor::new(DEFAULT_SAMPLE_RATE),
                state: SpectrogramState::new(),
            },
            _ => VisualRuntime::Placeholder,
        }
    }

    fn ingest(&mut self, samples: &[f32]) {
        match self {
            VisualRuntime::LufsMeter { processor, state } => {
                let snapshot = processor.ingest(samples);
                state.apply_snapshot(&snapshot);
            }
            VisualRuntime::Oscilloscope { processor, state } => {
                let snapshot = processor.ingest(samples);
                state.apply_snapshot(&snapshot);
            }
            VisualRuntime::Spectrogram { processor, state } => {
                if let ProcessorUpdate::Snapshot(update) = processor.ingest(samples) {
                    state.apply_update(&update);
                }
            }
            VisualRuntime::Placeholder => {}
        }
    }

    fn content(&self) -> VisualContent {
        match self {
            VisualRuntime::LufsMeter { state, .. } => VisualContent::LufsMeter {
                state: state.clone(),
            },
            VisualRuntime::Oscilloscope { state, .. } => VisualContent::Oscilloscope {
                state: state.clone(),
            },
            VisualRuntime::Spectrogram { state, .. } => VisualContent::Spectrogram {
                state: state.clone(),
            },
            VisualRuntime::Placeholder => VisualContent::Placeholder {
                message: Cow::Borrowed(PLACEHOLDER_MESSAGE),
            },
        }
    }
}

/// In-memory owner of all visual module state.
#[derive(Debug)]
pub struct VisualManager {
    entries: Vec<VisualEntry>,
    index: HashMap<VisualId, usize>,
    next_id: u32,
}

impl VisualManager {
    pub fn new() -> Self {
        let mut manager = Self {
            entries: Vec::new(),
            index: HashMap::new(),
            next_id: 1,
        };
        manager.bootstrap_defaults();
        manager
    }

    fn bootstrap_defaults(&mut self) {
        self.entries.reserve(VisualKind::ALL.len());
        for kind in VisualKind::ALL {
            let metadata = kind.metadata();
            if !metadata.available {
                continue;
            }

            self.insert_entry(kind, metadata);
        }
    }

    fn insert_entry(&mut self, kind: VisualKind, metadata: VisualMetadata) {
        let id = VisualId::next(&mut self.next_id);
        let slot = VisualSlot::new(id, kind, metadata);
        let runtime = VisualRuntime::new(kind);

        self.index.insert(id, self.entries.len());
        self.entries.push(VisualEntry { slot, runtime });
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
                content: entry.runtime.content(),
            });
        }

        VisualSnapshot { slots }
    }

    pub fn set_enabled_by_kind(&mut self, kind: VisualKind, enabled: bool) {
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|entry| entry.slot.kind == kind)
        {
            if entry.slot.enabled != enabled {
                entry.slot.enabled = enabled;
            }
        }
    }

    pub fn reorder(&mut self, new_order: &[VisualId]) {
        for (position, id) in new_order.iter().enumerate() {
            if position >= self.entries.len() {
                break;
            }

            let Some(current_index) = self.index.get(id).copied() else {
                continue;
            };

            if current_index == position {
                continue;
            }

            self.entries.swap(position, current_index);

            let left_id = self.entries[position].slot.id;
            let right_id = self.entries[current_index].slot.id;
            self.index.insert(left_id, position);
            self.index.insert(right_id, current_index);
        }
    }

    pub fn ingest_samples(&mut self, samples: &[f32]) {
        if samples.is_empty() {
            return;
        }

        for entry in &mut self.entries {
            if entry.slot.enabled {
                entry.runtime.ingest(samples);
            }
        }
    }
}

/// Shared handle used by UI components to access the visual manager.
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

    fn lufs_average(snapshot: &VisualSnapshot) -> f32 {
        snapshot
            .slots
            .iter()
            .find_map(|slot| match &slot.content {
                VisualContent::LufsMeter { state } => Some(state.short_term_average()),
                _ => None,
            })
            .expect("lufs meter missing from snapshot")
    }

    #[test]
    fn snapshot_includes_available_slots_enabled_by_default() {
        let manager = VisualManager::new();
        let snapshot = manager.snapshot();

        let mut lufs_enabled = false;
        let mut oscilloscope_enabled = false;

        for slot in &snapshot.slots {
            match slot.kind {
                VisualKind::LufsMeter => lufs_enabled = slot.enabled,
                VisualKind::Oscilloscope => oscilloscope_enabled = slot.enabled,
                _ => {}
            }
        }

        assert!(lufs_enabled, "LUFS meter should be enabled by default");
        assert!(!oscilloscope_enabled, "Oscilloscope should start disabled");
    }

    #[test]
    fn toggling_a_kind_updates_snapshot_state() {
        let mut manager = VisualManager::new();
        manager.set_enabled_by_kind(VisualKind::LufsMeter, false);

        let snapshot = manager.snapshot();
        let lufs_slot = snapshot
            .slots
            .iter()
            .find(|slot| slot.kind == VisualKind::LufsMeter)
            .expect("lufs slot not found");

        assert!(!lufs_slot.enabled);
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
        let baseline_lufs = lufs_average(&baseline);

        let samples = vec![0.5_f32; 480];
        manager.ingest_samples(&samples);

        let snapshot = manager.snapshot();
        let updated_lufs = lufs_average(&snapshot);

        assert!(updated_lufs > baseline_lufs);
    }
}
