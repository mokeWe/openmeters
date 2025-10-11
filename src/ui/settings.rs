use crate::dsp::oscilloscope::OscilloscopeConfig;
use crate::dsp::spectrogram::{SpectrogramConfig, WindowKind};
use crate::dsp::spectrum::{AveragingMode, SpectrumConfig};
use crate::ui::visualization::visual_manager::VisualKind;
use serde::{Deserialize, Serialize};
use std::cell::{Ref, RefCell};
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use tracing::{error, warn};

const SETTINGS_FILE_NAME: &str = "settings.json";

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct UiSettings {
    #[serde(default)]
    pub visuals: VisualSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct VisualSettings {
    #[serde(default)]
    pub modules: HashMap<VisualKind, ModuleSettings>,
    #[serde(default)]
    pub order: Vec<VisualKind>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ModuleSettings {
    pub enabled: Option<bool>,
    pub oscilloscope: Option<OscilloscopeSettings>,
    pub spectrogram: Option<SpectrogramSettings>,
    pub spectrum: Option<SpectrumSettings>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OscilloscopeSettings {
    pub segment_duration: f32,
    pub trigger_level: f32,
    pub trigger_rising: bool,
    pub target_sample_count: usize,
    pub persistence: f32,
}

impl Default for OscilloscopeSettings {
    fn default() -> Self {
        Self::from_config(&OscilloscopeConfig::default())
    }
}

impl OscilloscopeSettings {
    pub fn from_config(config: &OscilloscopeConfig) -> Self {
        Self {
            segment_duration: config.segment_duration,
            trigger_level: config.trigger_level,
            trigger_rising: config.trigger_rising,
            target_sample_count: config.target_sample_count,
            persistence: config.persistence,
        }
    }

    pub fn apply_to(&self, config: &mut OscilloscopeConfig) {
        config.segment_duration = self.segment_duration;
        config.trigger_level = self.trigger_level;
        config.trigger_rising = self.trigger_rising;
        config.target_sample_count = self.target_sample_count;
        config.persistence = self.persistence;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SpectrumSettings {
    pub fft_size: usize,
    pub hop_size: usize,
    pub averaging_factor: f32,
}

impl Default for SpectrumSettings {
    fn default() -> Self {
        let config = SpectrumConfig::default();
        let factor = match config.averaging {
            AveragingMode::Exponential { factor } => factor,
            _ => 0.5,
        };
        Self {
            fft_size: config.fft_size,
            hop_size: config.hop_size,
            averaging_factor: factor,
        }
    }
}

impl SpectrumSettings {
    pub fn from_config(config: &SpectrumConfig) -> Self {
        let factor = match config.averaging {
            AveragingMode::Exponential { factor } => factor,
            _ => 0.5,
        };
        Self {
            fft_size: config.fft_size,
            hop_size: config.hop_size,
            averaging_factor: factor,
        }
    }

    pub fn apply_to(&self, config: &mut SpectrumConfig) {
        config.fft_size = self.fft_size.max(128);
        config.hop_size = self.hop_size.max(1);
        config.averaging = AveragingMode::Exponential {
            factor: self.averaging_factor.clamp(0.0, 0.9999),
        };
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SpectrogramSettings {
    pub fft_size: usize,
    pub hop_size: usize,
    pub history_length: usize,
    pub window: WindowKind,
    pub use_reassignment: bool,
    pub reassignment_power_floor_db: f32,
    pub zero_padding_factor: usize,
    pub use_synchrosqueezing: bool,
    pub temporal_smoothing: f32,
    pub frequency_smoothing_radius: usize,
}

impl Default for SpectrogramSettings {
    fn default() -> Self {
        Self::from_config(&SpectrogramConfig::default())
    }
}

impl SpectrogramSettings {
    pub fn from_config(config: &SpectrogramConfig) -> Self {
        Self {
            fft_size: config.fft_size,
            hop_size: config.hop_size,
            history_length: config.history_length,
            window: config.window,
            use_reassignment: config.use_reassignment,
            reassignment_power_floor_db: config.reassignment_power_floor_db,
            zero_padding_factor: config.zero_padding_factor,
            use_synchrosqueezing: config.use_synchrosqueezing,
            temporal_smoothing: config.temporal_smoothing,
            frequency_smoothing_radius: config.frequency_smoothing_radius,
        }
    }

    pub fn apply_to(&self, config: &mut SpectrogramConfig) {
        config.fft_size = self.fft_size.max(128);
        config.hop_size = self.hop_size.max(1);
        config.history_length = self.history_length.max(1);
        config.window = self.window;
        config.use_reassignment = self.use_reassignment;
        config.reassignment_power_floor_db = self.reassignment_power_floor_db.clamp(-160.0, 0.0);
        config.zero_padding_factor = self.zero_padding_factor.max(1);
        config.use_synchrosqueezing = self.use_synchrosqueezing;
        config.temporal_smoothing = self.temporal_smoothing.clamp(0.0, 0.9999);
        config.frequency_smoothing_radius = self.frequency_smoothing_radius;
    }
}

fn config_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("XDG_CONFIG_HOME") {
        PathBuf::from(dir).join("openmeters")
    } else if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".config").join("openmeters")
    } else {
        PathBuf::from(".openmeters")
    }
}

#[derive(Debug)]
pub struct SettingsManager {
    path: PathBuf,
    data: UiSettings,
}

impl SettingsManager {
    pub fn load_or_default() -> Self {
        let path = config_dir().join(SETTINGS_FILE_NAME);
        let data = Self::load_from_disk(&path).unwrap_or_default();
        Self { path, data }
    }

    fn load_from_disk(path: &Path) -> Option<UiSettings> {
        let contents = fs::read_to_string(path).ok()?;
        match serde_json::from_str(&contents) {
            Ok(settings) => Some(settings),
            Err(err) => {
                warn!("[settings] failed to parse {path:?}: {err}");
                None
            }
        }
    }

    pub fn settings(&self) -> &UiSettings {
        &self.data
    }

    pub fn set_visual_enabled(&mut self, kind: VisualKind, enabled: bool) {
        let entry = self.data.visuals.modules.entry(kind).or_default();
        entry.enabled = Some(enabled);
    }

    pub fn set_oscilloscope_settings(&mut self, kind: VisualKind, config: &OscilloscopeConfig) {
        let entry = self.data.visuals.modules.entry(kind).or_default();
        entry.oscilloscope = Some(OscilloscopeSettings::from_config(config));
    }

    pub fn set_spectrogram_settings(&mut self, kind: VisualKind, config: &SpectrogramConfig) {
        let entry = self.data.visuals.modules.entry(kind).or_default();
        entry.spectrogram = Some(SpectrogramSettings::from_config(config));
    }

    pub fn set_spectrum_settings(&mut self, kind: VisualKind, config: &SpectrumConfig) {
        let entry = self.data.visuals.modules.entry(kind).or_default();
        entry.spectrum = Some(SpectrumSettings::from_config(config));
    }

    pub fn set_visual_order(&mut self, order: &[VisualKind]) {
        self.data.visuals.order = order.to_vec();
    }

    pub fn save(&self) -> io::Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(&self.data)?;
        let tmp_path = self.path.with_extension("json.tmp");
        fs::write(&tmp_path, &json)?;
        fs::rename(&tmp_path, &self.path)
    }
}

#[derive(Debug, Clone)]
pub struct SettingsHandle {
    inner: Rc<RefCell<SettingsManager>>,
}

impl SettingsHandle {
    pub fn load_or_default() -> Self {
        Self {
            inner: Rc::new(RefCell::new(SettingsManager::load_or_default())),
        }
    }

    pub fn borrow(&self) -> Ref<'_, SettingsManager> {
        self.inner.borrow()
    }

    pub fn update<F, R>(&self, mutator: F) -> R
    where
        F: FnOnce(&mut SettingsManager) -> R,
    {
        let mut manager = self.inner.borrow_mut();
        let result = mutator(&mut manager);
        if let Err(err) = manager.save() {
            error!("[settings] failed to persist UI settings: {err}");
        }
        result
    }
}
