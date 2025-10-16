//! this is responsible for loading and saving UI settings to disk
//!
//! responsible for:
//! - loading and saving settings
//! - providing access to settings for other parts of the UI
//! - converting between internal config structs and serializable settings structs

use crate::dsp::oscilloscope::OscilloscopeConfig;
use crate::dsp::spectrogram::{FrequencyScale, SpectrogramConfig, WindowKind};
use crate::dsp::spectrum::{AveragingMode, SpectrumConfig};
use crate::dsp::waveform::{DownsampleStrategy, WaveformConfig};
use crate::ui::visualization::visual_manager::VisualKind;
use serde::de::{self, Deserializer};
use serde::ser::Serializer;
use serde::{Deserialize, Serialize};
use serde_json::Value;
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

impl VisualSettings {
    pub fn sanitize(&mut self) {
        for (kind, module) in &mut self.modules {
            module.retain_only(*kind);
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ModuleSettings {
    pub enabled: Option<bool>,
    config: Option<StoredConfig>,
}

impl ModuleSettings {
    pub fn set_spectrogram(&mut self, config: SpectrogramSettings) {
        self.config = Some(StoredConfig::Spectrogram(config));
    }

    pub fn set_spectrum(&mut self, config: SpectrumSettings) {
        self.config = Some(StoredConfig::Spectrum(config));
    }

    pub fn set_oscilloscope(&mut self, config: OscilloscopeSettings) {
        self.config = Some(StoredConfig::Oscilloscope(config));
    }

    pub fn set_waveform(&mut self, config: WaveformSettings) {
        self.config = Some(StoredConfig::Waveform(config));
    }

    pub fn spectrogram(&self) -> Option<&SpectrogramSettings> {
        match &self.config {
            Some(StoredConfig::Spectrogram(cfg)) => Some(cfg),
            _ => None,
        }
    }

    pub fn spectrum(&self) -> Option<&SpectrumSettings> {
        match &self.config {
            Some(StoredConfig::Spectrum(cfg)) => Some(cfg),
            _ => None,
        }
    }

    pub fn oscilloscope(&self) -> Option<&OscilloscopeSettings> {
        match &self.config {
            Some(StoredConfig::Oscilloscope(cfg)) => Some(cfg),
            _ => None,
        }
    }

    pub fn waveform(&self) -> Option<&WaveformSettings> {
        match &self.config {
            Some(StoredConfig::Waveform(cfg)) => Some(cfg),
            _ => None,
        }
    }

    pub fn retain_only(&mut self, kind: VisualKind) {
        let is_configurable = matches!(
            kind,
            VisualKind::SPECTROGRAM
                | VisualKind::SPECTRUM
                | VisualKind::OSCILLOSCOPE
                | VisualKind::WAVEFORM
        );

        if !is_configurable || self.config.as_ref().is_some_and(|cfg| cfg.kind() != kind) {
            self.config = None;
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum StoredConfig {
    Spectrogram(SpectrogramSettings),
    Spectrum(SpectrumSettings),
    Oscilloscope(OscilloscopeSettings),
    Waveform(WaveformSettings),
}

impl StoredConfig {
    fn kind(&self) -> VisualKind {
        match self {
            StoredConfig::Spectrogram(_) => VisualKind::SPECTROGRAM,
            StoredConfig::Spectrum(_) => VisualKind::SPECTRUM,
            StoredConfig::Oscilloscope(_) => VisualKind::OSCILLOSCOPE,
            StoredConfig::Waveform(_) => VisualKind::WAVEFORM,
        }
    }

    fn from_value(value: Value) -> Option<Self> {
        serde_json::from_value(value).ok()
    }
}

#[derive(Serialize)]
struct ModuleSettingsSer<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    enabled: &'a Option<bool>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    config: Option<&'a StoredConfig>,
}

impl Serialize for ModuleSettings {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        ModuleSettingsSer {
            enabled: &self.enabled,
            config: self.config.as_ref(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ModuleSettings {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        let Value::Object(mut object) = value else {
            return Err(de::Error::custom("module settings must be an object"));
        };

        let enabled = object
            .remove("enabled")
            .map(|value| bool::deserialize(value).map_err(de::Error::custom))
            .transpose()?;

        let mut module = ModuleSettings {
            enabled,
            config: None,
        };

        if let Some(config_value) = object.remove("config") {
            module.config = StoredConfig::from_value(config_value);
        }

        if module.config.is_none() && !object.is_empty() {
            module.config = StoredConfig::from_value(Value::Object(object.clone()));
        }

        Ok(module)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(default, deny_unknown_fields)]
pub struct WaveformSettings {
    pub scroll_speed: f32,
    pub downsample: DownsampleStrategy,
    #[serde(default, rename = "max_columns", skip_serializing)]
    _legacy_max_columns: Option<usize>,
}

impl Default for WaveformSettings {
    fn default() -> Self {
        Self::from_config(&WaveformConfig::default())
    }
}

impl WaveformSettings {
    pub fn from_config(config: &WaveformConfig) -> Self {
        Self {
            scroll_speed: config.scroll_speed,
            downsample: config.downsample,
            _legacy_max_columns: None,
        }
    }

    pub fn apply_to(&self, config: &mut WaveformConfig) {
        config.scroll_speed = self.scroll_speed;
        config.downsample = self.downsample;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields, default)]
pub struct SpectrogramSettings {
    pub fft_size: usize,
    pub hop_size: usize,
    pub history_length: usize,
    pub window: WindowKind,
    pub frequency_scale: FrequencyScale,
    pub use_reassignment: bool,
    pub reassignment_power_floor_db: f32,
    pub reassignment_low_bin_limit: usize,
    pub zero_padding_factor: usize,
    pub use_synchrosqueezing: bool,
    pub synchrosqueezing_bin_count: usize,
    pub synchrosqueezing_min_hz: f32,
    pub temporal_smoothing: f32,
    pub temporal_smoothing_max_hz: f32,
    pub temporal_smoothing_blend_hz: f32,
    pub frequency_smoothing_radius: usize,
    pub frequency_smoothing_max_hz: f32,
    pub frequency_smoothing_blend_hz: f32,
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
            frequency_scale: config.frequency_scale,
            use_reassignment: config.use_reassignment,
            reassignment_power_floor_db: config.reassignment_power_floor_db,
            reassignment_low_bin_limit: config.reassignment_low_bin_limit,
            zero_padding_factor: config.zero_padding_factor,
            use_synchrosqueezing: config.use_synchrosqueezing,
            synchrosqueezing_bin_count: config.synchrosqueezing_bin_count,
            synchrosqueezing_min_hz: config.synchrosqueezing_min_hz,
            temporal_smoothing: config.temporal_smoothing,
            temporal_smoothing_max_hz: config.temporal_smoothing_max_hz,
            temporal_smoothing_blend_hz: config.temporal_smoothing_blend_hz,
            frequency_smoothing_radius: config.frequency_smoothing_radius,
            frequency_smoothing_max_hz: config.frequency_smoothing_max_hz,
            frequency_smoothing_blend_hz: config.frequency_smoothing_blend_hz,
        }
    }

    pub fn apply_to(&self, config: &mut SpectrogramConfig) {
        config.fft_size = self.fft_size.max(128);
        config.hop_size = self.hop_size.max(1);
        config.history_length = self.history_length.max(1);
        config.window = self.window;
        config.frequency_scale = self.frequency_scale;
        config.use_reassignment = self.use_reassignment;
        config.reassignment_power_floor_db = self.reassignment_power_floor_db.clamp(-160.0, 0.0);
        config.reassignment_low_bin_limit = self.reassignment_low_bin_limit;
        config.zero_padding_factor = self.zero_padding_factor.max(1);
        config.use_synchrosqueezing = self.use_synchrosqueezing;
        config.synchrosqueezing_bin_count = self.synchrosqueezing_bin_count.max(1);
        config.synchrosqueezing_min_hz = self.synchrosqueezing_min_hz.max(1.0);
        config.temporal_smoothing = self.temporal_smoothing.clamp(0.0, 0.9999);
        config.temporal_smoothing_max_hz = self.temporal_smoothing_max_hz.max(0.0);
        config.temporal_smoothing_blend_hz = self.temporal_smoothing_blend_hz.max(0.0);
        config.frequency_smoothing_radius = self.frequency_smoothing_radius;
        config.frequency_smoothing_max_hz = self.frequency_smoothing_max_hz.max(0.0);
        config.frequency_smoothing_blend_hz = self.frequency_smoothing_blend_hz.max(0.0);
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
        let mut data = Self::load_from_disk(&path).unwrap_or_default();
        data.visuals.sanitize();
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
        entry.set_oscilloscope(OscilloscopeSettings::from_config(config));
    }

    pub fn set_waveform_settings(&mut self, kind: VisualKind, config: &WaveformConfig) {
        let entry = self.data.visuals.modules.entry(kind).or_default();
        entry.set_waveform(WaveformSettings::from_config(config));
    }

    pub fn set_spectrogram_settings(&mut self, kind: VisualKind, config: &SpectrogramConfig) {
        let entry = self.data.visuals.modules.entry(kind).or_default();
        entry.set_spectrogram(SpectrogramSettings::from_config(config));
    }

    pub fn set_spectrum_settings(&mut self, kind: VisualKind, config: &SpectrumConfig) {
        let entry = self.data.visuals.modules.entry(kind).or_default();
        entry.set_spectrum(SpectrumSettings::from_config(config));
    }

    pub fn set_visual_order(&mut self, order: &[VisualKind]) {
        self.data.visuals.order = order.to_vec();
    }

    pub fn save(&self) -> io::Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut data = self.data.clone();
        data.visuals.sanitize();
        let json = serde_json::to_string_pretty(&data)?;
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
        manager.data.visuals.sanitize();
        if let Err(err) = manager.save() {
            error!("[settings] failed to persist UI settings: {err}");
        }
        result
    }
}
