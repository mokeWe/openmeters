//! this is responsible for loading and saving UI settings to disk
//!
//! responsible for:
//! - loading and saving settings
//! - providing access to settings for other parts of the UI
//! - converting between internal config structs and serializable settings structs

use crate::dsp::oscilloscope::OscilloscopeConfig;
use crate::dsp::spectrogram::{FrequencyScale, SpectrogramConfig, WindowKind};
use crate::dsp::spectrum::SpectrumConfig;
use crate::dsp::waveform::{DownsampleStrategy, WaveformConfig};
use crate::ui::theme;
use crate::ui::visualization::loudness::MeterMode;
use crate::ui::visualization::oscilloscope::DisplayMode;
use crate::ui::visualization::visual_manager::VisualKind;
use iced::Color;
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
        self.config = Some(config.into());
    }

    pub fn set_spectrum(&mut self, config: SpectrumSettings) {
        self.config = Some(config.into());
    }

    pub fn set_oscilloscope(&mut self, config: OscilloscopeSettings) {
        self.config = Some(config.into());
    }

    pub fn set_waveform(&mut self, config: WaveformSettings) {
        self.config = Some(config.into());
    }

    pub fn set_loudness(&mut self, config: LoudnessSettings) {
        self.config = Some(config.into());
    }

    pub fn spectrogram(&self) -> Option<&SpectrogramSettings> {
        self.config.as_ref().and_then(StoredConfig::as_spectrogram)
    }

    pub fn spectrum(&self) -> Option<&SpectrumSettings> {
        self.config.as_ref().and_then(StoredConfig::as_spectrum)
    }

    pub fn oscilloscope(&self) -> Option<&OscilloscopeSettings> {
        self.config.as_ref().and_then(StoredConfig::as_oscilloscope)
    }

    pub fn waveform(&self) -> Option<&WaveformSettings> {
        self.config.as_ref().and_then(StoredConfig::as_waveform)
    }

    pub fn loudness(&self) -> Option<&LoudnessSettings> {
        self.config.as_ref().and_then(StoredConfig::as_loudness)
    }

    pub fn with_spectrogram_settings(settings: &SpectrogramSettings) -> Self {
        Self {
            config: Some(settings.clone().into()),
            ..Default::default()
        }
    }

    pub fn with_spectrum_settings(settings: &SpectrumSettings) -> Self {
        Self {
            config: Some(settings.clone().into()),
            ..Default::default()
        }
    }

    pub fn with_waveform_settings(settings: &WaveformSettings) -> Self {
        Self {
            config: Some(settings.clone().into()),
            ..Default::default()
        }
    }

    pub fn with_oscilloscope_settings(settings: &OscilloscopeSettings) -> Self {
        Self {
            config: Some(settings.clone().into()),
            ..Default::default()
        }
    }

    pub fn with_loudness_settings(settings: LoudnessSettings) -> Self {
        Self {
            config: Some(settings.into()),
            ..Default::default()
        }
    }

    pub fn retain_only(&mut self, kind: VisualKind) {
        let is_configurable = matches!(
            kind,
            VisualKind::SPECTROGRAM
                | VisualKind::SPECTRUM
                | VisualKind::OSCILLOSCOPE
                | VisualKind::WAVEFORM
                | VisualKind::LOUDNESS
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
    Loudness(LoudnessSettings),
}

impl StoredConfig {
    fn kind(&self) -> VisualKind {
        match self {
            StoredConfig::Spectrogram(_) => VisualKind::SPECTROGRAM,
            StoredConfig::Spectrum(_) => VisualKind::SPECTRUM,
            StoredConfig::Oscilloscope(_) => VisualKind::OSCILLOSCOPE,
            StoredConfig::Waveform(_) => VisualKind::WAVEFORM,
            StoredConfig::Loudness(_) => VisualKind::LOUDNESS,
        }
    }

    fn from_value(value: Value) -> Option<Self> {
        serde_json::from_value(value).ok()
    }

    fn as_spectrogram(&self) -> Option<&SpectrogramSettings> {
        match self {
            StoredConfig::Spectrogram(cfg) => Some(cfg),
            _ => None,
        }
    }

    fn as_spectrum(&self) -> Option<&SpectrumSettings> {
        match self {
            StoredConfig::Spectrum(cfg) => Some(cfg),
            _ => None,
        }
    }

    fn as_oscilloscope(&self) -> Option<&OscilloscopeSettings> {
        match self {
            StoredConfig::Oscilloscope(cfg) => Some(cfg),
            _ => None,
        }
    }

    fn as_waveform(&self) -> Option<&WaveformSettings> {
        match self {
            StoredConfig::Waveform(cfg) => Some(cfg),
            _ => None,
        }
    }

    fn as_loudness(&self) -> Option<&LoudnessSettings> {
        match self {
            StoredConfig::Loudness(cfg) => Some(cfg),
            _ => None,
        }
    }
}

impl From<SpectrogramSettings> for StoredConfig {
    fn from(settings: SpectrogramSettings) -> Self {
        StoredConfig::Spectrogram(settings)
    }
}

impl From<SpectrumSettings> for StoredConfig {
    fn from(settings: SpectrumSettings) -> Self {
        StoredConfig::Spectrum(settings)
    }
}

impl From<OscilloscopeSettings> for StoredConfig {
    fn from(settings: OscilloscopeSettings) -> Self {
        StoredConfig::Oscilloscope(settings)
    }
}

impl From<WaveformSettings> for StoredConfig {
    fn from(settings: WaveformSettings) -> Self {
        StoredConfig::Waveform(settings)
    }
}

impl From<LoudnessSettings> for StoredConfig {
    fn from(settings: LoudnessSettings) -> Self {
        StoredConfig::Loudness(settings)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct ColorSetting {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl From<Color> for ColorSetting {
    fn from(c: Color) -> Self {
        Self {
            r: c.r,
            g: c.g,
            b: c.b,
            a: c.a,
        }
    }
}

impl ColorSetting {
    pub fn to_color(self) -> Color {
        Color {
            r: self.r,
            g: self.g,
            b: self.b,
            a: self.a,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct PaletteSettings {
    pub stops: Vec<ColorSetting>,
}

impl PaletteSettings {
    pub fn to_array<const N: usize>(&self) -> Option<[Color; N]> {
        (self.stops.len() == N).then(|| std::array::from_fn(|i| self.stops[i].to_color()))
    }

    pub fn maybe_from_colors(colors: &[Color], defaults: &[Color]) -> Option<Self> {
        if colors.len() != defaults.len() {
            return None;
        }

        if colors
            .iter()
            .zip(defaults)
            .all(|(color, default)| theme::colors_equal(*color, *default))
        {
            return None;
        }

        Some(Self {
            stops: colors.iter().copied().map(ColorSetting::from).collect(),
        })
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
    #[serde(default)]
    pub display_mode: DisplayMode,
    #[serde(default)]
    pub palette: Option<PaletteSettings>,
}

impl Default for OscilloscopeSettings {
    fn default() -> Self {
        Self::from_config(&OscilloscopeConfig::default())
    }
}

impl OscilloscopeSettings {
    pub fn from_config(config: &OscilloscopeConfig) -> Self {
        Self::from_config_with_view(config, 0.85, DisplayMode::default())
    }

    pub fn from_config_with_view(
        config: &OscilloscopeConfig,
        persistence: f32,
        display_mode: DisplayMode,
    ) -> Self {
        Self {
            segment_duration: config.segment_duration,
            trigger_level: config.trigger_level,
            trigger_rising: config.trigger_rising,
            target_sample_count: config.target_sample_count,
            persistence,
            display_mode,
            palette: None,
        }
    }

    pub fn apply_to(&self, config: &mut OscilloscopeConfig) {
        config.segment_duration = self.segment_duration;
        config.trigger_level = self.trigger_level;
        config.trigger_rising = self.trigger_rising;
        config.target_sample_count = self.target_sample_count;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WaveformSettings {
    pub scroll_speed: f32,
    pub downsample: DownsampleStrategy,
    #[serde(default, rename = "max_columns", skip_serializing)]
    _legacy_max_columns: Option<usize>,
    #[serde(default)]
    pub palette: Option<PaletteSettings>,
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
            palette: None,
        }
    }

    pub fn apply_to(&self, config: &mut WaveformConfig) {
        config.scroll_speed = self.scroll_speed;
        config.downsample = self.downsample;
    }

    pub fn to_config(&self) -> WaveformConfig {
        let mut config = WaveformConfig::default();
        self.apply_to(&mut config);
        config
    }

    pub fn palette_array<const N: usize>(&self) -> Option<[Color; N]> {
        self.palette
            .as_ref()
            .and_then(PaletteSettings::to_array::<N>)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SpectrumSettings {
    #[serde(flatten)]
    pub config: SpectrumConfig,
    #[serde(default)]
    pub palette: Option<PaletteSettings>,
}

impl Default for SpectrumSettings {
    fn default() -> Self {
        Self::from_config(&SpectrumConfig::default())
    }
}

impl SpectrumSettings {
    pub fn from_config(config: &SpectrumConfig) -> Self {
        Self {
            config: config.normalized(),
            palette: None,
        }
    }

    pub fn apply_to(&self, config: &mut SpectrumConfig) {
        *config = self.config.normalized();
    }

    pub fn to_config(&self) -> SpectrumConfig {
        self.config.normalized()
    }

    pub fn palette_array<const N: usize>(&self) -> Option<[Color; N]> {
        self.palette
            .as_ref()
            .and_then(PaletteSettings::to_array::<N>)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LoudnessSettings {
    pub left_mode: MeterMode,
    pub right_mode: MeterMode,
}

impl LoudnessSettings {
    pub const fn new(left_mode: MeterMode, right_mode: MeterMode) -> Self {
        Self {
            left_mode,
            right_mode,
        }
    }
}

impl Default for LoudnessSettings {
    fn default() -> Self {
        Self::new(MeterMode::TruePeak, MeterMode::LufsShortTerm)
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
    #[serde(default)]
    pub palette: Option<PaletteSettings>,
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
            palette: None,
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

    pub fn to_config(&self) -> SpectrogramConfig {
        let mut config = SpectrogramConfig::default();
        self.apply_to(&mut config);
        config
    }

    pub fn palette_array<const N: usize>(&self) -> Option<[Color; N]> {
        self.palette
            .as_ref()
            .and_then(PaletteSettings::to_array::<N>)
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

    pub fn set_oscilloscope_settings(&mut self, kind: VisualKind, settings: &OscilloscopeSettings) {
        let entry = self.data.visuals.modules.entry(kind).or_default();
        entry.set_oscilloscope(settings.clone());
    }

    pub fn set_waveform_settings(&mut self, kind: VisualKind, settings: &WaveformSettings) {
        let entry = self.data.visuals.modules.entry(kind).or_default();
        entry.set_waveform(settings.clone());
    }

    pub fn set_spectrogram_settings(&mut self, kind: VisualKind, settings: &SpectrogramSettings) {
        let entry = self.data.visuals.modules.entry(kind).or_default();
        entry.set_spectrogram(settings.clone());
    }

    pub fn set_spectrum_settings(&mut self, kind: VisualKind, settings: &SpectrumSettings) {
        let entry = self.data.visuals.modules.entry(kind).or_default();
        entry.set_spectrum(settings.clone());
    }

    pub fn set_loudness_settings(&mut self, kind: VisualKind, settings: LoudnessSettings) {
        let entry = self.data.visuals.modules.entry(kind).or_default();
        entry.set_loudness(settings);
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
