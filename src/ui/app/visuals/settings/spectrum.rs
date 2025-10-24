use super::widgets::{SliderRange, labeled_pick_list, labeled_slider, set_f32};
use super::{ModuleSettingsPane, SettingsMessage};
use crate::dsp::spectrogram::FrequencyScale;
use crate::dsp::spectrum::{AveragingMode, SpectrumConfig};
use crate::ui::settings::{
    DEFAULT_SPECTRUM_AVERAGING_FACTOR, DEFAULT_SPECTRUM_PEAK_HOLD_DECAY, ModuleSettings,
    SettingsHandle, SpectrumAveragingMode, SpectrumSettings,
};
use crate::ui::visualization::visual_manager::{VisualId, VisualKind, VisualManagerHandle};
use iced::Element;
use iced::widget::{column, toggler};
use std::fmt;

const AVERAGING_MIN: f32 = 0.0;
const AVERAGING_MAX: f32 = 0.95;
const AVERAGING_STEP: f32 = 0.01;
const FFT_OPTIONS: [usize; 4] = [1024, 2048, 4096, 8192];
const SCALE_OPTIONS: [FrequencyScale; 3] = [
    FrequencyScale::Linear,
    FrequencyScale::Logarithmic,
    FrequencyScale::Mel,
];
const PEAK_HOLD_DECAY_MIN: f32 = 0.0;
const PEAK_HOLD_DECAY_MAX: f32 = 60.0;
const PEAK_HOLD_DECAY_STEP: f32 = 0.5;
const AVERAGING_OPTIONS: [SpectrumAveragingMode; 3] = [
    SpectrumAveragingMode::None,
    SpectrumAveragingMode::Exponential,
    SpectrumAveragingMode::PeakHold,
];
const EXPONENTIAL_RANGE: SliderRange =
    SliderRange::new(AVERAGING_MIN, AVERAGING_MAX, AVERAGING_STEP);
const PEAK_HOLD_DECAY_RANGE: SliderRange = SliderRange::new(
    PEAK_HOLD_DECAY_MIN,
    PEAK_HOLD_DECAY_MAX,
    PEAK_HOLD_DECAY_STEP,
);

impl fmt::Display for SpectrumAveragingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpectrumAveragingMode::None => f.write_str("None"),
            SpectrumAveragingMode::Exponential => f.write_str("Exponential"),
            SpectrumAveragingMode::PeakHold => f.write_str("Peak hold"),
        }
    }
}

#[derive(Debug)]
pub struct SpectrumSettingsPane {
    visual_id: VisualId,
    config: SpectrumConfig,
    averaging_mode: SpectrumAveragingMode,
    averaging_factor: f32,
    peak_hold_decay: f32,
}

#[derive(Debug, Clone)]
pub enum Message {
    FftSize(usize),
    AveragingMode(SpectrumAveragingMode),
    AveragingFactor(f32),
    PeakHoldDecay(f32),
    FrequencyScale(FrequencyScale),
    ReverseFrequency(bool),
}

pub fn create(visual_id: VisualId, visual_manager: &VisualManagerHandle) -> SpectrumSettingsPane {
    let config = visual_manager
        .borrow()
        .module_settings(VisualKind::SPECTRUM)
        .and_then(|stored| stored.spectrum_config())
        .unwrap_or_default();
    let (averaging_mode, averaging_factor, peak_hold_decay) =
        SpectrumSettings::split_averaging(config.averaging);

    let mut pane = SpectrumSettingsPane {
        visual_id,
        config,
        averaging_mode,
        averaging_factor,
        peak_hold_decay,
    };
    pane.sync_averaging_config();
    pane
}

impl ModuleSettingsPane for SpectrumSettingsPane {
    fn visual_id(&self) -> VisualId {
        self.visual_id
    }

    fn view(&self) -> Element<'_, SettingsMessage> {
        let fft_row = labeled_pick_list(
            "FFT size",
            FFT_OPTIONS.as_slice(),
            Some(self.config.fft_size),
            |size| SettingsMessage::Spectrum(Message::FftSize(size)),
        )
        .spacing(12);

        let scale_row = labeled_pick_list(
            "Frequency scale",
            SCALE_OPTIONS.as_slice(),
            Some(self.config.frequency_scale),
            |scale| SettingsMessage::Spectrum(Message::FrequencyScale(scale)),
        )
        .spacing(12);

        let direction_label = if self.config.reverse_frequency {
            "High → Low"
        } else {
            "Low → High"
        };

        let direction_toggle = toggler(self.config.reverse_frequency)
            .label(direction_label)
            .spacing(8)
            .text_size(11)
            .on_toggle(|value| SettingsMessage::Spectrum(Message::ReverseFrequency(value)));

        let mode_row = labeled_pick_list(
            "Averaging mode",
            AVERAGING_OPTIONS.as_slice(),
            Some(self.averaging_mode),
            |mode| SettingsMessage::Spectrum(Message::AveragingMode(mode)),
        )
        .spacing(12);

        let mut content = column![fft_row, scale_row, direction_toggle, mode_row].spacing(16);

        match self.averaging_mode {
            SpectrumAveragingMode::Exponential => {
                content = content.push(labeled_slider(
                    "Exponential factor",
                    self.averaging_factor,
                    format!("{:.2}", self.averaging_factor),
                    EXPONENTIAL_RANGE,
                    |value| SettingsMessage::Spectrum(Message::AveragingFactor(value)),
                ));
            }
            SpectrumAveragingMode::PeakHold => {
                content = content.push(labeled_slider(
                    "Peak decay (dB/s)",
                    self.peak_hold_decay,
                    format!("{:.1} dB/s", self.peak_hold_decay),
                    PEAK_HOLD_DECAY_RANGE,
                    |value| SettingsMessage::Spectrum(Message::PeakHoldDecay(value)),
                ));
            }
            SpectrumAveragingMode::None => {}
        }

        content.into()
    }

    fn handle(
        &mut self,
        message: &SettingsMessage,
        visual_manager: &VisualManagerHandle,
        settings: &SettingsHandle,
    ) {
        let SettingsMessage::Spectrum(msg) = message else {
            return;
        };

        let changed = match msg {
            Message::FftSize(size) => {
                if self.config.fft_size != *size {
                    self.config.fft_size = *size;
                    self.config.hop_size = (size / 4).max(1);
                    true
                } else {
                    false
                }
            }
            Message::AveragingMode(mode) => {
                if self.averaging_mode != *mode {
                    self.averaging_mode = *mode;
                    self.sync_averaging_config();
                    true
                } else {
                    false
                }
            }
            Message::AveragingFactor(value) => {
                let snapped = SpectrumSettings::normalized_factor(*value);
                if set_f32(&mut self.averaging_factor, snapped) {
                    self.sync_averaging_config();
                    true
                } else {
                    false
                }
            }
            Message::PeakHoldDecay(value) => {
                let snapped = SpectrumSettings::normalized_decay(*value);
                if set_f32(&mut self.peak_hold_decay, snapped) {
                    self.sync_averaging_config();
                    true
                } else {
                    false
                }
            }
            Message::FrequencyScale(scale) => {
                if self.config.frequency_scale != *scale {
                    self.config.frequency_scale = *scale;
                    true
                } else {
                    false
                }
            }
            Message::ReverseFrequency(value) => {
                if self.config.reverse_frequency != *value {
                    self.config.reverse_frequency = *value;
                    true
                } else {
                    false
                }
            }
        };

        if changed {
            apply_spectrum_config(&self.config, visual_manager, settings);
        }
    }
}

fn apply_spectrum_config(
    config: &SpectrumConfig,
    visual_manager: &VisualManagerHandle,
    settings: &SettingsHandle,
) {
    let module_settings = ModuleSettings::with_spectrum_config(config);

    if visual_manager
        .borrow_mut()
        .apply_module_settings(VisualKind::SPECTRUM, &module_settings)
    {
        settings.update(|mgr| {
            mgr.set_spectrum_settings(VisualKind::SPECTRUM, config);
        });
    }
}

impl SpectrumSettingsPane {
    fn sync_averaging_config(&mut self) {
        let factor = SpectrumSettings::normalized_factor(self.averaging_factor);
        if factor.to_bits() != self.averaging_factor.to_bits() {
            self.averaging_factor = factor;
        }

        let decay = SpectrumSettings::normalized_decay(self.peak_hold_decay);
        if decay.to_bits() != self.peak_hold_decay.to_bits() {
            self.peak_hold_decay = decay;
        }

        self.config.averaging = SpectrumSettings::combine_averaging(
            self.averaging_mode,
            self.averaging_factor,
            self.peak_hold_decay,
        );
    }
}

impl Default for SpectrumSettings {
    fn default() -> Self {
        Self::from_config(&SpectrumConfig::default())
    }
}

impl SpectrumSettings {
    pub fn from_config(config: &SpectrumConfig) -> Self {
        let (mode, factor, decay) = Self::split_averaging(config.averaging);
        Self {
            fft_size: config.fft_size,
            hop_size: config.hop_size,
            averaging_mode: mode,
            averaging_factor: factor,
            peak_hold_decay: decay,
            frequency_scale: config.frequency_scale,
            reverse_frequency: config.reverse_frequency,
        }
    }

    pub fn apply_to(&self, config: &mut SpectrumConfig) {
        config.fft_size = self.fft_size.max(128);
        config.hop_size = self.hop_size.max(1);
        config.averaging = self.averaging_config();
        config.frequency_scale = self.frequency_scale;
        config.reverse_frequency = self.reverse_frequency;
    }

    pub fn to_config(&self) -> SpectrumConfig {
        let mut config = SpectrumConfig::default();
        self.apply_to(&mut config);
        config
    }

    pub fn averaging_config(&self) -> AveragingMode {
        Self::combine_averaging(
            self.averaging_mode,
            self.averaging_factor,
            self.peak_hold_decay,
        )
    }

    pub fn normalized_factor(value: f32) -> f32 {
        EXPONENTIAL_RANGE.snap(value)
    }

    pub fn normalized_decay(value: f32) -> f32 {
        PEAK_HOLD_DECAY_RANGE.snap(value).max(0.0)
    }

    pub fn split_averaging(averaging: AveragingMode) -> (SpectrumAveragingMode, f32, f32) {
        match averaging {
            AveragingMode::None => (
                SpectrumAveragingMode::None,
                DEFAULT_SPECTRUM_AVERAGING_FACTOR,
                DEFAULT_SPECTRUM_PEAK_HOLD_DECAY,
            ),
            AveragingMode::Exponential { factor } => (
                SpectrumAveragingMode::Exponential,
                Self::normalized_factor(factor),
                DEFAULT_SPECTRUM_PEAK_HOLD_DECAY,
            ),
            AveragingMode::PeakHold { decay_per_second } => (
                SpectrumAveragingMode::PeakHold,
                DEFAULT_SPECTRUM_AVERAGING_FACTOR,
                Self::normalized_decay(decay_per_second),
            ),
        }
    }

    pub fn combine_averaging(
        mode: SpectrumAveragingMode,
        factor: f32,
        decay: f32,
    ) -> AveragingMode {
        match mode {
            SpectrumAveragingMode::None => AveragingMode::None,
            SpectrumAveragingMode::Exponential => AveragingMode::Exponential {
                factor: Self::normalized_factor(factor),
            },
            SpectrumAveragingMode::PeakHold => AveragingMode::PeakHold {
                decay_per_second: Self::normalized_decay(decay),
            },
        }
    }
}
