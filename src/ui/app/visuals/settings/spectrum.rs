use super::widgets::{SliderRange, labeled_pick_list, labeled_slider, set_f32};
use super::{ModuleSettingsPane, SettingsMessage};
use crate::dsp::spectrogram::FrequencyScale;
use crate::dsp::spectrum::{AveragingMode, SpectrumConfig};
use crate::ui::settings::{ModuleSettings, SettingsHandle};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum SpectrumAveragingMode {
    None,
    #[default]
    Exponential,
    PeakHold,
}

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
    let (averaging_mode, averaging_factor, peak_hold_decay) = split_averaging(config.averaging);

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
                let snapped = normalize_factor(*value);
                if set_f32(&mut self.averaging_factor, snapped) {
                    self.sync_averaging_config();
                    true
                } else {
                    false
                }
            }
            Message::PeakHoldDecay(value) => {
                let snapped = normalize_decay(*value);
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
        let factor = normalize_factor(self.averaging_factor);
        if factor.to_bits() != self.averaging_factor.to_bits() {
            self.averaging_factor = factor;
        }

        let decay = normalize_decay(self.peak_hold_decay);
        if decay.to_bits() != self.peak_hold_decay.to_bits() {
            self.peak_hold_decay = decay;
        }

        self.config.averaging = combine_averaging(
            self.averaging_mode,
            self.averaging_factor,
            self.peak_hold_decay,
        );
    }
}

fn normalize_factor(value: f32) -> f32 {
    let snapped = EXPONENTIAL_RANGE.snap(value);
    AveragingMode::clamp_factor(snapped)
}

fn normalize_decay(value: f32) -> f32 {
    let snapped = PEAK_HOLD_DECAY_RANGE.snap(value);
    AveragingMode::clamp_decay(snapped)
}

fn split_averaging(averaging: AveragingMode) -> (SpectrumAveragingMode, f32, f32) {
    match averaging.normalized() {
        AveragingMode::None => (
            SpectrumAveragingMode::None,
            default_averaging_factor(),
            default_peak_hold_decay(),
        ),
        AveragingMode::Exponential { factor } => (
            SpectrumAveragingMode::Exponential,
            normalize_factor(factor),
            default_peak_hold_decay(),
        ),
        AveragingMode::PeakHold { decay_per_second } => (
            SpectrumAveragingMode::PeakHold,
            default_averaging_factor(),
            normalize_decay(decay_per_second),
        ),
    }
}

fn combine_averaging(mode: SpectrumAveragingMode, factor: f32, decay: f32) -> AveragingMode {
    let mode = match mode {
        SpectrumAveragingMode::None => AveragingMode::None,
        SpectrumAveragingMode::Exponential => AveragingMode::Exponential {
            factor: normalize_factor(factor),
        },
        SpectrumAveragingMode::PeakHold => AveragingMode::PeakHold {
            decay_per_second: normalize_decay(decay),
        },
    };
    mode.normalized()
}

fn default_averaging_factor() -> f32 {
    normalize_factor(AveragingMode::default_exponential_factor())
}

fn default_peak_hold_decay() -> f32 {
    normalize_decay(AveragingMode::default_peak_decay())
}
