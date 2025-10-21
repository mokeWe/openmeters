use super::widgets::{SliderRange, labeled_pick_list, labeled_slider, set_f32};
use super::{ModuleSettingsPane, SettingsMessage};
use crate::dsp::spectrogram::FrequencyScale;
use crate::dsp::spectrum::{AveragingMode, SpectrumConfig};
use crate::ui::settings::{ModuleSettings, SettingsHandle};
use crate::ui::visualization::visual_manager::{VisualId, VisualKind, VisualManagerHandle};
use iced::Element;
use iced::widget::column;

const AVERAGING_MIN: f32 = 0.0;
const AVERAGING_MAX: f32 = 0.95;
const FFT_OPTIONS: [usize; 4] = [1024, 2048, 4096, 8192];
const SCALE_OPTIONS: [FrequencyScale; 3] = [
    FrequencyScale::Linear,
    FrequencyScale::Logarithmic,
    FrequencyScale::Mel,
];

#[derive(Debug)]
pub struct SpectrumSettingsPane {
    visual_id: VisualId,
    config: SpectrumConfig,
    averaging_factor: f32,
}

#[derive(Debug, Clone)]
pub enum Message {
    FftSize(usize),
    Averaging(f32),
    FrequencyScale(FrequencyScale),
}

pub fn create(visual_id: VisualId, visual_manager: &VisualManagerHandle) -> SpectrumSettingsPane {
    let config = visual_manager
        .borrow()
        .module_settings(VisualKind::SPECTRUM)
        .and_then(|stored| stored.spectrum_config())
        .unwrap_or_default();

    let averaging_factor = match config.averaging {
        AveragingMode::Exponential { factor } => factor,
        _ => 0.5,
    };

    SpectrumSettingsPane {
        visual_id,
        config,
        averaging_factor,
    }
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

        let averaging = labeled_slider(
            "Averaging",
            self.averaging_factor,
            format!("{:.2}", self.averaging_factor),
            SliderRange::new(AVERAGING_MIN, AVERAGING_MAX, 0.01),
            |value| SettingsMessage::Spectrum(Message::Averaging(value)),
        );

        column![fft_row, scale_row, averaging].spacing(16).into()
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
            Message::Averaging(value) => {
                let clamped = value.clamp(AVERAGING_MIN, AVERAGING_MAX);
                if set_f32(&mut self.averaging_factor, clamped) {
                    self.config.averaging = AveragingMode::Exponential {
                        factor: self.averaging_factor,
                    };
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
