use super::{ModuleSettingsPane, SettingsMessage};
use crate::dsp::spectrum::{AveragingMode, SpectrumConfig};
use crate::ui::settings::{ModuleSettings, SettingsHandle, SpectrumSettings};
use crate::ui::visualization::visual_manager::{VisualId, VisualKind, VisualManagerHandle};
use iced::Element;
use iced::widget::{column, pick_list, row, slider, text};

const AVERAGING_MIN: f32 = 0.0;
const AVERAGING_MAX: f32 = 0.95;
const FFT_OPTIONS: [usize; 4] = [1024, 2048, 4096, 8192];

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
}

pub fn create(visual_id: VisualId, visual_manager: &VisualManagerHandle) -> SpectrumSettingsPane {
    let config = visual_manager
        .borrow()
        .module_settings(VisualKind::SPECTRUM)
        .and_then(|stored| stored.spectrum().cloned())
        .map_or_else(SpectrumConfig::default, |stored| {
            let mut config = SpectrumConfig::default();
            stored.apply_to(&mut config);
            config
        });

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
        let fft_pick = pick_list(FFT_OPTIONS.to_vec(), Some(self.config.fft_size), |size| {
            SettingsMessage::Spectrum(Message::FftSize(size))
        });

        let fft_row = row![text("FFT size"), fft_pick].spacing(12);

        let averaging = column![
            row![
                text("Averaging"),
                text(format!("{:.2}", self.averaging_factor)).size(12)
            ]
            .spacing(8),
            slider::Slider::new(
                AVERAGING_MIN..=AVERAGING_MAX,
                self.averaging_factor,
                |value| SettingsMessage::Spectrum(Message::Averaging(value)),
            )
            .step(0.01)
        ]
        .spacing(8);

        column![fft_row, averaging].spacing(16).into()
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

        let mut changed = false;
        match msg {
            Message::FftSize(size) => {
                if self.config.fft_size != *size {
                    self.config.fft_size = *size;
                    self.config.hop_size = (size / 4).max(1);
                    changed = true;
                }
            }
            Message::Averaging(value) => {
                let clamped = value.clamp(AVERAGING_MIN, AVERAGING_MAX);
                if (self.averaging_factor - clamped).abs() > f32::EPSILON {
                    self.averaging_factor = clamped;
                    self.config.averaging = AveragingMode::Exponential {
                        factor: self.averaging_factor,
                    };
                    changed = true;
                }
            }
        }

        if changed {
            let mut module_settings = ModuleSettings::default();
            module_settings.set_spectrum(SpectrumSettings::from_config(&self.config));

            if visual_manager
                .borrow_mut()
                .apply_module_settings(VisualKind::SPECTRUM, &module_settings)
            {
                settings.update(|settings| {
                    settings.set_spectrum_settings(VisualKind::SPECTRUM, &self.config);
                });
            }
        }
    }
}
