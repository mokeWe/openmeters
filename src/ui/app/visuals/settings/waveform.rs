use super::{ModuleSettingsPane, SettingsMessage};
use crate::dsp::waveform::{MAX_SCROLL_SPEED, MIN_SCROLL_SPEED, WaveformConfig};
use crate::ui::settings::{ModuleSettings, SettingsHandle, WaveformSettings};
use crate::ui::visualization::visual_manager::{VisualId, VisualKind, VisualManagerHandle};
use iced::Element;
use iced::widget::{column, row, slider, text};

const SCROLL_SPEED_MIN: f32 = MIN_SCROLL_SPEED;
const SCROLL_SPEED_MAX: f32 = MAX_SCROLL_SPEED;

#[derive(Debug)]
pub struct WaveformSettingsPane {
    visual_id: VisualId,
    config: WaveformConfig,
}

#[derive(Debug, Clone)]
pub enum Message {
    ScrollSpeed(f32),
}

pub fn create(visual_id: VisualId, visual_manager: &VisualManagerHandle) -> WaveformSettingsPane {
    let config = visual_manager
        .borrow()
        .module_settings(VisualKind::WAVEFORM)
        .and_then(|stored| stored.waveform().cloned())
        .map_or_else(WaveformConfig::default, |stored| {
            let mut config = WaveformConfig::default();
            stored.apply_to(&mut config);
            config
        });

    WaveformSettingsPane { visual_id, config }
}

impl ModuleSettingsPane for WaveformSettingsPane {
    fn visual_id(&self) -> VisualId {
        self.visual_id
    }

    fn view(&self) -> Element<'_, SettingsMessage> {
        let scroll_speed = column![
            row![
                text("Scroll speed"),
                text(format!("{:.0} px/s", self.config.scroll_speed)).size(12)
            ]
            .spacing(8),
            slider::Slider::new(
                SCROLL_SPEED_MIN..=SCROLL_SPEED_MAX,
                self.config.scroll_speed,
                |value| SettingsMessage::Waveform(Message::ScrollSpeed(value)),
            )
            .step(1.0)
        ]
        .spacing(8);

        column![scroll_speed].spacing(16).into()
    }

    fn handle(
        &mut self,
        message: &SettingsMessage,
        visual_manager: &VisualManagerHandle,
        settings: &SettingsHandle,
    ) {
        let SettingsMessage::Waveform(msg) = message else {
            return;
        };

        let mut changed = false;
        match msg {
            Message::ScrollSpeed(value) => {
                let clamped = value.clamp(SCROLL_SPEED_MIN, SCROLL_SPEED_MAX);
                if (self.config.scroll_speed - clamped).abs() > f32::EPSILON {
                    self.config.scroll_speed = clamped;
                    changed = true;
                }
            }
        }

        if changed {
            let mut module_settings = ModuleSettings::default();
            module_settings.set_waveform(WaveformSettings::from_config(&self.config));

            if visual_manager
                .borrow_mut()
                .apply_module_settings(VisualKind::WAVEFORM, &module_settings)
            {
                settings.update(|settings| {
                    settings.set_waveform_settings(VisualKind::WAVEFORM, &self.config)
                });
            }
        }
    }
}
