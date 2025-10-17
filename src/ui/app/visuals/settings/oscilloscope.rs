use super::{ModuleSettingsPane, SettingsMessage};
use crate::dsp::oscilloscope::{DisplayMode, OscilloscopeConfig};
use crate::ui::settings::{ModuleSettings, OscilloscopeSettings, SettingsHandle};
use crate::ui::visualization::visual_manager::{VisualId, VisualKind, VisualManagerHandle};
use iced::Element;
use iced::widget::{checkbox, column, pick_list, row, slider, text};

#[derive(Debug)]
pub struct OscilloscopeSettingsPane {
    visual_id: VisualId,
    config: OscilloscopeConfig,
}

#[derive(Debug, Clone)]
pub enum Message {
    SegmentDuration(f32),
    TriggerLevel(f32),
    Persistence(f32),
    TriggerMode(bool),
    DisplayMode(DisplayMode),
}

pub fn create(
    visual_id: VisualId,
    visual_manager: &VisualManagerHandle,
) -> OscilloscopeSettingsPane {
    let config = visual_manager
        .borrow()
        .module_settings(VisualKind::OSCILLOSCOPE)
        .and_then(|stored| {
            stored.oscilloscope().map(|settings| {
                let mut config = OscilloscopeConfig::default();
                settings.apply_to(&mut config);
                config
            })
        })
        .unwrap_or_default();

    OscilloscopeSettingsPane { visual_id, config }
}

impl ModuleSettingsPane for OscilloscopeSettingsPane {
    fn visual_id(&self) -> VisualId {
        self.visual_id
    }

    fn view(&self) -> Element<'_, SettingsMessage> {
        column![
            column![
                text("Display mode"),
                pick_list(
                    DisplayMode::ALL.to_vec(),
                    Some(self.config.display_mode),
                    |mode| { SettingsMessage::Oscilloscope(Message::DisplayMode(mode)) }
                )
            ]
            .spacing(8),
            column![
                row![
                    text("Segment duration"),
                    text(format!("{:.1} ms", self.config.segment_duration * 1_000.0)).size(12)
                ]
                .spacing(8),
                slider(0.005..=0.1, self.config.segment_duration, |value| {
                    SettingsMessage::Oscilloscope(Message::SegmentDuration(value))
                })
                .step(0.001)
            ]
            .spacing(8),
            column![
                row![
                    text("Trigger level"),
                    text(format!("{:.2}", self.config.trigger_level)).size(12)
                ]
                .spacing(8),
                slider(0.0..=1.0, self.config.trigger_level, |value| {
                    SettingsMessage::Oscilloscope(Message::TriggerLevel(value))
                })
                .step(0.01)
            ]
            .spacing(8),
            column![
                row![
                    text("Persistence"),
                    text(format!("{:.2}", self.config.persistence)).size(12)
                ]
                .spacing(8),
                slider(0.0..=1.0, self.config.persistence, |value| {
                    SettingsMessage::Oscilloscope(Message::Persistence(value))
                })
                .step(0.01)
            ]
            .spacing(8),
            checkbox("Rising-edge trigger", self.config.trigger_rising)
                .on_toggle(|value| SettingsMessage::Oscilloscope(Message::TriggerMode(value)))
        ]
        .spacing(16)
        .into()
    }

    fn handle(
        &mut self,
        message: &SettingsMessage,
        visual_manager: &VisualManagerHandle,
        settings: &SettingsHandle,
    ) {
        let SettingsMessage::Oscilloscope(msg) = message else {
            return;
        };

        let changed = match msg {
            Message::SegmentDuration(value) => {
                let new = value.clamp(0.005, 0.1);
                if (self.config.segment_duration - new).abs() > f32::EPSILON {
                    self.config.segment_duration = new;
                    true
                } else {
                    false
                }
            }
            Message::TriggerLevel(value) => {
                let new = value.clamp(0.0, 1.0);
                if (self.config.trigger_level - new).abs() > f32::EPSILON {
                    self.config.trigger_level = new;
                    true
                } else {
                    false
                }
            }
            Message::Persistence(value) => {
                let new = value.clamp(0.0, 1.0);
                if (self.config.persistence - new).abs() > f32::EPSILON {
                    self.config.persistence = new;
                    true
                } else {
                    false
                }
            }
            Message::TriggerMode(rising) => {
                if self.config.trigger_rising != *rising {
                    self.config.trigger_rising = *rising;
                    true
                } else {
                    false
                }
            }
            Message::DisplayMode(mode) => {
                if self.config.display_mode != *mode {
                    self.config.display_mode = *mode;
                    true
                } else {
                    false
                }
            }
        };

        if changed {
            let mut module_settings = ModuleSettings::default();
            module_settings.set_oscilloscope(OscilloscopeSettings::from_config(&self.config));

            if visual_manager
                .borrow_mut()
                .apply_module_settings(VisualKind::OSCILLOSCOPE, &module_settings)
            {
                settings.update(|mgr| {
                    mgr.set_oscilloscope_settings(VisualKind::OSCILLOSCOPE, &self.config);
                });
            }
        }
    }
}
