use super::widgets::{SliderRange, labeled_slider, set_f32};
use super::{ModuleSettingsPane, SettingsMessage};
use crate::ui::settings::{ModuleSettings, OscilloscopeSettings, SettingsHandle};
use crate::ui::visualization::oscilloscope::DisplayMode;
use crate::ui::visualization::visual_manager::{VisualId, VisualKind, VisualManagerHandle};
use iced::widget::{column, pick_list, row, slider, text, toggler};
use iced::{Element, Length};

#[derive(Debug)]
pub struct OscilloscopeSettingsPane {
    visual_id: VisualId,
    settings: OscilloscopeSettings,
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
    let settings = visual_manager
        .borrow()
        .module_settings(VisualKind::OSCILLOSCOPE)
        .and_then(|stored| stored.oscilloscope().cloned())
        .unwrap_or_default();

    OscilloscopeSettingsPane {
        visual_id,
        settings,
    }
}

impl ModuleSettingsPane for OscilloscopeSettingsPane {
    fn visual_id(&self) -> VisualId {
        self.visual_id
    }

    fn view(&self) -> Element<'_, SettingsMessage> {
        let trigger_label = if self.settings.trigger_rising {
            "Rising edge"
        } else {
            "Falling edge"
        };

        column![
            column![
                text("Display mode"),
                pick_list(
                    DisplayMode::ALL.to_vec(),
                    Some(self.settings.display_mode),
                    |mode| { SettingsMessage::Oscilloscope(Message::DisplayMode(mode)) }
                )
            ]
            .spacing(8),
            labeled_slider(
                "Segment duration",
                self.settings.segment_duration,
                format!("{:.1} ms", self.settings.segment_duration * 1_000.0),
                SliderRange::new(0.005, 0.1, 0.001),
                |value| SettingsMessage::Oscilloscope(Message::SegmentDuration(value)),
            ),
            labeled_slider(
                "Persistence",
                self.settings.persistence,
                format!("{:.2}", self.settings.persistence),
                SliderRange::new(0.0, 1.0, 0.01),
                |value| SettingsMessage::Oscilloscope(Message::Persistence(value)),
            ),
            column![
                text("Trigger"),
                row![
                    slider(0.0..=1.0, self.settings.trigger_level, |value| {
                        SettingsMessage::Oscilloscope(Message::TriggerLevel(value))
                    })
                    .step(0.01)
                    .width(Length::Fill),
                    toggler(self.settings.trigger_rising)
                        .label(trigger_label)
                        .spacing(4)
                        .text_size(12)
                        .on_toggle(|value| {
                            SettingsMessage::Oscilloscope(Message::TriggerMode(value))
                        })
                ]
                .spacing(12),
                text(format!("Level {:.2}", self.settings.trigger_level)).size(12)
            ]
            .spacing(8)
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
                set_f32(&mut self.settings.segment_duration, value.clamp(0.005, 0.1))
            }
            Message::TriggerLevel(value) => {
                set_f32(&mut self.settings.trigger_level, value.clamp(0.0, 1.0))
            }
            Message::Persistence(value) => {
                set_f32(&mut self.settings.persistence, value.clamp(0.0, 1.0))
            }
            Message::TriggerMode(rising) => {
                if self.settings.trigger_rising != *rising {
                    self.settings.trigger_rising = *rising;
                    true
                } else {
                    false
                }
            }
            Message::DisplayMode(mode) => {
                if self.settings.display_mode != *mode {
                    self.settings.display_mode = *mode;
                    true
                } else {
                    false
                }
            }
        };

        if changed {
            let module_settings = ModuleSettings::with_oscilloscope_settings(&self.settings);

            if visual_manager
                .borrow_mut()
                .apply_module_settings(VisualKind::OSCILLOSCOPE, &module_settings)
            {
                settings.update(|mgr| {
                    mgr.set_oscilloscope_settings(VisualKind::OSCILLOSCOPE, &self.settings);
                });
            }
        }
    }
}
