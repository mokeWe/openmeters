use super::{ModuleSettingsPane, SettingsMessage};
use crate::ui::settings::{LoudnessSettings, ModuleSettings, SettingsHandle};
use crate::ui::visualization::loudness::MeterMode;
use crate::ui::visualization::visual_manager::{VisualId, VisualKind, VisualManagerHandle};
use iced::widget::{column, pick_list, text};
use iced::{Element, Length};

#[derive(Debug)]
pub struct LoudnessSettingsPane {
    visual_id: VisualId,
    settings: LoudnessSettings,
}

#[derive(Debug, Clone)]
pub enum Message {
    LeftMode(MeterMode),
    RightMode(MeterMode),
}

pub fn create(visual_id: VisualId, visual_manager: &VisualManagerHandle) -> LoudnessSettingsPane {
    let settings = visual_manager
        .borrow()
        .module_settings(VisualKind::LOUDNESS)
        .and_then(|stored| stored.loudness().copied())
        .unwrap_or_default();

    LoudnessSettingsPane {
        visual_id,
        settings,
    }
}

impl ModuleSettingsPane for LoudnessSettingsPane {
    fn visual_id(&self) -> VisualId {
        self.visual_id
    }

    fn view(&self) -> Element<'_, SettingsMessage> {
        column![
            column![
                text("Left meter mode"),
                pick_list(MeterMode::ALL, Some(self.settings.left_mode), |mode| {
                    SettingsMessage::Loudness(Message::LeftMode(mode))
                })
            ]
            .spacing(8),
            column![
                text("Right meter mode"),
                pick_list(MeterMode::ALL, Some(self.settings.right_mode), |mode| {
                    SettingsMessage::Loudness(Message::RightMode(mode))
                })
            ]
            .spacing(8),
        ]
        .spacing(16)
        .width(Length::Fill)
        .into()
    }

    fn handle(
        &mut self,
        message: &SettingsMessage,
        visual_manager: &VisualManagerHandle,
        settings: &SettingsHandle,
    ) {
        let SettingsMessage::Loudness(msg) = message else {
            return;
        };

        let changed = match msg {
            Message::LeftMode(mode) => {
                if self.settings.left_mode != *mode {
                    self.settings.left_mode = *mode;
                    true
                } else {
                    false
                }
            }
            Message::RightMode(mode) => {
                if self.settings.right_mode != *mode {
                    self.settings.right_mode = *mode;
                    true
                } else {
                    false
                }
            }
        };

        if changed {
            self.push_changes(visual_manager, settings);
        }
    }
}

impl LoudnessSettingsPane {
    fn push_changes(&self, visual_manager: &VisualManagerHandle, settings_handle: &SettingsHandle) {
        let module_settings = ModuleSettings::with_loudness_settings(self.settings);

        if visual_manager
            .borrow_mut()
            .apply_module_settings(VisualKind::LOUDNESS, &module_settings)
        {
            let updated = self.settings;
            settings_handle.update(|mgr| {
                mgr.set_loudness_settings(VisualKind::LOUDNESS, updated);
            });
        }
    }
}
