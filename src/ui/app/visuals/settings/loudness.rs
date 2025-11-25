use super::widgets::{CONTROL_SPACING, labeled_pick_list};
use super::{ModuleSettingsPane, SettingsMessage};
use crate::ui::settings::{LoudnessSettings, ModuleSettings, SettingsHandle};
use crate::ui::visualization::loudness::MeterMode;
use crate::ui::visualization::visual_manager::{VisualId, VisualKind, VisualManagerHandle};
use iced::Element;
use iced::widget::column;

#[derive(Debug)]
pub struct LoudnessSettingsPane {
    visual_id: VisualId,
    settings: LoudnessSettings,
}

#[derive(Debug, Clone)]
pub enum Message {
    Mode(MeterSide, MeterMode),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeterSide {
    Left,
    Right,
}

pub fn create(visual_id: VisualId, visual_manager: &VisualManagerHandle) -> LoudnessSettingsPane {
    let settings = visual_manager
        .borrow()
        .module_settings(VisualKind::LOUDNESS)
        .and_then(|stored| stored.config::<LoudnessSettings>())
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
        let left_mode = labeled_pick_list(
            "Left meter mode",
            MeterMode::ALL,
            Some(self.settings.left_mode),
            |mode| SettingsMessage::Loudness(Message::Mode(MeterSide::Left, mode)),
        )
        .spacing(CONTROL_SPACING);

        let right_mode = labeled_pick_list(
            "Right meter mode",
            MeterMode::ALL,
            Some(self.settings.right_mode),
            |mode| SettingsMessage::Loudness(Message::Mode(MeterSide::Right, mode)),
        )
        .spacing(CONTROL_SPACING);

        column![left_mode, right_mode].spacing(16).into()
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
            Message::Mode(side, mode) => self.set_meter_mode(*side, *mode),
        };

        if changed {
            self.push_changes(visual_manager, settings);
        }
    }
}

impl LoudnessSettingsPane {
    fn set_meter_mode(&mut self, side: MeterSide, mode: MeterMode) -> bool {
        let target = match side {
            MeterSide::Left => &mut self.settings.left_mode,
            MeterSide::Right => &mut self.settings.right_mode,
        };

        if *target == mode {
            return false;
        }

        *target = mode;
        true
    }

    fn push_changes(&self, visual_manager: &VisualManagerHandle, settings_handle: &SettingsHandle) {
        let module_settings = ModuleSettings::with_config(&self.settings);

        if visual_manager
            .borrow_mut()
            .apply_module_settings(VisualKind::LOUDNESS, &module_settings)
        {
            settings_handle.update(|mgr| {
                mgr.set_module_config(VisualKind::LOUDNESS, &self.settings);
            });
        }
    }
}
