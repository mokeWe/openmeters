use super::palette::{PaletteEditor, PaletteEvent};
use super::widgets::{
    SliderRange, labeled_pick_list, labeled_slider, section_title, set_f32, set_if_changed,
};
use super::{ModuleSettingsPane, SettingsMessage};
use crate::ui::settings::{ModuleSettings, SettingsHandle, StereometerMode, StereometerScale, StereometerSettings};
use crate::ui::theme;
use crate::ui::visualization::visual_manager::{VisualId, VisualKind, VisualManagerHandle};
use iced::widget::{column, row};
use iced::{Element, Length};

const MODE_OPTIONS: [StereometerMode; 2] = [StereometerMode::Lissajous, StereometerMode::DotCloud];
const SCALE_OPTIONS: [StereometerScale; 2] = [StereometerScale::Linear, StereometerScale::Exponential];

#[derive(Debug)]
pub struct StereometerSettingsPane {
    visual_id: VisualId,
    settings: StereometerSettings,
    palette: PaletteEditor,
}

#[derive(Debug, Clone)]
pub enum Message {
    SegmentDuration(f32),
    TargetSampleCount(f32),
    Persistence(f32),
    Rotation(f32),
    Mode(StereometerMode),
    Scale(StereometerScale),
    ScaleRange(f32),
    Palette(PaletteEvent),
}

pub fn create(
    visual_id: VisualId,
    visual_manager: &VisualManagerHandle,
) -> StereometerSettingsPane {
    let settings = visual_manager
        .borrow()
        .module_settings(VisualKind::STEREOMETER)
        .and_then(|stored| stored.config::<StereometerSettings>())
        .unwrap_or_default();

    let palette = settings
        .palette
        .as_ref()
        .and_then(|p| p.to_array::<1>())
        .unwrap_or(theme::DEFAULT_STEREOMETER_PALETTE);

    StereometerSettingsPane {
        visual_id,
        settings,
        palette: PaletteEditor::new(&palette, &theme::DEFAULT_STEREOMETER_PALETTE),
    }
}

impl ModuleSettingsPane for StereometerSettingsPane {
    fn visual_id(&self) -> VisualId {
        self.visual_id
    }

    fn view(&self) -> Element<'_, SettingsMessage> {
        let left_col = column![
            labeled_pick_list("Mode", &MODE_OPTIONS, Some(self.settings.mode), |mode| {
                SettingsMessage::Stereometer(Message::Mode(mode))
            }),
            labeled_slider(
                "Rotation",
                self.settings.rotation as f32,
                format!("{}", self.settings.rotation),
                SliderRange::new(-4.0, 4.0, 1.0),
                |value| SettingsMessage::Stereometer(Message::Rotation(value)),
            ),
        ]
        .spacing(16)
        .width(Length::Fill);

        let mut right_col = column![labeled_pick_list(
            "Scale",
            &SCALE_OPTIONS,
            Some(self.settings.scale),
            |scale| { SettingsMessage::Stereometer(Message::Scale(scale)) }
        ),]
        .spacing(16)
        .width(Length::Fill);

        if self.settings.scale == StereometerScale::Exponential {
            right_col = right_col.push(labeled_slider(
                "Scale range",
                self.settings.scale_range,
                format!("{:.1}", self.settings.scale_range),
                SliderRange::new(1.0, 30.0, 0.5),
                |value| SettingsMessage::Stereometer(Message::ScaleRange(value)),
            ));
        }

        column![
            row![left_col, right_col].spacing(24),
            labeled_slider(
                "Segment duration",
                self.settings.segment_duration,
                format!("{:.1} ms", self.settings.segment_duration * 1_000.0),
                SliderRange::new(0.005, 0.2, 0.001),
                |value| SettingsMessage::Stereometer(Message::SegmentDuration(value)),
            ),
            labeled_slider(
                "Sample count",
                self.settings.target_sample_count as f32,
                format!("{}", self.settings.target_sample_count),
                SliderRange::new(100.0, 2000.0, 50.0),
                |value| SettingsMessage::Stereometer(Message::TargetSampleCount(value)),
            ),
            labeled_slider(
                "Persistence",
                self.settings.persistence,
                format!("{:.2}", self.settings.persistence),
                SliderRange::new(0.0, 1.0, 0.01),
                |value| SettingsMessage::Stereometer(Message::Persistence(value)),
            ),
            column![
                section_title("Colors"),
                self.palette
                    .view()
                    .map(|e| SettingsMessage::Stereometer(Message::Palette(e)))
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
        let SettingsMessage::Stereometer(msg) = message else {
            return;
        };

        let changed = match msg {
            Message::SegmentDuration(value) => {
                set_f32(&mut self.settings.segment_duration, value.clamp(0.005, 0.2))
            }
            Message::TargetSampleCount(value) => {
                let new_count = (value.round() as usize).clamp(100, 2000);
                if self.settings.target_sample_count != new_count {
                    self.settings.target_sample_count = new_count;
                    true
                } else {
                    false
                }
            }
            Message::Persistence(value) => {
                set_f32(&mut self.settings.persistence, value.clamp(0.0, 1.0))
            }
            Message::Rotation(value) => {
                let new_rotation = (value.round() as i8).clamp(-4, 4);
                if self.settings.rotation != new_rotation {
                    self.settings.rotation = new_rotation;
                    true
                } else {
                    false
                }
            }
            Message::Mode(mode) => set_if_changed(&mut self.settings.mode, *mode),
            Message::Scale(scale) => set_if_changed(&mut self.settings.scale, *scale),
            Message::ScaleRange(value) => {
                set_f32(&mut self.settings.scale_range, value.clamp(1.0, 30.0))
            }
            Message::Palette(event) => self.palette.update(*event),
        };

        if changed {
            self.settings.palette = crate::ui::settings::PaletteSettings::maybe_from_colors(
                self.palette.colors(),
                &theme::DEFAULT_STEREOMETER_PALETTE,
            );

            let module_settings = ModuleSettings::with_config(&self.settings);

            if visual_manager
                .borrow_mut()
                .apply_module_settings(VisualKind::STEREOMETER, &module_settings)
            {
                settings.update(|mgr| {
                    mgr.set_module_config(VisualKind::STEREOMETER, &self.settings);
                });
            }
        }
    }
}
