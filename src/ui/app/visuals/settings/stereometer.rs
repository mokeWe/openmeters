use super::palette::{PaletteEditor, PaletteEvent};
use super::widgets::{
    SliderRange, labeled_pick_list, labeled_slider, section_title, set_f32, set_if_changed,
};
use super::{ModuleSettingsPane, SettingsMessage};
use crate::ui::settings::{
    ModuleSettings, PaletteSettings, SettingsHandle, StereometerMode, StereometerScale,
    StereometerSettings,
};
use crate::ui::theme;
use crate::ui::visualization::visual_manager::{VisualId, VisualKind, VisualManagerHandle};
use iced::widget::{column, row, toggler};
use iced::{Element, Length};

const MODE_OPTIONS: [StereometerMode; 2] = [StereometerMode::Lissajous, StereometerMode::DotCloud];
const SCALE_OPTIONS: [StereometerScale; 2] =
    [StereometerScale::Linear, StereometerScale::Exponential];

#[inline]
fn st(m: Message) -> SettingsMessage {
    SettingsMessage::Stereometer(m)
}

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
    Flip(bool),
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
        .and_then(|s| s.config::<StereometerSettings>())
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
        let s = &self.settings;
        let left = column![
            labeled_pick_list("Mode", &MODE_OPTIONS, Some(s.mode), |m| st(Message::Mode(
                m
            ))),
            labeled_slider(
                "Rotation",
                s.rotation as f32,
                s.rotation.to_string(),
                SliderRange::new(-4.0, 4.0, 1.0),
                |v| st(Message::Rotation(v))
            ),
            toggler(s.flip)
                .label("Flip")
                .on_toggle(|v| st(Message::Flip(v))),
        ]
        .spacing(16)
        .width(Length::Fill);

        let mut right = column![labeled_pick_list(
            "Scale",
            &SCALE_OPTIONS,
            Some(s.scale),
            |v| st(Message::Scale(v))
        )]
        .spacing(16)
        .width(Length::Fill);
        if s.scale == StereometerScale::Exponential {
            right = right.push(labeled_slider(
                "Scale range",
                s.scale_range,
                format!("{:.1}", s.scale_range),
                SliderRange::new(1.0, 30.0, 0.5),
                |v| st(Message::ScaleRange(v)),
            ));
        }

        column![
            row![left, right].spacing(24),
            labeled_slider(
                "Segment duration",
                s.segment_duration,
                format!("{:.1} ms", s.segment_duration * 1000.0),
                SliderRange::new(0.005, 0.2, 0.001),
                |v| st(Message::SegmentDuration(v))
            ),
            labeled_slider(
                "Sample count",
                s.target_sample_count as f32,
                s.target_sample_count.to_string(),
                SliderRange::new(100.0, 2000.0, 50.0),
                |v| st(Message::TargetSampleCount(v))
            ),
            labeled_slider(
                "Persistence",
                s.persistence,
                format!("{:.2}", s.persistence),
                SliderRange::new(0.0, 1.0, 0.01),
                |v| st(Message::Persistence(v))
            ),
            column![
                section_title("Colors"),
                self.palette.view().map(|e| st(Message::Palette(e)))
            ]
            .spacing(8)
        ]
        .spacing(16)
        .into()
    }

    fn handle(
        &mut self,
        message: &SettingsMessage,
        vm: &VisualManagerHandle,
        settings: &SettingsHandle,
    ) {
        let SettingsMessage::Stereometer(msg) = message else {
            return;
        };
        let s = &mut self.settings;
        let changed = match msg {
            Message::SegmentDuration(v) => set_f32(&mut s.segment_duration, v.clamp(0.005, 0.2)),
            Message::TargetSampleCount(v) => set_if_changed(
                &mut s.target_sample_count,
                (v.round() as usize).clamp(100, 2000),
            ),
            Message::Persistence(v) => set_f32(&mut s.persistence, v.clamp(0.0, 1.0)),
            Message::Rotation(v) => set_if_changed(&mut s.rotation, (v.round() as i8).clamp(-4, 4)),
            Message::Flip(v) => set_if_changed(&mut s.flip, *v),
            Message::Mode(m) => set_if_changed(&mut s.mode, *m),
            Message::Scale(sc) => set_if_changed(&mut s.scale, *sc),
            Message::ScaleRange(v) => set_f32(&mut s.scale_range, v.clamp(1.0, 30.0)),
            Message::Palette(e) => self.palette.update(*e),
        };
        if changed {
            self.settings.palette = PaletteSettings::maybe_from_colors(
                self.palette.colors(),
                &theme::DEFAULT_STEREOMETER_PALETTE,
            );
            if vm.borrow_mut().apply_module_settings(
                VisualKind::STEREOMETER,
                &ModuleSettings::with_config(&self.settings),
            ) {
                settings.update(|m| m.set_module_config(VisualKind::STEREOMETER, &self.settings));
            }
        }
    }
}
