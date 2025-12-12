use super::palette::{PaletteEditor, PaletteEvent};
use super::widgets::{
    SliderRange, labeled_pick_list, labeled_slider, section_title, set_f32, set_if_changed,
};
use super::{ModuleSettingsPane, SettingsMessage};
use crate::dsp::oscilloscope::TriggerMode;
use crate::ui::settings::{
    ModuleSettings, OscilloscopeChannelMode as ChannelMode, OscilloscopeSettings, PaletteSettings,
    SettingsHandle,
};
use crate::ui::theme;
use crate::ui::visualization::visual_manager::{VisualId, VisualKind, VisualManagerHandle};
use iced::Element;
use iced::widget::column;

#[derive(Debug)]
pub struct OscilloscopeSettingsPane {
    visual_id: VisualId,
    settings: OscilloscopeSettings,
    palette: PaletteEditor,
}

#[derive(Debug, Clone, Copy)]
pub enum Message {
    SegmentDuration(f32),
    Persistence(f32),
    TriggerMode(TriggerMode),
    NumCycles(usize),
    ChannelMode(ChannelMode),
    Palette(PaletteEvent),
}

#[inline]
fn osc(message: Message) -> SettingsMessage {
    SettingsMessage::Oscilloscope(message)
}

pub fn create(
    visual_id: VisualId,
    visual_manager: &VisualManagerHandle,
) -> OscilloscopeSettingsPane {
    let settings = visual_manager
        .borrow()
        .module_settings(VisualKind::OSCILLOSCOPE)
        .and_then(|s| s.config::<OscilloscopeSettings>())
        .unwrap_or_default();
    let palette = settings
        .palette
        .as_ref()
        .and_then(|p| p.to_array::<1>())
        .unwrap_or(theme::DEFAULT_OSCILLOSCOPE_PALETTE);
    OscilloscopeSettingsPane {
        visual_id,
        settings,
        palette: PaletteEditor::new(&palette, &theme::DEFAULT_OSCILLOSCOPE_PALETTE),
    }
}

impl ModuleSettingsPane for OscilloscopeSettingsPane {
    fn visual_id(&self) -> VisualId {
        self.visual_id
    }

    fn view(&self) -> Element<'_, SettingsMessage> {
        let mode = self.settings.trigger_mode;
        let is_stable = matches!(mode, TriggerMode::Stable { .. });
        let dur_label = if is_stable {
            "Segment duration (fallback)"
        } else {
            "Segment duration"
        };

        let mut content = column![
            labeled_pick_list(
                "Mode",
                &["Free-run", "Stable"],
                Some(if is_stable { "Stable" } else { "Free-run" }),
                |l| osc(Message::TriggerMode(if l == "Stable" {
                    TriggerMode::Stable { num_cycles: 1 }
                } else {
                    TriggerMode::FreeRun
                }))
            ),
            labeled_pick_list(
                "Channels",
                &[
                    ChannelMode::Both,
                    ChannelMode::Left,
                    ChannelMode::Right,
                    ChannelMode::Mono
                ],
                Some(self.settings.channel_mode),
                |m| osc(Message::ChannelMode(m))
            ),
        ]
        .spacing(16);

        if let TriggerMode::Stable { num_cycles } = mode {
            content = content.push(labeled_slider(
                "Cycles",
                num_cycles as f32,
                num_cycles.to_string(),
                SliderRange::new(1.0, 4.0, 1.0),
                |v| osc(Message::NumCycles(v as usize)),
            ));
        }

        content
            .push(labeled_slider(
                dur_label,
                self.settings.segment_duration,
                format!("{:.1} ms", self.settings.segment_duration * 1000.0),
                SliderRange::new(0.005, 0.1, 0.001),
                |v| osc(Message::SegmentDuration(v)),
            ))
            .push(labeled_slider(
                "Persistence",
                self.settings.persistence,
                format!("{:.2}", self.settings.persistence),
                SliderRange::new(0.0, 1.0, 0.01),
                |v| osc(Message::Persistence(v)),
            ))
            .push(
                column![
                    section_title("Color"),
                    self.palette.view().map(|e| osc(Message::Palette(e)))
                ]
                .spacing(8),
            )
            .into()
    }

    fn handle(
        &mut self,
        message: &SettingsMessage,
        vm: &VisualManagerHandle,
        settings: &SettingsHandle,
    ) {
        let SettingsMessage::Oscilloscope(msg) = message else {
            return;
        };
        let changed = match *msg {
            Message::SegmentDuration(v) => {
                set_f32(&mut self.settings.segment_duration, v.clamp(0.005, 0.1))
            }
            Message::Persistence(v) => set_f32(&mut self.settings.persistence, v.clamp(0.0, 1.0)),
            Message::TriggerMode(m) => set_if_changed(&mut self.settings.trigger_mode, m),
            Message::NumCycles(c) => match self.settings.trigger_mode {
                TriggerMode::Stable { .. } => set_if_changed(
                    &mut self.settings.trigger_mode,
                    TriggerMode::Stable {
                        num_cycles: c.clamp(1, 4),
                    },
                ),
                TriggerMode::FreeRun => false,
            },
            Message::ChannelMode(m) => set_if_changed(&mut self.settings.channel_mode, m),
            Message::Palette(e) => self.palette.update(e),
        };
        if changed {
            self.settings.palette = PaletteSettings::maybe_from_colors(
                self.palette.colors(),
                &theme::DEFAULT_OSCILLOSCOPE_PALETTE,
            );
            if vm.borrow_mut().apply_module_settings(
                VisualKind::OSCILLOSCOPE,
                &ModuleSettings::with_config(&self.settings),
            ) {
                settings.update(|m| m.set_module_config(VisualKind::OSCILLOSCOPE, &self.settings));
            }
        }
    }
}
