use super::palette::{PaletteEditor, PaletteEvent};
use super::widgets::{
    CONTROL_SPACING, LABEL_SIZE, SliderRange, VALUE_GAP, VALUE_SIZE, labeled_pick_list,
    labeled_slider, section_title, set_f32, set_if_changed,
};
use super::{ModuleSettingsPane, SettingsMessage};
use crate::dsp::oscilloscope::TriggerMode;
use crate::ui::settings::{
    ModuleSettings, OscilloscopeChannelMode, OscilloscopeSettings, SettingsHandle,
};
use crate::ui::theme;
use crate::ui::visualization::visual_manager::{VisualId, VisualKind, VisualManagerHandle};
use iced::Element;
use iced::widget::text::Wrapping;
use iced::widget::{column, container, row, slider, text};

#[derive(Debug)]
pub struct OscilloscopeSettingsPane {
    visual_id: VisualId,
    settings: OscilloscopeSettings,
    palette: PaletteEditor,
}

#[derive(Debug, Clone)]
pub enum Message {
    SegmentDuration(f32),
    Persistence(f32),
    TriggerMode(TriggerMode),
    NumCycles(usize),
    ChannelMode(OscilloscopeChannelMode),
    Palette(PaletteEvent),
}

const CHANNEL_OPTIONS: [OscilloscopeChannelMode; 4] = [
    OscilloscopeChannelMode::Both,
    OscilloscopeChannelMode::Left,
    OscilloscopeChannelMode::Right,
    OscilloscopeChannelMode::Mono,
];

const TRIGGER_MODE_OPTIONS: [(&str, TriggerMode); 2] = [
    ("Free-run", TriggerMode::FreeRun),
    ("Stable", TriggerMode::Stable { num_cycles: 1 }),
];

const TRIGGER_MODE_LABELS: [&str; 2] = ["Free-run", "Stable"];

pub fn create(
    visual_id: VisualId,
    visual_manager: &VisualManagerHandle,
) -> OscilloscopeSettingsPane {
    let settings = visual_manager
        .borrow()
        .module_settings(VisualKind::OSCILLOSCOPE)
        .and_then(|stored| stored.config::<OscilloscopeSettings>())
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
        let current_mode_label = match self.settings.trigger_mode {
            TriggerMode::FreeRun => "Free-run",
            TriggerMode::Stable { .. } => "Stable",
        };

        let mut items: Vec<Element<'_, SettingsMessage>> = vec![
            labeled_pick_list(
                "Mode",
                &TRIGGER_MODE_LABELS,
                Some(current_mode_label),
                |label| {
                    let mode = TRIGGER_MODE_OPTIONS
                        .iter()
                        .find(|(l, _)| *l == label)
                        .map(|(_, m)| *m)
                        .unwrap_or(TriggerMode::FreeRun);
                    SettingsMessage::Oscilloscope(Message::TriggerMode(mode))
                },
            )
            .into(),
        ];

        if let TriggerMode::Stable { num_cycles } = self.settings.trigger_mode {
            items.push(
                labeled_slider(
                    "Cycles",
                    num_cycles as f32,
                    format!("{}", num_cycles),
                    SliderRange::new(1.0, 4.0, 1.0),
                    |value| SettingsMessage::Oscilloscope(Message::NumCycles(value as usize)),
                )
                .into(),
            );
        }

        items.extend(vec![
            labeled_pick_list(
                "Channels",
                &CHANNEL_OPTIONS,
                Some(self.settings.channel_mode),
                |mode| SettingsMessage::Oscilloscope(Message::ChannelMode(mode)),
            )
            .into(),
        ]);

        let duration_value = format!("{:.1} ms", self.settings.segment_duration * 1_000.0);
        let segment_duration_widget = match self.settings.trigger_mode {
            TriggerMode::Stable { .. } => column![
                row![
                    container(
                        text("Segment duration")
                            .size(LABEL_SIZE)
                            .wrapping(Wrapping::None)
                    )
                    .clip(true),
                    container(
                        text("(fallback)")
                            .size(VALUE_SIZE - 1)
                            .wrapping(Wrapping::None)
                    )
                    .clip(true),
                    container(text(duration_value).size(VALUE_SIZE)).clip(true),
                ]
                .spacing(VALUE_GAP),
                slider::Slider::new(0.005..=0.1, self.settings.segment_duration, |value| {
                    SettingsMessage::Oscilloscope(Message::SegmentDuration(value))
                })
                .step(0.001)
                .style(theme::slider_style),
            ]
            .spacing(CONTROL_SPACING),
            TriggerMode::FreeRun => labeled_slider(
                "Segment duration",
                self.settings.segment_duration,
                duration_value,
                SliderRange::new(0.005, 0.1, 0.001),
                |value| SettingsMessage::Oscilloscope(Message::SegmentDuration(value)),
            ),
        };

        items.extend(vec![
            segment_duration_widget.into(),
            labeled_slider(
                "Persistence",
                self.settings.persistence,
                format!("{:.2}", self.settings.persistence),
                SliderRange::new(0.0, 1.0, 0.01),
                |value| SettingsMessage::Oscilloscope(Message::Persistence(value)),
            )
            .into(),
        ]);

        items.push(
            column![
                section_title("Color"),
                self.palette
                    .view()
                    .map(|e| SettingsMessage::Oscilloscope(Message::Palette(e)))
            ]
            .spacing(8)
            .into(),
        );

        column(items).spacing(16).into()
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
            Message::Persistence(value) => {
                set_f32(&mut self.settings.persistence, value.clamp(0.0, 1.0))
            }
            Message::TriggerMode(mode) => {
                if self.settings.trigger_mode != *mode {
                    self.settings.trigger_mode = *mode;
                    true
                } else {
                    false
                }
            }
            Message::NumCycles(cycles) => {
                if let TriggerMode::Stable { num_cycles } = self.settings.trigger_mode {
                    let clamped = (*cycles).clamp(1, 4);
                    if num_cycles != clamped {
                        self.settings.trigger_mode = TriggerMode::Stable {
                            num_cycles: clamped,
                        };
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            Message::ChannelMode(mode) => set_if_changed(&mut self.settings.channel_mode, *mode),
            Message::Palette(event) => self.palette.update(*event),
        };

        if changed {
            self.settings.palette = crate::ui::settings::PaletteSettings::maybe_from_colors(
                self.palette.colors(),
                &theme::DEFAULT_OSCILLOSCOPE_PALETTE,
            );

            if visual_manager.borrow_mut().apply_module_settings(
                VisualKind::OSCILLOSCOPE,
                &ModuleSettings::with_config(&self.settings),
            ) {
                settings.update(|mgr| {
                    mgr.set_module_config(VisualKind::OSCILLOSCOPE, &self.settings);
                });
            }
        }
    }
}
