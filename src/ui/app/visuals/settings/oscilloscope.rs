use super::{ModuleSettingsPane, SettingsMessage};
use crate::dsp::oscilloscope::OscilloscopeConfig;
use crate::ui::settings::SettingsHandle;
use crate::ui::visualization::visual_manager::{VisualId, VisualKind, VisualManagerHandle};
use iced::Element;
use iced::widget::{checkbox, column, row, slider, text};

const SEGMENT_MIN: f32 = 0.005;
const SEGMENT_MAX: f32 = 0.1;
const TRIGGER_MIN: f32 = 0.0;
const TRIGGER_MAX: f32 = 1.0;
const PERSISTENCE_MIN: f32 = 0.0;
const PERSISTENCE_MAX: f32 = 1.0;

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
}

pub fn create(
    visual_id: VisualId,
    visual_manager: &VisualManagerHandle,
) -> OscilloscopeSettingsPane {
    let config = visual_manager
        .borrow()
        .oscilloscope_config()
        .unwrap_or_default();

    OscilloscopeSettingsPane { visual_id, config }
}

impl ModuleSettingsPane for OscilloscopeSettingsPane {
    fn visual_id(&self) -> VisualId {
        self.visual_id
    }

    fn view(&self) -> Element<'_, SettingsMessage> {
        let segment = column![
            row![
                text("Segment duration"),
                text(format!("{:.1} ms", self.config.segment_duration * 1_000.0)).size(12)
            ]
            .spacing(8),
            slider::Slider::new(
                SEGMENT_MIN..=SEGMENT_MAX,
                self.config.segment_duration,
                |value| SettingsMessage::Oscilloscope(Message::SegmentDuration(value)),
            )
            .step(0.001)
        ]
        .spacing(8);

        let trigger_level = column![
            row![
                text("Trigger level"),
                text(format!("{:.2}", self.config.trigger_level)).size(12)
            ]
            .spacing(8),
            slider::Slider::new(
                TRIGGER_MIN..=TRIGGER_MAX,
                self.config.trigger_level,
                |value| SettingsMessage::Oscilloscope(Message::TriggerLevel(value)),
            )
            .step(0.01)
        ]
        .spacing(8);

        let persistence = column![
            row![
                text("Persistence"),
                text(format!("{:.2}", self.config.persistence)).size(12)
            ]
            .spacing(8),
            slider::Slider::new(
                PERSISTENCE_MIN..=PERSISTENCE_MAX,
                self.config.persistence,
                |value| SettingsMessage::Oscilloscope(Message::Persistence(value)),
            )
            .step(0.01)
        ]
        .spacing(8);

        let trigger_mode = checkbox("Rising-edge trigger", self.config.trigger_rising)
            .on_toggle(|value| SettingsMessage::Oscilloscope(Message::TriggerMode(value)));

        column![segment, trigger_level, persistence, trigger_mode]
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

        let mut changed = false;
        match msg {
            Message::SegmentDuration(value) => {
                let clamped = value.clamp(SEGMENT_MIN, SEGMENT_MAX);
                if (self.config.segment_duration - clamped).abs() > f32::EPSILON {
                    self.config.segment_duration = clamped;
                    changed = true;
                }
            }
            Message::TriggerLevel(value) => {
                let clamped = value.clamp(TRIGGER_MIN, TRIGGER_MAX);
                if (self.config.trigger_level - clamped).abs() > f32::EPSILON {
                    self.config.trigger_level = clamped;
                    changed = true;
                }
            }
            Message::Persistence(value) => {
                let clamped = value.clamp(PERSISTENCE_MIN, PERSISTENCE_MAX);
                if (self.config.persistence - clamped).abs() > f32::EPSILON {
                    self.config.persistence = clamped;
                    changed = true;
                }
            }
            Message::TriggerMode(rising) => {
                if self.config.trigger_rising != *rising {
                    self.config.trigger_rising = *rising;
                    changed = true;
                }
            }
        }

        if changed
            && visual_manager
                .borrow_mut()
                .set_oscilloscope_config(self.config)
        {
            settings.update(|settings| {
                settings.set_oscilloscope_settings(VisualKind::Oscilloscope, &self.config)
            });
        }
    }
}
