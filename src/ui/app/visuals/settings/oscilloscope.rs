use super::{ModuleSettingsPane, SettingsMessage};
use crate::dsp::oscilloscope::OscilloscopeConfig;
use crate::ui::settings::{ModuleSettings, OscilloscopeSettings, SettingsHandle};
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

        let changed = match msg {
            Message::SegmentDuration(value) => update_if_changed(
                &mut self.config.segment_duration,
                value.clamp(SEGMENT_MIN, SEGMENT_MAX),
            ),
            Message::TriggerLevel(value) => update_if_changed(
                &mut self.config.trigger_level,
                value.clamp(TRIGGER_MIN, TRIGGER_MAX),
            ),
            Message::Persistence(value) => update_if_changed(
                &mut self.config.persistence,
                value.clamp(PERSISTENCE_MIN, PERSISTENCE_MAX),
            ),
            Message::TriggerMode(rising) => {
                update_if_changed(&mut self.config.trigger_rising, *rising)
            }
        };

        if changed {
            apply_oscilloscope_config(&self.config, visual_manager, settings);
        }
    }
}

#[inline]
fn update_if_changed<T: PartialEq + Copy>(target: &mut T, new_value: T) -> bool {
    if *target != new_value {
        *target = new_value;
        true
    } else {
        false
    }
}

fn apply_oscilloscope_config(
    config: &OscilloscopeConfig,
    visual_manager: &VisualManagerHandle,
    settings: &SettingsHandle,
) {
    let mut module_settings = ModuleSettings::default();
    module_settings.set_oscilloscope(OscilloscopeSettings::from_config(config));

    if visual_manager
        .borrow_mut()
        .apply_module_settings(VisualKind::OSCILLOSCOPE, &module_settings)
    {
        settings.update(|mgr| {
            mgr.set_oscilloscope_settings(VisualKind::OSCILLOSCOPE, config);
        });
    }
}
