use super::{ModuleSettingsPane, SettingsMessage};
use crate::dsp::waveform::{
    DownsampleStrategy, MAX_SCROLL_SPEED, MIN_SCROLL_SPEED, WaveformConfig,
};
use crate::ui::settings::{ModuleSettings, SettingsHandle, WaveformSettings};
use crate::ui::visualization::visual_manager::{VisualId, VisualKind, VisualManagerHandle};
use iced::Element;
use iced::widget::{column, pick_list, row, slider, text};
use std::fmt;

const SCROLL_SPEED_RANGE: (f32, f32, f32) = (MIN_SCROLL_SPEED, MAX_SCROLL_SPEED, 1.0);

#[derive(Debug)]
pub struct WaveformSettingsPane {
    visual_id: VisualId,
    config: WaveformConfig,
}

impl fmt::Display for DownsampleStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            DownsampleStrategy::MinMax => "Min/Max",
            DownsampleStrategy::Average => "Average",
        };
        write!(f, "{}", label)
    }
}

#[derive(Debug, Clone)]
pub enum Message {
    ScrollSpeed(f32),
    Downsample(DownsampleStrategy),
}

pub fn create(visual_id: VisualId, visual_manager: &VisualManagerHandle) -> WaveformSettingsPane {
    let config = visual_manager
        .borrow()
        .module_settings(VisualKind::WAVEFORM)
        .and_then(|stored| {
            stored.waveform().map(|settings| {
                let mut config = WaveformConfig::default();
                settings.apply_to(&mut config);
                config
            })
        })
        .unwrap_or_default();

    WaveformSettingsPane { visual_id, config }
}

fn labeled_slider<'a>(
    label: &'static str,
    value: f32,
    format: String,
    range: (f32, f32, f32),
    on_change: impl Fn(f32) -> SettingsMessage + 'a,
) -> iced::widget::Column<'a, SettingsMessage> {
    let (min, max, step) = range;
    column![
        row![text(label), text(format).size(12)].spacing(8),
        slider::Slider::new(min..=max, value, on_change).step(step),
    ]
    .spacing(8)
}

impl ModuleSettingsPane for WaveformSettingsPane {
    fn visual_id(&self) -> VisualId {
        self.visual_id
    }

    fn view(&self) -> Element<'_, SettingsMessage> {
        let scroll_speed = labeled_slider(
            "Scroll speed",
            self.config.scroll_speed,
            format!("{:.0} px/s", self.config.scroll_speed),
            SCROLL_SPEED_RANGE,
            |value| SettingsMessage::Waveform(Message::ScrollSpeed(value)),
        );

        let strategies = [DownsampleStrategy::MinMax, DownsampleStrategy::Average];
        let downsample = column![
            text("Downsampling strategy"),
            pick_list(strategies.to_vec(), Some(self.config.downsample), |value| {
                SettingsMessage::Waveform(Message::Downsample(value))
            })
            .text_size(14)
        ]
        .spacing(8);

        column![scroll_speed, downsample].spacing(16).into()
    }

    fn handle(
        &mut self,
        message: &SettingsMessage,
        visual_manager: &VisualManagerHandle,
        settings: &SettingsHandle,
    ) {
        let SettingsMessage::Waveform(msg) = message else {
            return;
        };

        let changed = match msg {
            Message::ScrollSpeed(value) => update_if_changed(
                &mut self.config.scroll_speed,
                value.clamp(MIN_SCROLL_SPEED, MAX_SCROLL_SPEED),
            ),
            Message::Downsample(strategy) => {
                update_if_changed(&mut self.config.downsample, *strategy)
            }
        };

        if changed {
            apply_waveform_config(&self.config, visual_manager, settings);
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

fn apply_waveform_config(
    config: &WaveformConfig,
    visual_manager: &VisualManagerHandle,
    settings: &SettingsHandle,
) {
    let mut module_settings = ModuleSettings::default();
    module_settings.set_waveform(WaveformSettings::from_config(config));

    if visual_manager
        .borrow_mut()
        .apply_module_settings(VisualKind::WAVEFORM, &module_settings)
    {
        settings.update(|mgr| {
            mgr.set_waveform_settings(VisualKind::WAVEFORM, config);
        });
    }
}
