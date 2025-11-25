use super::palette::{PaletteEditor, PaletteEvent};
use super::widgets::{SliderRange, labeled_slider};
use super::{ModuleSettingsPane, SettingsMessage};
use crate::dsp::waveform::{
    DownsampleStrategy, MAX_SCROLL_SPEED, MIN_SCROLL_SPEED, WaveformConfig,
};
use crate::ui::settings::{ModuleSettings, PaletteSettings, SettingsHandle, WaveformSettings};
use crate::ui::theme;
use crate::ui::visualization::visual_manager::{VisualId, VisualKind, VisualManagerHandle};
use iced::Element;
use iced::widget::{column, pick_list, text};
use std::fmt;

#[derive(Debug)]
pub struct WaveformSettingsPane {
    visual_id: VisualId,
    config: WaveformConfig,
    palette: PaletteEditor,
}

impl fmt::Display for DownsampleStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            DownsampleStrategy::MinMax => "Min/Max",
            DownsampleStrategy::Average => "Average",
        })
    }
}

#[derive(Debug, Clone)]
pub enum Message {
    ScrollSpeed(f32),
    Downsample(DownsampleStrategy),
    Palette(PaletteEvent),
}

pub fn create(visual_id: VisualId, visual_manager: &VisualManagerHandle) -> WaveformSettingsPane {
    let stored = visual_manager
        .borrow()
        .module_settings(VisualKind::WAVEFORM)
        .and_then(|s| s.config::<WaveformSettings>());

    let config = stored.as_ref().map(|s| s.to_config()).unwrap_or_default();
    let palette = stored
        .as_ref()
        .and_then(|s| s.palette_array::<{ theme::DEFAULT_WAVEFORM_PALETTE.len() }>())
        .unwrap_or(theme::DEFAULT_WAVEFORM_PALETTE);

    WaveformSettingsPane {
        visual_id,
        config,
        palette: PaletteEditor::new(&palette, &theme::DEFAULT_WAVEFORM_PALETTE),
    }
}

impl ModuleSettingsPane for WaveformSettingsPane {
    fn visual_id(&self) -> VisualId {
        self.visual_id
    }

    fn view(&self) -> Element<'_, SettingsMessage> {
        column![
            labeled_slider(
                "Scroll speed",
                self.config.scroll_speed,
                format!("{:.0} px/s", self.config.scroll_speed),
                SliderRange::new(MIN_SCROLL_SPEED, MAX_SCROLL_SPEED, 1.0),
                |v| SettingsMessage::Waveform(Message::ScrollSpeed(v)),
            ),
            column![
                text("Downsampling strategy"),
                pick_list(
                    [DownsampleStrategy::MinMax, DownsampleStrategy::Average],
                    Some(self.config.downsample),
                    |v| SettingsMessage::Waveform(Message::Downsample(v))
                )
                .text_size(14)
            ]
            .spacing(8),
            column![
                text("Colors").size(14),
                self.palette
                    .view()
                    .map(|e| SettingsMessage::Waveform(Message::Palette(e)))
            ]
            .spacing(8)
        ]
        .spacing(16)
        .into()
    }

    fn handle(&mut self, message: &SettingsMessage, vm: &VisualManagerHandle, s: &SettingsHandle) {
        let SettingsMessage::Waveform(msg) = message else {
            return;
        };

        let changed = match msg {
            Message::ScrollSpeed(v) => {
                let new = v.clamp(MIN_SCROLL_SPEED, MAX_SCROLL_SPEED);
                if self.config.scroll_speed != new {
                    self.config.scroll_speed = new;
                    true
                } else {
                    false
                }
            }
            Message::Downsample(d) => {
                if self.config.downsample != *d {
                    self.config.downsample = *d;
                    true
                } else {
                    false
                }
            }
            Message::Palette(e) => self.palette.update(*e),
        };

        if changed {
            let mut stored = WaveformSettings::from_config(&self.config);
            stored.palette = PaletteSettings::maybe_from_colors(
                self.palette.colors(),
                &theme::DEFAULT_WAVEFORM_PALETTE,
            );

            if vm
                .borrow_mut()
                .apply_module_settings(VisualKind::WAVEFORM, &ModuleSettings::with_config(&stored))
            {
                s.update(|m| m.set_module_config(VisualKind::WAVEFORM, &stored));
            }
        }
    }
}
