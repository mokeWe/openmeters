//! Contains the settings panes for visual modules.

mod loudness;
mod oscilloscope;
pub mod palette;
mod spectrogram;
mod spectrum;
mod stereometer;
mod waveform;
mod widgets;

use crate::ui::settings::SettingsHandle;
use crate::ui::visualization::visual_manager::{VisualId, VisualKind, VisualManagerHandle};
use iced::Element;

#[derive(Debug, Clone)]
pub enum SettingsMessage {
    Loudness(loudness::Message),
    Oscilloscope(oscilloscope::Message),
    Spectrogram(spectrogram::Message),
    Spectrum(spectrum::Message),
    Stereometer(stereometer::Message),
    Waveform(waveform::Message),
}

pub trait ModuleSettingsPane: std::fmt::Debug + 'static {
    fn visual_id(&self) -> VisualId;
    fn view(&self) -> Element<'_, SettingsMessage>;
    fn handle(
        &mut self,
        message: &SettingsMessage,
        visual_manager: &VisualManagerHandle,
        settings: &SettingsHandle,
    );
}

#[derive(Debug)]
pub struct ActiveSettings {
    pane: Box<dyn ModuleSettingsPane>,
}

impl ActiveSettings {
    pub fn new(pane: Box<dyn ModuleSettingsPane>) -> Self {
        Self { pane }
    }

    pub fn visual_id(&self) -> VisualId {
        self.pane.visual_id()
    }

    pub fn view(&self) -> Element<'_, SettingsMessage> {
        self.pane.view()
    }

    pub fn handle_message(
        &mut self,
        message: &SettingsMessage,
        visual_manager: &VisualManagerHandle,
        settings: &SettingsHandle,
    ) {
        self.pane.handle(message, visual_manager, settings);
    }
}

pub fn create_panel(
    visual_id: VisualId,
    kind: VisualKind,
    visual_manager: &VisualManagerHandle,
) -> ActiveSettings {
    let pane: Box<dyn ModuleSettingsPane> = match kind {
        VisualKind::LOUDNESS => Box::new(loudness::create(visual_id, visual_manager)),
        VisualKind::OSCILLOSCOPE => Box::new(oscilloscope::create(visual_id, visual_manager)),
        VisualKind::SPECTROGRAM => Box::new(spectrogram::create(visual_id, visual_manager)),
        VisualKind::SPECTRUM => Box::new(spectrum::create(visual_id, visual_manager)),
        VisualKind::STEREOMETER => Box::new(stereometer::create(visual_id, visual_manager)),
        VisualKind::WAVEFORM => Box::new(waveform::create(visual_id, visual_manager)),
    };

    ActiveSettings::new(pane)
}
