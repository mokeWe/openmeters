mod oscilloscope;
mod spectrogram;
mod spectrum;

use crate::ui::settings::SettingsHandle;
use crate::ui::visualization::visual_manager::{VisualId, VisualKind, VisualManagerHandle};
use iced::widget::{container, text};
use iced::{Element, Length};

#[derive(Debug, Clone)]
pub enum SettingsMessage {
    Oscilloscope(oscilloscope::Message),
    Spectrogram(spectrogram::Message),
    Spectrum(spectrum::Message),
}

impl SettingsMessage {
    pub fn kind(&self) -> VisualKind {
        match self {
            SettingsMessage::Oscilloscope(_) => VisualKind::Oscilloscope,
            SettingsMessage::Spectrogram(_) => VisualKind::Spectrogram,
            SettingsMessage::Spectrum(_) => VisualKind::Spectrum,
        }
    }
}

pub trait ModuleSettingsPane: std::fmt::Debug + 'static {
    fn visual_id(&self) -> VisualId;
    fn kind(&self) -> VisualKind;
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
    title: String,
    pane: Box<dyn ModuleSettingsPane>,
}

impl ActiveSettings {
    pub fn new(title: String, pane: Box<dyn ModuleSettingsPane>) -> Self {
        Self { title, pane }
    }

    pub fn unsupported(title: String, visual_id: VisualId, kind: VisualKind) -> Self {
        let pane = Box::new(UnsupportedSettingsPane { visual_id, kind });
        Self { title, pane }
    }

    pub fn title(&self) -> &str {
        &self.title
    }

    pub fn set_title(&mut self, title: String) {
        self.title = title;
    }

    pub fn visual_id(&self) -> VisualId {
        self.pane.visual_id()
    }

    pub fn kind(&self) -> VisualKind {
        self.pane.kind()
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
    title: String,
    visual_id: VisualId,
    kind: VisualKind,
    visual_manager: &VisualManagerHandle,
) -> ActiveSettings {
    match kind {
        VisualKind::Oscilloscope => {
            let pane = oscilloscope::create(visual_id, visual_manager);
            ActiveSettings::new(title, Box::new(pane))
        }
        VisualKind::Spectrogram => {
            let pane = spectrogram::create(visual_id, visual_manager);
            ActiveSettings::new(title, Box::new(pane))
        }
        VisualKind::Spectrum => {
            let pane = spectrum::create(visual_id, visual_manager);
            ActiveSettings::new(title, Box::new(pane))
        }
        _ => ActiveSettings::unsupported(title, visual_id, kind),
    }
}

#[derive(Debug)]
struct UnsupportedSettingsPane {
    visual_id: VisualId,
    kind: VisualKind,
}

impl ModuleSettingsPane for UnsupportedSettingsPane {
    fn visual_id(&self) -> VisualId {
        self.visual_id
    }

    fn kind(&self) -> VisualKind {
        self.kind
    }

    fn view(&self) -> Element<'_, SettingsMessage> {
        container(text("No adjustable settings available yet.").size(14))
            .width(Length::Fill)
            .into()
    }

    fn handle(
        &mut self,
        _message: &SettingsMessage,
        _visual_manager: &VisualManagerHandle,
        _settings: &SettingsHandle,
    ) {
    }
}
