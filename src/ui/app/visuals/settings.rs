//! Contains the settings panes for visual modules.
//! MOST modules in ./settings correspond to a module in ../../visualization.

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
use iced::widget::{container, text};
use iced::{Element, Length, Size};

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

const DEFAULT_UNSUPPORTED_SIZE: Size = Size::new(320.0, 160.0);
const LOUDNESS_SETTINGS_SIZE: Size = Size::new(360.0, 180.0);
const OSCILLOSCOPE_SETTINGS_SIZE: Size = Size::new(420.0, 720.0);
const SPECTROGRAM_SETTINGS_SIZE: Size = Size::new(560.0, 880.0);
const SPECTRUM_SETTINGS_SIZE: Size = Size::new(420.0, 720.0);
const STEREOMETER_SETTINGS_SIZE: Size = Size::new(400.0, 500.0);
const WAVEFORM_SETTINGS_SIZE: Size = Size::new(500.0, 400.0);

#[derive(Debug)]
pub struct ActiveSettings {
    title: String,
    pane: Box<dyn ModuleSettingsPane>,
    preferred_size: Size,
}

impl ActiveSettings {
    pub fn new(title: String, preferred_size: Size, pane: Box<dyn ModuleSettingsPane>) -> Self {
        Self {
            title,
            pane,
            preferred_size,
        }
    }

    pub fn unsupported(title: String, visual_id: VisualId, kind: VisualKind) -> Self {
        let pane = Box::new(UnsupportedSettingsPane {
            visual_id,
            _kind: kind,
        });
        Self {
            title,
            pane,
            preferred_size: settings_size_for(kind),
        }
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

    pub fn preferred_size(&self) -> Size {
        self.preferred_size
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
    let preferred_size = settings_size_for(kind);

    let pane: Box<dyn ModuleSettingsPane> = match kind {
        VisualKind::LOUDNESS => Box::new(loudness::create(visual_id, visual_manager)),
        VisualKind::OSCILLOSCOPE => Box::new(oscilloscope::create(visual_id, visual_manager)),
        VisualKind::SPECTROGRAM => Box::new(spectrogram::create(visual_id, visual_manager)),
        VisualKind::SPECTRUM => Box::new(spectrum::create(visual_id, visual_manager)),
        VisualKind::STEREOMETER => Box::new(stereometer::create(visual_id, visual_manager)),
        VisualKind::WAVEFORM => Box::new(waveform::create(visual_id, visual_manager)),
        _ => return ActiveSettings::unsupported(title, visual_id, kind),
    };

    ActiveSettings::new(title, preferred_size, pane)
}

fn settings_size_for(kind: VisualKind) -> Size {
    match kind {
        VisualKind::LOUDNESS => LOUDNESS_SETTINGS_SIZE,
        VisualKind::OSCILLOSCOPE => OSCILLOSCOPE_SETTINGS_SIZE,
        VisualKind::SPECTROGRAM => SPECTROGRAM_SETTINGS_SIZE,
        VisualKind::SPECTRUM => SPECTRUM_SETTINGS_SIZE,
        VisualKind::STEREOMETER => STEREOMETER_SETTINGS_SIZE,
        VisualKind::WAVEFORM => WAVEFORM_SETTINGS_SIZE,
        _ => DEFAULT_UNSUPPORTED_SIZE,
    }
}

#[derive(Debug)]
struct UnsupportedSettingsPane {
    visual_id: VisualId,
    _kind: VisualKind,
}

impl ModuleSettingsPane for UnsupportedSettingsPane {
    fn visual_id(&self) -> VisualId {
        self.visual_id
    }

    fn view(&self) -> Element<'_, SettingsMessage> {
        container(text("No adjustable settings available yet.").size(14))
            .width(Length::Shrink)
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
