use crate::ui::app::visuals::{ActiveSettings, SettingsMessage};
use crate::ui::theme;
use iced::widget::{container, scrollable};
use iced::{Element, Length, Size};

pub const DEFAULT_SETTINGS_SIZE: Size = Size::new(480.0, 600.0);

#[derive(Debug)]
pub struct SettingsWindow {
    pub id: iced::window::Id,
    pub panel: ActiveSettings,
}

impl SettingsWindow {
    pub fn new(id: iced::window::Id, panel: ActiveSettings) -> Self {
        Self { id, panel }
    }

    pub fn view(&self) -> Element<'_, SettingsMessage> {
        let body = self.panel.view();

        let content = scrollable(body).width(Length::Fill).height(Length::Fill);

        container(content)
            .width(Length::Fill)
            .height(Length::Fill)
            .padding(16)
            .style(theme::weak_container)
            .into()
    }
}
