use crate::ui::app::visuals::{ActiveSettings, SettingsMessage};
use iced::widget::{container, scrollable};
use iced::{Background, Element, Length, Size};

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
            .style(|theme: &iced::Theme| {
                let palette = theme.extended_palette();
                iced::widget::container::Style {
                    background: Some(Background::Color(palette.background.weak.color)),
                    text_color: Some(palette.background.base.text),
                    border: crate::ui::theme::sharp_border(),
                    ..Default::default()
                }
            })
            .into()
    }
}
