use crate::ui::app::visuals::{ActiveSettings, SettingsMessage};
use crate::ui::theme;
use iced::widget::{button, column, container, mouse_area, row, scrollable, text};
use iced::{Background, Element, Length, Padding, Size};

#[derive(Debug, Clone)]
pub enum SettingsWindowMessage {
    WindowDragged,
    Close,
    Settings(SettingsMessage),
}

#[derive(Debug)]
pub struct SettingsWindow {
    pub id: iced::window::Id,
    pub panel: ActiveSettings,
}

impl SettingsWindow {
    pub fn view(&self, decorations: bool) -> Element<'_, SettingsWindowMessage> {
        let (target_size, use_scroll) =
            compute_window_layout(self.panel.preferred_size(), decorations);
        let body = self.panel.view().map(SettingsWindowMessage::Settings);

        let content: Element<'_, SettingsWindowMessage> = if use_scroll {
            scrollable(body)
                .width(Length::Fill)
                .height(Length::Fill)
                .into()
        } else {
            container(body).width(Length::Fill).into()
        };

        let inner = container(content)
            .width(Length::Fill)
            .height(Length::Fill)
            .padding(16);

        let content: Element<_> = if !decorations {
            column![
                row![
                    mouse_area(
                        container(text(self.panel.title()).size(14))
                            .width(Length::Fill)
                            .align_y(iced::alignment::Vertical::Center)
                    )
                    .on_press(SettingsWindowMessage::WindowDragged)
                    .interaction(iced::mouse::Interaction::Grab),
                    button(text("×").size(18).line_height(1.0))
                        .on_press(SettingsWindowMessage::Close)
                        .padding([2, 8])
                        .style(theme::surface_button_style)
                ]
                .spacing(8)
                .padding(Padding {
                    top: 8.0,
                    right: 8.0,
                    bottom: 0.0,
                    left: 8.0
                })
                .align_y(iced::alignment::Vertical::Center),
                inner
            ]
            .into()
        } else {
            inner.into()
        };

        container(content)
            .width(Length::Fixed(target_size.width))
            .height(if use_scroll {
                Length::Fixed(target_size.height)
            } else {
                Length::Shrink
            })
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

pub fn compute_window_layout(preferred: Size, decorations: bool) -> (Size, bool) {
    let chrome = if decorations { 32.0 } else { 64.0 };
    let size = Size::new(preferred.width + 32.0, preferred.height + chrome);
    let max = Size::new(560.0, 680.0);
    let use_scroll = size.height > max.height;
    (
        Size::new(
            size.width.min(max.width),
            if use_scroll { max.height } else { size.height },
        ),
        use_scroll,
    )
}

pub fn size_changed(before: Size, after: Size) -> bool {
    (before.width - after.width).abs() > 0.5 || (before.height - after.height).abs() > 0.5
}
