pub mod config;
pub mod visuals;

use crate::audio::pw_registry::RegistrySnapshot;
use crate::ui::theme;
use crate::ui::visualization::audio_stream::AudioStreamSubscription;
use crate::ui::visualization::visual_manager::{VisualManager, VisualManagerHandle};
use async_channel::Receiver as AsyncReceiver;
use config::{ConfigMessage, ConfigPage};
use visuals::{VisualsMessage, VisualsPage};

use iced::advanced::subscription::from_recipe;
use iced::alignment::Horizontal;
use iced::keyboard::{self, Key};
use iced::widget::{button, column, container, row, text};
use iced::{Element, Length, Result, Settings, Size, Subscription, Task, application};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

pub use config::RoutingCommand;

const APP_PADDING: f32 = 16.0;

pub struct UiConfig {
    routing_sender: mpsc::Sender<RoutingCommand>,
    registry_updates: Option<Arc<AsyncReceiver<RegistrySnapshot>>>,
    audio_frames: Option<Arc<AsyncReceiver<Vec<f32>>>>,
}

impl UiConfig {
    pub fn new(
        routing_sender: mpsc::Sender<RoutingCommand>,
        registry_updates: Option<Arc<AsyncReceiver<RegistrySnapshot>>>,
    ) -> Self {
        Self {
            routing_sender,
            registry_updates,
            audio_frames: None,
        }
    }

    pub fn with_audio_stream(mut self, audio_frames: Arc<AsyncReceiver<Vec<f32>>>) -> Self {
        self.audio_frames = Some(audio_frames);
        self
    }
}

pub fn run(config: UiConfig) -> Result {
    let settings = Settings {
        id: Some(String::from("openmeters-ui")),
        ..Settings::default()
    };

    application("OpenMeters", update, view)
        .settings(settings)
        .window_size(Size::new(420.0, 520.0))
        .resizable(true)
        .theme(|_| theme::theme())
        .subscription(|state: &UiApp| state.subscription())
        .run_with(move || UiApp::new(config))
}

#[derive(Debug)]
struct UiApp {
    current_page: Page,
    config_page: ConfigPage,
    visuals_page: VisualsPage,
    visual_manager: VisualManagerHandle,
    audio_frames: Option<Arc<AsyncReceiver<Vec<f32>>>>,
    ui_visible: bool,
    overlay_until: Option<Instant>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Page {
    Config,
    Visuals,
}

#[derive(Debug, Clone)]
enum Message {
    PageSelected(Page),
    Config(ConfigMessage),
    Visuals(VisualsMessage),
    AudioFrame(Vec<f32>),
    ToggleChrome,
}

impl UiApp {
    fn new(config: UiConfig) -> (Self, Task<Message>) {
        let UiConfig {
            routing_sender,
            registry_updates,
            audio_frames,
        } = config;

        let visual_manager = VisualManagerHandle::new(VisualManager::new());

        let config_page = ConfigPage::new(
            routing_sender.clone(),
            registry_updates.clone(),
            visual_manager.clone(),
        );
        let visuals_page = VisualsPage::new(visual_manager.clone());

        (
            Self {
                current_page: Page::Config,
                config_page,
                visuals_page,
                visual_manager,
                audio_frames,
                ui_visible: true,
                overlay_until: None,
            },
            Task::none(),
        )
    }

    fn subscription(&self) -> Subscription<Message> {
        let page_subscription = match self.current_page {
            Page::Config => self.config_page.subscription().map(Message::Config),
            Page::Visuals => self.visuals_page.subscription().map(Message::Visuals),
        };

        let mut subscriptions = vec![page_subscription];

        if let Some(receiver) = &self.audio_frames {
            subscriptions.push(
                from_recipe(AudioStreamSubscription::new(Arc::clone(receiver)))
                    .map(Message::AudioFrame),
            );
        }

        subscriptions.push(keyboard::on_key_press(|key, modifiers| {
            if modifiers.control()
                && modifiers.shift()
                && matches!(key, Key::Character(value) if value.eq_ignore_ascii_case("h"))
            {
                return Some(Message::ToggleChrome);
            }

            None
        }));

        Subscription::batch(subscriptions)
    }

    fn toggle_ui_visibility(&mut self) {
        self.ui_visible = !self.ui_visible;

        if self.ui_visible {
            self.overlay_until = None;
        } else {
            self.overlay_until = Some(Instant::now() + Duration::from_secs(2));

            if self.current_page != Page::Visuals {
                self.current_page = Page::Visuals;
            }
        }
    }
}

fn update(app: &mut UiApp, message: Message) -> Task<Message> {
    match message {
        Message::PageSelected(page) => {
            app.current_page = page;
            Task::none()
        }
        Message::Config(msg) => {
            let task = app.config_page.update(msg).map(Message::Config);
            app.visuals_page.sync_with_manager();
            task
        }
        Message::Visuals(msg) => app.visuals_page.update(msg).map(Message::Visuals),
        Message::ToggleChrome => {
            app.toggle_ui_visibility();
            Task::none()
        }
        Message::AudioFrame(samples) => {
            let snapshot = {
                let mut manager = app.visual_manager.borrow_mut();
                manager.ingest_samples(&samples);
                manager.snapshot()
            };

            app.visuals_page.apply_snapshot(snapshot);
            Task::none()
        }
    }
}

fn view(app: &UiApp) -> Element<'_, Message> {
    let content: Element<'_, Message> = if app.ui_visible {
        let tabs = row![
            tab_button("config", Page::Config, app.current_page),
            tab_button("visuals", Page::Visuals, app.current_page)
        ]
        .spacing(8)
        .width(Length::Fill);

        let page_content = match app.current_page {
            Page::Config => app.config_page.view().map(Message::Config),
            Page::Visuals => app.visuals_page.view().map(Message::Visuals),
        };

        let layout = column![
            tabs,
            container(page_content)
                .width(Length::Fill)
                .height(Length::Fill)
        ]
        .spacing(12);

        container(layout)
            .width(Length::Fill)
            .height(Length::Fill)
            .padding(APP_PADDING)
            .into()
    } else {
        let overlay_active = app
            .overlay_until
            .map(|deadline| Instant::now() < deadline)
            .unwrap_or(false);

        let visuals = app.visuals_page.view().map(Message::Visuals);

        if overlay_active {
            let toast =
                container(text("press ctrl+shift+h to restore controls").size(14)).padding(12);

            column![
                container(visuals).width(Length::Fill).height(Length::Fill),
                container(toast)
                    .width(Length::Fill)
                    .align_x(Horizontal::Center),
            ]
            .spacing(12)
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
        } else {
            container(visuals)
                .width(Length::Fill)
                .height(Length::Fill)
                .into()
        }
    };

    content
}

fn tab_button(
    label: &'static str,
    target: Page,
    current: Page,
) -> iced::widget::Button<'static, Message> {
    let active = current == target;
    let mut btn = button(text(label)).style(move |_theme, status| tab_button_style(active, status));

    if !active {
        btn = btn.on_press(Message::PageSelected(target));
    }

    btn.width(Length::Fill).padding(8)
}

fn tab_button_style(
    active: bool,
    status: iced::widget::button::Status,
) -> iced::widget::button::Style {
    let base_background = if active {
        theme::elevated_color()
    } else {
        theme::surface_color()
    };
    let mut style = iced::widget::button::Style {
        background: Some(iced::Background::Color(base_background)),
        text_color: theme::text_color(),
        border: theme::sharp_border(),
        ..Default::default()
    };

    match status {
        iced::widget::button::Status::Hovered => {
            style.background = Some(iced::Background::Color(theme::hover_color()));
        }
        iced::widget::button::Status::Pressed => {
            style.border = theme::focus_border();
        }
        _ => {}
    }

    style
}
