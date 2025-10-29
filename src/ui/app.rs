//! the main application code.
//!
//! responsible for:
//! - managing application state and transitions
//!   between pages, e.g. config <-> visuals
//! - handling window management (e.g. main window, settings windows)
//! - routing messages between subcomponents
//! - managing the main event loop

pub mod config;
pub mod visuals;

use crate::audio::pw_registry::RegistrySnapshot;
use crate::ui::channel_subscription::channel_subscription;
use crate::ui::settings::SettingsHandle;
use crate::ui::theme;
use crate::ui::visualization::visual_manager::{
    VisualId, VisualKind, VisualManager, VisualManagerHandle, VisualSnapshot,
};
use async_channel::Receiver as AsyncReceiver;
use config::{ConfigMessage, ConfigPage};
use visuals::{
    ActiveSettings, SettingsMessage, VisualsMessage, VisualsPage, create_settings_panel,
};

use iced::alignment::Horizontal;
use iced::keyboard::{self, Key};
use iced::widget::{button, column, container, horizontal_space, row, scrollable, text};
use iced::{
    Background, Element, Length, Result, Settings, Size, Subscription, Task, daemon, exit, window,
};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

pub use config::RoutingCommand;

const APP_PADDING: f32 = 16.0;
const SETTINGS_WINDOW_PADDING: f32 = 16.0;
const SETTINGS_WINDOW_CONTENT_SPACING: f32 = 16.0;
const SETTINGS_WINDOW_HEADER_HEIGHT: f32 = 32.0;
const SETTINGS_WINDOW_MAX_SIZE: Size = Size {
    width: 560.0,
    height: 680.0,
};

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
    daemon(UiApp::title, update, view)
        .settings(settings)
        .subscription(|state: &UiApp| state.subscription())
        .theme(|state, window| state.theme(window))
        .run_with(move || UiApp::new(config))
}

#[derive(Debug)]
struct UiApp {
    current_page: Page,
    config_page: ConfigPage,
    visuals_page: VisualsPage,
    visual_manager: VisualManagerHandle,
    settings_handle: SettingsHandle,
    audio_frames: Option<Arc<AsyncReceiver<Vec<f32>>>>,
    ui_visible: bool,
    rendering_paused: bool,
    overlay_until: Option<Instant>,
    main_window_id: window::Id,
    settings_window: Option<SettingsWindow>,
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
    TogglePause,
    WindowOpened,
    WindowClosed(window::Id),
    SettingsWindow(window::Id, SettingsWindowMessage),
}

#[derive(Debug, Clone)]
enum SettingsWindowMessage {
    Close,
    Settings(SettingsMessage),
}

#[derive(Debug)]
struct SettingsWindow {
    id: window::Id,
    panel: ActiveSettings,
}

impl SettingsWindow {
    fn new(id: window::Id, panel: ActiveSettings) -> Self {
        Self { id, panel }
    }

    fn id(&self) -> window::Id {
        self.id
    }

    fn visual_id(&self) -> VisualId {
        self.panel.visual_id()
    }

    fn title(&self) -> &str {
        self.panel.title()
    }

    fn preferred_size(&self) -> Size {
        self.panel.preferred_size()
    }

    fn view(&self) -> Element<'_, SettingsWindowMessage> {
        let preferred = self.preferred_size();
        let (target_size, use_scroll) = compute_window_layout(preferred);
        let width = target_size.width;
        let height = target_size.height;

        let header_row = row![
            text(format!("{} settings", self.panel.title())).size(16),
            horizontal_space().width(Length::Fill),
            button(text("Close"))
                .padding([8, 12])
                .style(settings_button_style)
                .on_press(SettingsWindowMessage::Close)
        ]
        .spacing(12)
        .width(Length::Fill);

        let header = container(header_row).width(Length::Fill);

        let body_content = self.panel.view().map(SettingsWindowMessage::Settings);

        let body: Element<'_, SettingsWindowMessage> = if use_scroll {
            scrollable(body_content)
                .width(Length::Fill)
                .height(Length::Fill)
                .into()
        } else {
            container(body_content).width(Length::Fill).into()
        };

        let content = column![header, body]
            .spacing(SETTINGS_WINDOW_CONTENT_SPACING)
            .width(Length::Fill)
            .height(if use_scroll {
                Length::Fill
            } else {
                Length::Shrink
            });

        container(content)
            .width(Length::Fixed(width))
            .height(if use_scroll {
                Length::Fixed(height)
            } else {
                Length::Shrink
            })
            .padding(SETTINGS_WINDOW_PADDING)
            .style(settings_panel_style)
            .into()
    }
}

impl UiApp {
    fn new(config: UiConfig) -> (Self, Task<Message>) {
        let UiConfig {
            routing_sender,
            registry_updates,
            audio_frames,
        } = config;

        let settings = SettingsHandle::load_or_default();
        let visual_settings = {
            let guard = settings.borrow();
            guard.settings().visuals.clone()
        };

        let mut manager = VisualManager::new();
        manager.apply_visual_settings(&visual_settings);
        let visual_manager = VisualManagerHandle::new(manager);

        let config_page = ConfigPage::new(
            routing_sender.clone(),
            registry_updates.clone(),
            visual_manager.clone(),
            settings.clone(),
        );
        let visuals_page = VisualsPage::new(visual_manager.clone(), settings.clone());

        let (main_window_id, open_main) = window::open(window::Settings {
            size: Size::new(420.0, 520.0),
            resizable: true,
            ..window::Settings::default()
        });

        (
            Self {
                current_page: Page::Config,
                config_page,
                visuals_page,
                visual_manager,
                settings_handle: settings,
                audio_frames,
                ui_visible: true,
                rendering_paused: false,
                overlay_until: None,
                main_window_id,
                settings_window: None,
            },
            open_main.map(|_| Message::WindowOpened),
        )
    }

    fn subscription(&self) -> Subscription<Message> {
        let page_subscription = match self.current_page {
            Page::Config => self.config_page.subscription().map(Message::Config),
            Page::Visuals => self.visuals_page.subscription().map(Message::Visuals),
        };

        let mut subscriptions = vec![page_subscription];

        if let Some(receiver) = &self.audio_frames {
            subscriptions.push(channel_subscription(Arc::clone(receiver)).map(Message::AudioFrame));
        }

        subscriptions.push(window::close_events().map(Message::WindowClosed));

        subscriptions.push(keyboard::on_key_press(|key, modifiers| match key {
            Key::Character(ref v)
                if modifiers.control() && modifiers.shift() && v.eq_ignore_ascii_case("h") =>
            {
                Some(Message::ToggleChrome)
            }
            Key::Character(ref v) if modifiers.is_empty() && v.eq_ignore_ascii_case("p") => {
                Some(Message::TogglePause)
            }
            _ => None,
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

    fn open_settings_window(
        &mut self,
        title: String,
        visual_id: VisualId,
        kind: VisualKind,
    ) -> Task<Message> {
        let panel = create_settings_panel(title, visual_id, kind, &self.visual_manager);
        let (window_size, _) = compute_window_layout(panel.preferred_size());

        if let Some(mut existing) = self.settings_window.take() {
            if existing.visual_id() == visual_id {
                existing.panel = panel;
                let id = existing.id();
                self.settings_window = Some(existing);
                return window::resize::<Message>(id, window_size);
            }

            let close_task = window::close::<Message>(existing.id());
            let (id, open_task) = window::open(window::Settings {
                size: window_size,
                resizable: false,
                ..window::Settings::default()
            });

            self.settings_window = Some(SettingsWindow::new(id, panel));

            return Task::batch([close_task, open_task.map(|_| Message::WindowOpened)]);
        }

        let (id, open_task) = window::open(window::Settings {
            size: window_size,
            resizable: false,
            ..window::Settings::default()
        });

        self.settings_window = Some(SettingsWindow::new(id, panel));

        open_task.map(|_| Message::WindowOpened)
    }

    fn close_settings_window(&mut self, id: window::Id) -> Task<Message> {
        if self.settings_window.as_ref().is_some_and(|w| w.id() == id) {
            self.settings_window = None;
            window::close::<Message>(id)
        } else {
            Task::none()
        }
    }

    fn on_window_closed(&mut self, id: window::Id) -> Task<Message> {
        if id == self.main_window_id {
            return exit();
        }

        if self.settings_window.as_ref().is_some_and(|w| w.id() == id) {
            self.settings_window = None;
        }

        Task::none()
    }

    fn sync_settings_window_with_snapshot(
        &mut self,
        snapshot: &VisualSnapshot,
    ) -> Option<Task<Message>> {
        let settings_window = self.settings_window.as_mut()?;

        let visual_id = settings_window.visual_id();

        if let Some(slot) = snapshot
            .slots
            .iter()
            .find(|slot| slot.id == visual_id && slot.enabled)
        {
            settings_window
                .panel
                .set_title(slot.metadata.display_name.to_string());
            None
        } else {
            let id = settings_window.id();
            self.settings_window = None;
            Some(window::close::<Message>(id))
        }
    }

    fn title(&self, window: window::Id) -> String {
        if window == self.main_window_id {
            return "OpenMeters".to_string();
        }

        self.settings_window
            .as_ref()
            .filter(|w| w.id() == window)
            .map(|w| format!("{} settings — OpenMeters", w.title()))
            .unwrap_or_else(|| "OpenMeters".to_string())
    }

    fn theme(&self, _window: window::Id) -> iced::Theme {
        theme::theme()
    }

    fn main_window_view(&self) -> Element<'_, Message> {
        let visuals = self.visuals_page.view().map(Message::Visuals);
        let show_overlay = self
            .overlay_until
            .map(|deadline| Instant::now() < deadline)
            .unwrap_or(false);

        let mut toasts = Vec::new();
        if !self.ui_visible && show_overlay {
            toasts.push("press ctrl+shift+h to restore controls");
        }
        if self.rendering_paused {
            toasts.push("rendering paused (press p to resume)");
        }

        let toast_bar = if !toasts.is_empty() {
            let toast_elements: Vec<Element<'_, Message>> = toasts
                .iter()
                .map(|msg| container(text(*msg).size(11)).padding([2, 6]).into())
                .collect();
            Some(
                container(row(toast_elements).spacing(12))
                    .width(Length::Fill)
                    .align_x(Horizontal::Center),
            )
        } else {
            None
        };

        if self.ui_visible {
            let tabs = row![
                tab_button("config", Page::Config, self.current_page),
                tab_button("visuals", Page::Visuals, self.current_page)
            ]
            .spacing(8)
            .width(Length::Fill);

            let page_content = match self.current_page {
                Page::Config => self.config_page.view().map(Message::Config),
                Page::Visuals => visuals,
            };

            let mut content = column![
                tabs,
                container(page_content)
                    .width(Length::Fill)
                    .height(Length::Fill)
            ]
            .spacing(12);

            if let Some(toast) = toast_bar {
                content = column![content, toast].spacing(0);
            }

            container(content)
                .width(Length::Fill)
                .height(Length::Fill)
                .padding(APP_PADDING)
                .into()
        } else {
            let mut content = column![container(visuals).width(Length::Fill).height(Length::Fill)];

            if let Some(toast) = toast_bar {
                content = content.push(toast);
            }

            content
                .spacing(0)
                .width(Length::Fill)
                .height(Length::Fill)
                .into()
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

            if let Some(close_task) =
                app.sync_settings_window_with_snapshot(&app.visual_manager.snapshot())
            {
                Task::batch([task, close_task])
            } else {
                task
            }
        }
        Message::Visuals(VisualsMessage::SettingsRequested {
            title,
            visual_id,
            kind,
        }) => app.open_settings_window(title, visual_id, kind),
        Message::Visuals(msg) => app.visuals_page.update(msg).map(Message::Visuals),
        Message::ToggleChrome => {
            app.toggle_ui_visibility();
            Task::none()
        }
        Message::TogglePause => {
            app.rendering_paused = !app.rendering_paused;
            Task::none()
        }
        Message::AudioFrame(samples) => {
            if app.rendering_paused {
                return Task::none();
            }

            let snapshot = {
                let mut manager = app.visual_manager.borrow_mut();
                manager.ingest_samples(&samples);
                manager.snapshot()
            };

            let maybe_close = app.sync_settings_window_with_snapshot(&snapshot);
            app.visuals_page.apply_snapshot(snapshot);

            maybe_close.unwrap_or_else(Task::none)
        }
        Message::WindowOpened => Task::none(),
        Message::WindowClosed(id) => app.on_window_closed(id),
        Message::SettingsWindow(id, SettingsWindowMessage::Close) => app.close_settings_window(id),
        Message::SettingsWindow(id, SettingsWindowMessage::Settings(settings_message)) => {
            if let Some(window) = app.settings_window.as_mut()
                && window.id() == id
            {
                let before = compute_window_layout(window.preferred_size()).0;
                window.panel.handle_message(
                    &settings_message,
                    &app.visual_manager,
                    &app.settings_handle,
                );
                let after = compute_window_layout(window.preferred_size()).0;
                if size_changed(before, after) {
                    return window::resize::<Message>(id, after);
                }
            }
            Task::none()
        }
    }
}

fn view(app: &UiApp, window: window::Id) -> Element<'_, Message> {
    if window == app.main_window_id {
        app.main_window_view()
    } else if let Some(settings_window) = app
        .settings_window
        .as_ref()
        .filter(|window_state| window_state.id() == window)
    {
        settings_window
            .view()
            .map(move |msg| Message::SettingsWindow(window, msg))
    } else {
        container(text(""))
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }
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
    theme::button_style(base_background, status)
}

fn settings_button_style(
    _theme: &iced::Theme,
    status: iced::widget::button::Status,
) -> iced::widget::button::Style {
    theme::surface_button_style(status)
}

fn settings_panel_style(_theme: &iced::Theme) -> iced::widget::container::Style {
    let border = theme::sharp_border();
    iced::widget::container::Style {
        background: Some(Background::Color(theme::surface_color())),
        text_color: Some(theme::text_color()),
        border,
        ..Default::default()
    }
}

fn compute_window_layout(preferred: Size) -> (Size, bool) {
    let chrome_width = SETTINGS_WINDOW_PADDING * 2.0;
    let chrome_height = SETTINGS_WINDOW_PADDING * 2.0
        + SETTINGS_WINDOW_HEADER_HEIGHT
        + SETTINGS_WINDOW_CONTENT_SPACING;

    let desired_width = preferred.width + chrome_width;
    let desired_height = preferred.height + chrome_height;

    let width = desired_width.min(SETTINGS_WINDOW_MAX_SIZE.width);
    let max_height = SETTINGS_WINDOW_MAX_SIZE.height;

    if desired_height > max_height {
        (Size::new(width, max_height), true)
    } else {
        (Size::new(width, desired_height), false)
    }
}

fn size_changed(before: Size, after: Size) -> bool {
    const EPSILON: f32 = 0.5;
    (before.width - after.width).abs() > EPSILON || (before.height - after.height).abs() > EPSILON
}
