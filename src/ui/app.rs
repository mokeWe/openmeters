//! Main application logic.

pub mod config;
pub mod visuals;

use crate::audio::pw_registry::RegistrySnapshot;
use crate::ui::channel_subscription::channel_subscription;
use crate::ui::settings::SettingsHandle;
use crate::ui::settings_window::{DEFAULT_SETTINGS_SIZE, SettingsWindow};
use crate::ui::theme;
use crate::ui::visualization::visual_manager::{
    VisualId, VisualKind, VisualManager, VisualManagerHandle, VisualSnapshot,
};
use async_channel::Receiver as AsyncReceiver;
use config::{ConfigMessage, ConfigPage};
use visuals::{VisualsMessage, VisualsPage, create_settings_panel};

use iced::alignment::Horizontal;
use iced::keyboard::{self, Key};
use iced::widget::{button, column, container, mouse_area, row, stack, text};
use iced::{
    Element, Length, Point, Result, Settings, Size, Subscription, Task, daemon, event, exit, mouse,
    window,
};
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
    is_resizing: bool,
    overlay_until: Option<Instant>,
    main_window_id: window::Id,
    main_window_size: Size,
    settings_window: Option<SettingsWindow>,
    exit_warning_until: Option<Instant>,
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
    QuitRequested,
    ResizeStarted,
    ResizeEnded,
    CursorMoved(Point),
    WindowOpened,
    WindowClosed(window::Id),
    WindowResized(window::Id, Size),
    WindowDragged(window::Id),
    SettingsWindow(window::Id, visuals::SettingsMessage),
}

impl UiApp {
    fn new(config: UiConfig) -> (Self, Task<Message>) {
        let UiConfig {
            routing_sender,
            registry_updates,
            audio_frames,
        } = config;

        let settings = SettingsHandle::load_or_default();
        let (visual_settings, decorations) = {
            let guard = settings.borrow();
            (
                guard.settings().visuals.clone(),
                guard.settings().decorations,
            )
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
            decorations,
            transparent: true,
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
                is_resizing: false,
                overlay_until: None,
                main_window_id,
                main_window_size: Size::new(420.0, 520.0),
                settings_window: None,
                exit_warning_until: None,
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
        subscriptions
            .push(window::resize_events().map(|(id, size)| Message::WindowResized(id, size)));

        if self.is_resizing {
            subscriptions.push(event::listen().map(|e| match e {
                event::Event::Mouse(mouse::Event::CursorMoved { position }) => {
                    Message::CursorMoved(position)
                }
                event::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) => {
                    Message::ResizeEnded
                }
                _ => Message::CursorMoved(Point::ORIGIN),
            }));
        }

        subscriptions.push(keyboard::on_key_press(|key, modifiers| match key {
            Key::Character(ref v)
                if modifiers.control() && modifiers.shift() && v.eq_ignore_ascii_case("h") =>
            {
                Some(Message::ToggleChrome)
            }
            Key::Character(ref v) if modifiers.is_empty() && v.eq_ignore_ascii_case("p") => {
                Some(Message::TogglePause)
            }
            Key::Character(ref v) if modifiers.is_empty() && v.eq_ignore_ascii_case("q") => {
                Some(Message::QuitRequested)
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

    fn open_settings_window(&mut self, visual_id: VisualId, kind: VisualKind) -> Task<Message> {
        let panel = create_settings_panel(visual_id, kind, &self.visual_manager);

        let mut tasks = Vec::new();

        if let Some(existing) = self.settings_window.take() {
            if existing.panel.visual_id() == visual_id {
                self.settings_window = Some(SettingsWindow::new(existing.id, panel));
                return Task::none();
            }
            tasks.push(window::close(existing.id));
        }

        let (id, open) = window::open(window::Settings {
            size: DEFAULT_SETTINGS_SIZE,
            resizable: true,
            decorations: true,
            transparent: false,
            ..window::Settings::default()
        });

        self.settings_window = Some(SettingsWindow::new(id, panel));
        tasks.push(open.map(|_| Message::WindowOpened));

        Task::batch(tasks)
    }

    fn on_window_closed(&mut self, id: window::Id) -> Task<Message> {
        if id == self.main_window_id {
            return exit();
        }

        if self.settings_window.as_ref().is_some_and(|w| w.id == id) {
            self.settings_window = None;
        }

        Task::none()
    }

    fn sync_settings_window_with_snapshot(
        &mut self,
        snapshot: &VisualSnapshot,
    ) -> Option<Task<Message>> {
        let settings_window = self.settings_window.as_ref()?;
        let visual_id = settings_window.panel.visual_id();

        let still_enabled = snapshot
            .slots
            .iter()
            .any(|slot| slot.id == visual_id && slot.enabled);

        if still_enabled {
            None
        } else {
            let id = settings_window.id;
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
            .filter(|w| w.id == window)
            .and_then(|w| {
                let snapshot = self.visual_manager.snapshot();
                snapshot
                    .slots
                    .iter()
                    .find(|slot| slot.id == w.panel.visual_id())
                    .map(|slot| format!("{} settings - OpenMeters", slot.metadata.display_name))
            })
            .unwrap_or_else(|| "OpenMeters".to_string())
    }

    fn theme(&self, window: window::Id) -> iced::Theme {
        let bg = if window == self.main_window_id {
            self.settings_handle
                .borrow()
                .settings()
                .background_color
                .map(|c| c.to_color())
        } else {
            None
        };
        theme::theme(bg)
    }

    fn main_window_view(&self) -> Element<'_, Message> {
        let decorations = self.settings_handle.borrow().settings().decorations;
        let visuals = self
            .visuals_page
            .view(self.ui_visible)
            .map(Message::Visuals);
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
        if let Some(deadline) = self.exit_warning_until
            && Instant::now() < deadline
        {
            toasts.push("press q again to exit");
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
            let mut content = column![].spacing(12);

            let mut tabs = row![
                tab_button("config", Page::Config, self.current_page),
                tab_button("visuals", Page::Visuals, self.current_page),
            ]
            .spacing(8)
            .width(Length::Fill);

            if !decorations {
                tabs = tabs.push(drag_handle(Message::WindowDragged(self.main_window_id)));
            }

            let page_content = match self.current_page {
                Page::Config => {
                    let content = self.config_page.view().map(Message::Config);
                    container(content)
                        .width(Length::Fill)
                        .height(Length::Fill)
                        .style(theme::opaque_container)
                        .into()
                }
                Page::Visuals => visuals,
            };

            content = content.push(tabs).push(
                container(page_content)
                    .width(Length::Fill)
                    .height(Length::Fill),
            );

            if let Some(toast) = toast_bar {
                content = column![content, toast].spacing(0);
            }

            let content = container(content)
                .width(Length::Fill)
                .height(Length::Fill)
                .padding(APP_PADDING);

            if !decorations {
                stack![
                    content,
                    container(resize_handle(Message::ResizeStarted))
                        .width(Length::Fill)
                        .height(Length::Fill)
                        .align_x(Horizontal::Right)
                        .align_y(iced::alignment::Vertical::Bottom)
                        .padding(4)
                ]
                .into()
            } else {
                content.into()
            }
        } else {
            let mut content = column![container(visuals).width(Length::Fill).height(Length::Fill)];

            if let Some(toast) = toast_bar {
                content = content.push(toast);
            }

            let content = content.spacing(0).width(Length::Fill).height(Length::Fill);

            if !decorations {
                stack![
                    content,
                    container(resize_handle(Message::ResizeStarted))
                        .width(Length::Fill)
                        .height(Length::Fill)
                        .align_x(Horizontal::Right)
                        .align_y(iced::alignment::Vertical::Bottom)
                        .padding(4)
                ]
                .into()
            } else {
                content.into()
            }
        }
    }

    fn recreate_windows(&mut self, decorations: bool) -> Task<Message> {
        let old_main_id = self.main_window_id;
        let (new_main_id, open_main) = window::open(window::Settings {
            size: self.main_window_size,
            resizable: true,
            decorations,
            transparent: true,
            ..window::Settings::default()
        });
        self.main_window_id = new_main_id;

        let settings_task = if let Some(sw) = &self.settings_window {
            let old_sw_id = sw.id;
            let visual_id = sw.panel.visual_id();

            let snapshot = self.visual_manager.snapshot();
            if let Some(slot) = snapshot.slots.iter().find(|s| s.id == visual_id) {
                let panel = create_settings_panel(visual_id, slot.kind, &self.visual_manager);

                let (new_sw_id, open_sw) = window::open(window::Settings {
                    size: DEFAULT_SETTINGS_SIZE,
                    resizable: true,
                    decorations: true,
                    transparent: false,
                    ..window::Settings::default()
                });

                self.settings_window = Some(SettingsWindow::new(new_sw_id, panel));

                Task::batch([
                    open_sw.map(|_| Message::WindowOpened),
                    window::close(old_sw_id),
                ])
            } else {
                self.settings_window = None;
                window::close(old_sw_id)
            }
        } else {
            Task::none()
        };

        Task::batch([
            open_main.map(|_| Message::WindowOpened),
            window::close(old_main_id),
            settings_task,
        ])
    }
}

fn update(app: &mut UiApp, message: Message) -> Task<Message> {
    match message {
        Message::PageSelected(page) => {
            app.current_page = page;
            Task::none()
        }
        Message::Config(msg) => {
            let window_task = if let ConfigMessage::DecorationsToggled(enabled) = &msg {
                app.recreate_windows(*enabled)
            } else {
                Task::none()
            };

            let task = app.config_page.update(msg).map(Message::Config);
            app.visuals_page.sync_with_manager();

            if let Some(close_task) =
                app.sync_settings_window_with_snapshot(&app.visual_manager.snapshot())
            {
                Task::batch([task, window_task, close_task])
            } else {
                Task::batch([task, window_task])
            }
        }
        Message::Visuals(VisualsMessage::SettingsRequested { visual_id, kind }) => {
            app.open_settings_window(visual_id, kind)
        }
        Message::Visuals(VisualsMessage::WindowDragRequested) => window::drag(app.main_window_id),
        Message::Visuals(msg) => app.visuals_page.update(msg).map(Message::Visuals),
        Message::ToggleChrome => {
            app.toggle_ui_visibility();
            Task::none()
        }
        Message::TogglePause => {
            app.rendering_paused = !app.rendering_paused;
            Task::none()
        }
        Message::QuitRequested => {
            let now = Instant::now();
            if let Some(deadline) = app.exit_warning_until
                && now < deadline
            {
                return exit();
            }
            app.exit_warning_until = Some(now + Duration::from_secs(2));
            Task::none()
        }
        Message::ResizeStarted => {
            app.is_resizing = true;
            Task::none()
        }
        Message::ResizeEnded => {
            app.is_resizing = false;
            Task::none()
        }
        Message::CursorMoved(position) => {
            if app.is_resizing && position != Point::ORIGIN {
                window::resize(app.main_window_id, Size::new(position.x, position.y))
            } else {
                Task::none()
            }
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
        Message::WindowResized(id, size) => {
            if id == app.main_window_id {
                app.main_window_size = size;
            }
            Task::none()
        }
        Message::WindowDragged(id) => window::drag(id),
        Message::SettingsWindow(id, settings_message) => {
            if let Some(window) = app.settings_window.as_mut()
                && window.id == id
            {
                window.panel.handle_message(
                    &settings_message,
                    &app.visual_manager,
                    &app.settings_handle,
                );
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
        .filter(|window_state| window_state.id == window)
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
    let mut btn = button(text(label))
        .style(move |theme, status| theme::tab_button_style(theme, active, status));

    if !active {
        btn = btn.on_press(Message::PageSelected(target));
    }

    btn.width(Length::Fill).padding(8)
}

fn drag_handle<M: Clone + 'static>(message: M) -> Element<'static, M> {
    mouse_area(
        container(
            text("::")
                .size(14)
                .align_y(iced::alignment::Vertical::Center),
        )
        .padding(4),
    )
    .on_press(message)
    .interaction(iced::mouse::Interaction::Grab)
    .into()
}

fn resize_handle<M: Clone + 'static>(message: M) -> Element<'static, M> {
    mouse_area(container(text(" ")).width(20).height(20))
        .on_press(message)
        .interaction(iced::mouse::Interaction::ResizingDiagonallyDown)
        .into()
}
