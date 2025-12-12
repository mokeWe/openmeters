//! Main application logic.

pub mod config;
pub mod visuals;

use crate::audio::pw_registry::RegistrySnapshot;
use crate::ui::channel_subscription::channel_subscription;
use crate::ui::settings::SettingsHandle;
use crate::ui::theme;
use crate::ui::visualization::visual_manager::{
    VisualContent, VisualId, VisualKind, VisualManager, VisualManagerHandle, VisualMetadata,
    VisualSnapshot,
};
use async_channel::Receiver as AsyncReceiver;
use config::{ConfigMessage, ConfigPage};
use rustc_hash::FxHashMap;
use visuals::{
    ActiveSettings, SettingsMessage, VisualsMessage, VisualsPage, create_settings_panel,
};

use iced::alignment::Horizontal;
use iced::event::{self, Event};
use iced::keyboard::{self, Key};
use iced::widget::text::Wrapping;
use iced::widget::{button, column, container, mouse_area, row, scrollable, stack, text};
use iced::{Element, Length, Result, Settings, Size, Subscription, Task, daemon, exit, window};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

pub use config::RoutingCommand;

const APP_PADDING: f32 = 16.0;
const MIN_WINDOW_SIZE: Size = Size::new(200.0, 150.0);

fn open_window(size: Size, decorations: bool, transparent: bool) -> (window::Id, Task<window::Id>) {
    window::open(window::Settings {
        size,
        min_size: Some(MIN_WINDOW_SIZE),
        resizable: true,
        decorations,
        transparent,
        ..Default::default()
    })
}

fn empty_view<M: 'static>() -> Element<'static, M> {
    container(text(""))
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

#[derive(Debug)]
struct PopoutWindow {
    visual_id: VisualId,
    kind: VisualKind,
    original_index: usize,
    cached: Option<(VisualMetadata, VisualContent)>,
}

impl PopoutWindow {
    fn sync(&mut self, snapshot: &VisualSnapshot) {
        self.cached = snapshot
            .slots
            .iter()
            .find(|s| s.id == self.visual_id && s.enabled)
            .map(|s| (s.metadata, s.content.clone()));
    }

    fn view(&self) -> Element<'_, VisualsMessage> {
        let Some((meta, content)) = &self.cached else {
            return empty_view();
        };
        mouse_area(
            container(content.render(*meta))
                .width(Length::Fill)
                .height(Length::Fill),
        )
        .on_right_press(VisualsMessage::SettingsRequested {
            visual_id: self.visual_id,
            kind: self.kind,
        })
        .into()
    }
}

#[derive(Clone)]
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
    daemon(move || UiApp::new(config.clone()), update, view)
        .settings(settings)
        .subscription(|state: &UiApp| state.subscription())
        .title(|state: &UiApp, window| state.title(window))
        .theme(|state: &UiApp, window| state.theme(window))
        .run()
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
    overlay_until: Option<std::time::Instant>,
    main_window_id: window::Id,
    main_window_size: Size,
    settings_window: Option<(window::Id, ActiveSettings)>,
    popout_windows: FxHashMap<window::Id, PopoutWindow>,
    focused_window: Option<window::Id>,
    exit_warning_until: Option<std::time::Instant>,
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
    PopOutOrDock,
    QuitRequested,
    ResizeRequested,
    WindowOpened,
    WindowClosed(window::Id),
    WindowResized(window::Id, Size),
    WindowFocused(window::Id),
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
            min_size: Some(MIN_WINDOW_SIZE),
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
                overlay_until: None,
                main_window_id,
                main_window_size: Size::new(420.0, 520.0),
                settings_window: None,
                popout_windows: FxHashMap::default(),
                focused_window: Some(main_window_id),
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

        subscriptions.push(event::listen_with(|evt, _status, window_id| match evt {
            Event::Window(window::Event::Focused) => Some(Message::WindowFocused(window_id)),
            _ => None,
        }));

        subscriptions.push(keyboard::listen().filter_map(|event| {
            let keyboard::Event::KeyPressed { key, modifiers, .. } = event else {
                return None;
            };
            match key {
                Key::Character(ref v)
                    if modifiers.control() && modifiers.shift() && v.eq_ignore_ascii_case("h") =>
                {
                    Some(Message::ToggleChrome)
                }
                Key::Named(keyboard::key::Named::Space) if modifiers.control() => {
                    Some(Message::PopOutOrDock)
                }
                Key::Character(ref v) if modifiers.is_empty() && v.eq_ignore_ascii_case("p") => {
                    Some(Message::TogglePause)
                }
                Key::Character(ref v) if modifiers.is_empty() && v.eq_ignore_ascii_case("q") => {
                    Some(Message::QuitRequested)
                }
                _ => None,
            }
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
        let old = self.settings_window.take();
        if old
            .as_ref()
            .is_some_and(|(_, p)| p.visual_id() == visual_id)
        {
            self.settings_window = old.map(|(id, _)| (id, panel));
            return Task::none();
        }
        let (id, open) = open_window(Size::new(480.0, 600.0), true, false);
        self.settings_window = Some((id, panel));
        if let Some((old_id, _)) = old {
            Task::batch([window::close(old_id), open.map(|_| Message::WindowOpened)])
        } else {
            open.map(|_| Message::WindowOpened)
        }
    }

    fn open_popout_window(&mut self, visual_id: VisualId, kind: VisualKind) -> Task<Message> {
        if self
            .popout_windows
            .values()
            .any(|w| w.visual_id == visual_id)
        {
            return Task::none();
        }
        let snapshot = self.visual_manager.snapshot();
        let Some((idx, slot)) = snapshot
            .slots
            .iter()
            .enumerate()
            .find(|(_, s)| s.id == visual_id)
        else {
            return Task::none();
        };
        let size = Size::new(
            slot.metadata.preferred_width.max(400.0),
            slot.metadata.preferred_height.max(300.0),
        );
        let (id, open) = open_window(size, true, true);
        let mut popout = PopoutWindow {
            visual_id,
            kind,
            original_index: idx,
            cached: None,
        };
        popout.sync(&snapshot);
        self.popout_windows.insert(id, popout);
        open.map(|_| Message::WindowOpened)
    }

    fn on_window_closed(&mut self, id: window::Id) -> Task<Message> {
        if id == self.main_window_id {
            return exit();
        }
        if self
            .settings_window
            .as_ref()
            .is_some_and(|(wid, _)| *wid == id)
        {
            self.settings_window = None;
        }
        self.popout_windows.remove(&id);
        Task::none()
    }

    fn popped_out_ids(&self) -> Vec<VisualId> {
        self.popout_windows.values().map(|w| w.visual_id).collect()
    }

    fn sync_all_windows(&mut self) -> Task<Message> {
        let snapshot = self.visual_manager.snapshot();
        let settings_task = if let Some((id, panel)) = &self.settings_window {
            let invalid = !snapshot
                .slots
                .iter()
                .any(|s| s.id == panel.visual_id() && s.enabled);
            if invalid {
                let id = *id;
                self.settings_window = None;
                window::close::<Message>(id)
            } else {
                Task::none()
            }
        } else {
            Task::none()
        };
        for p in self.popout_windows.values_mut() {
            p.sync(&snapshot);
        }
        let invalid: Vec<_> = self
            .popout_windows
            .iter()
            .filter_map(|(id, p)| p.cached.is_none().then_some(*id))
            .collect();
        self.popout_windows.retain(|_, p| p.cached.is_some());
        self.visuals_page
            .apply_snapshot_excluding(snapshot, &self.popped_out_ids());
        if invalid.is_empty() {
            settings_task
        } else {
            Task::batch([
                settings_task,
                Task::batch(invalid.into_iter().map(window::close)),
            ])
        }
    }

    fn title(&self, window: window::Id) -> String {
        if window == self.main_window_id {
            return "OpenMeters".into();
        }
        let is_settings = self
            .settings_window
            .as_ref()
            .is_some_and(|(id, _)| *id == window);
        let vid = self
            .settings_window
            .as_ref()
            .filter(|(id, _)| *id == window)
            .map(|(_, p)| p.visual_id())
            .or_else(|| self.popout_windows.get(&window).map(|p| p.visual_id));
        vid.and_then(|id| {
            self.visual_manager
                .snapshot()
                .slots
                .iter()
                .find(|s| s.id == id)
                .map(|s| {
                    format!(
                        "{}{} - OpenMeters",
                        s.metadata.display_name,
                        if is_settings { " settings" } else { "" }
                    )
                })
        })
        .unwrap_or_else(|| "OpenMeters".into())
    }

    fn theme(&self, window: window::Id) -> iced::Theme {
        let bg = (window == self.main_window_id || self.popout_windows.contains_key(&window))
            .then(|| {
                self.settings_handle
                    .borrow()
                    .settings()
                    .background_color
                    .map(|c| c.to_color())
            })
            .flatten();
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
            .map(|d| Instant::now() < d)
            .unwrap_or(false);

        let mut toasts = Vec::new();
        if !self.ui_visible && show_overlay {
            toasts.push("press ctrl+shift+h to restore controls");
        }
        if self.rendering_paused {
            toasts.push("rendering paused (press p to resume)");
        }
        if self.exit_warning_until.is_some_and(|d| Instant::now() < d) {
            toasts.push("press q again to exit");
        }

        let toast_bar = if toasts.is_empty() {
            None
        } else {
            Some(
                container(
                    row(toasts
                        .iter()
                        .map(|m| container(text(*m).size(11)).padding([2, 6]).into())
                        .collect::<Vec<_>>())
                    .spacing(12),
                )
                .width(Length::Fill)
                .align_x(Horizontal::Center),
            )
        };

        let content: Element<'_, Message> = if self.ui_visible {
            let mut tabs = row![
                tab_button("config", Page::Config, self.current_page),
                tab_button("visuals", Page::Visuals, self.current_page),
            ]
            .spacing(8)
            .width(Length::Fill);
            if !decorations {
                tabs = tabs.push(drag_handle(Message::WindowDragged(self.main_window_id)));
            }

            let page_content: Element<'_, Message> = match self.current_page {
                Page::Config => container(self.config_page.view().map(Message::Config))
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .style(theme::opaque_container)
                    .into(),
                Page::Visuals => visuals,
            };

            let mut inner = column![
                tabs,
                container(page_content)
                    .width(Length::Fill)
                    .height(Length::Fill)
            ]
            .spacing(12);
            if let Some(t) = toast_bar {
                inner = column![inner, t].spacing(0);
            }
            container(inner)
                .width(Length::Fill)
                .height(Length::Fill)
                .padding(APP_PADDING)
                .into()
        } else {
            let mut inner = column![container(visuals).width(Length::Fill).height(Length::Fill)];
            if let Some(t) = toast_bar {
                inner = inner.push(t);
            }
            inner
                .spacing(0)
                .width(Length::Fill)
                .height(Length::Fill)
                .into()
        };

        if decorations {
            content
        } else {
            stack![
                content,
                container(resize_handle(Message::ResizeRequested))
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .align_x(Horizontal::Right)
                    .align_y(iced::alignment::Vertical::Bottom)
                    .padding(4)
            ]
            .into()
        }
    }

    fn recreate_windows(&mut self, decorations: bool) -> Task<Message> {
        let old_main = self.main_window_id;
        let (id, open) = open_window(self.main_window_size, decorations, true);
        self.main_window_id = id;
        let settings_task = self
            .settings_window
            .take()
            .map(|(old_id, panel)| {
                let vid = panel.visual_id();
                let snapshot = self.visual_manager.snapshot();
                snapshot
                    .slots
                    .iter()
                    .find(|s| s.id == vid)
                    .map(|s| {
                        let (nid, nopen) = open_window(Size::new(480.0, 600.0), true, false);
                        self.settings_window = Some((
                            nid,
                            create_settings_panel(vid, s.kind, &self.visual_manager),
                        ));
                        Task::batch([nopen.map(|_| Message::WindowOpened), window::close(old_id)])
                    })
                    .unwrap_or_else(|| window::close(old_id))
            })
            .unwrap_or_else(Task::none);
        Task::batch([
            open.map(|_| Message::WindowOpened),
            window::close(old_main),
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
            let sync_task = app.sync_all_windows();
            Task::batch([task, window_task, sync_task])
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
        Message::PopOutOrDock => {
            if let Some(focused) = app.focused_window
                && let Some(popout) = app.popout_windows.remove(&focused)
            {
                app.visual_manager
                    .borrow_mut()
                    .restore_position(popout.visual_id, popout.original_index);
                app.visuals_page
                    .apply_snapshot_excluding(app.visual_manager.snapshot(), &app.popped_out_ids());
                return window::close(focused);
            }
            if let Some((id, kind)) = app.visuals_page.hovered_visual() {
                let task = app.open_popout_window(id, kind);
                let snapshot = app.visual_manager.snapshot();
                app.visuals_page
                    .apply_snapshot_excluding(snapshot, &app.popped_out_ids());
                task
            } else {
                Task::none()
            }
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
        Message::ResizeRequested => {
            window::drag_resize(app.main_window_id, window::Direction::SouthEast)
        }
        Message::AudioFrame(samples) => {
            if app.rendering_paused {
                return Task::none();
            }
            app.visual_manager.borrow_mut().ingest_samples(&samples);
            app.sync_all_windows()
        }
        Message::WindowOpened => Task::none(),
        Message::WindowClosed(id) => app.on_window_closed(id),
        Message::WindowResized(id, size) => {
            if id == app.main_window_id {
                app.main_window_size = size;
            }
            Task::none()
        }
        Message::WindowFocused(id) => {
            app.focused_window = Some(id);
            Task::none()
        }
        Message::WindowDragged(id) => window::drag(id),
        Message::SettingsWindow(id, settings_message) => {
            if let Some((wid, panel)) = app.settings_window.as_mut()
                && *wid == id
            {
                panel.handle_message(&settings_message, &app.visual_manager, &app.settings_handle);
            }
            Task::none()
        }
    }
}

fn settings_view(panel: &ActiveSettings) -> Element<'_, SettingsMessage> {
    container(
        scrollable(panel.view())
            .width(Length::Fill)
            .height(Length::Fill),
    )
    .width(Length::Fill)
    .height(Length::Fill)
    .padding(16)
    .style(theme::weak_container)
    .into()
}

fn view(app: &UiApp, window: window::Id) -> Element<'_, Message> {
    if window == app.main_window_id {
        app.main_window_view()
    } else if let Some((_, panel)) = app.settings_window.as_ref().filter(|(id, _)| *id == window) {
        settings_view(panel).map(move |msg| Message::SettingsWindow(window, msg))
    } else if let Some(popout) = app.popout_windows.get(&window) {
        popout.view().map(Message::Visuals)
    } else {
        empty_view()
    }
}

fn tab_button(
    label: &'static str,
    target: Page,
    current: Page,
) -> iced::widget::Button<'static, Message> {
    let active = current == target;
    let mut btn = button(
        container(text(label).wrapping(Wrapping::None))
            .width(Length::Fill)
            .clip(true),
    )
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
