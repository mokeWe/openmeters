use crate::audio::VIRTUAL_SINK_NAME;
use crate::audio::pw_registry::RegistrySnapshot;
use crate::ui::application_row::ApplicationRow;
use crate::ui::hardware_sink::HardwareSinkCache;
use crate::ui::theme;
use crate::ui::visualization::visual_manager::{VisualKind, VisualManagerHandle};
use async_channel::Receiver as AsyncReceiver;
use iced::advanced::subscription::{EventStream, Hasher, Recipe, from_recipe};
use iced::alignment;
use iced::futures::{self, StreamExt};
use iced::widget::{Column, Row, Space, button, container, scrollable, text};
use iced::{Element, Length, Subscription, Task};
use std::collections::{HashMap, HashSet};
use std::hash::Hasher as _;
use std::sync::{Arc, mpsc};

const GRID_COLUMNS: usize = 4;

#[derive(Debug, Clone)]
pub enum RoutingCommand {
    SetApplicationEnabled { node_id: u32, enabled: bool },
}

#[derive(Debug, Clone)]
pub enum ConfigMessage {
    RegistryUpdated(RegistrySnapshot),
    ToggleChanged { node_id: u32, enabled: bool },
    ToggleApplicationsVisibility,
    VisualToggled { kind: VisualKind, enabled: bool },
}

#[derive(Debug)]
pub struct ConfigPage {
    routing_sender: mpsc::Sender<RoutingCommand>,
    registry_updates: Option<Arc<AsyncReceiver<RegistrySnapshot>>>,
    visual_manager: VisualManagerHandle,
    preferences: HashMap<u32, bool>,
    applications: Vec<ApplicationRow>,
    hardware_sink: HardwareSinkCache,
    registry_ready: bool,
    applications_expanded: bool,
}

impl ConfigPage {
    pub fn new(
        routing_sender: mpsc::Sender<RoutingCommand>,
        registry_updates: Option<Arc<AsyncReceiver<RegistrySnapshot>>>,
        visual_manager: VisualManagerHandle,
    ) -> Self {
        Self {
            routing_sender,
            registry_updates,
            visual_manager,
            preferences: HashMap::new(),
            applications: Vec::new(),
            hardware_sink: HardwareSinkCache::new(),
            registry_ready: false,
            applications_expanded: false,
        }
    }

    pub fn subscription(&self) -> Subscription<ConfigMessage> {
        self.registry_updates
            .as_ref()
            .map(|receiver| {
                from_recipe(RegistrySubscription {
                    receiver: Arc::clone(receiver),
                })
            })
            .unwrap_or_else(Subscription::none)
    }

    pub fn update(&mut self, message: ConfigMessage) -> Task<ConfigMessage> {
        match message {
            ConfigMessage::RegistryUpdated(snapshot) => {
                self.registry_ready = true;
                self.apply_snapshot(snapshot);
            }
            ConfigMessage::ToggleChanged { node_id, enabled } => {
                self.preferences.insert(node_id, enabled);

                if let Some(entry) = self
                    .applications
                    .iter_mut()
                    .find(|entry| entry.node_id == node_id)
                {
                    entry.enabled = enabled;
                }

                if let Err(err) = self
                    .routing_sender
                    .send(RoutingCommand::SetApplicationEnabled { node_id, enabled })
                {
                    tracing::error!("[ui] failed to send routing command: {err}");
                }
            }
            ConfigMessage::ToggleApplicationsVisibility => {
                self.applications_expanded = !self.applications_expanded;
            }
            ConfigMessage::VisualToggled { kind, enabled } => {
                self.visual_manager
                    .borrow_mut()
                    .set_enabled_by_kind(kind, enabled);
            }
        }

        Task::none()
    }

    pub fn view(&self) -> Element<'_, ConfigMessage> {
        let visuals_snapshot = self.visual_manager.snapshot();
        let sink_label = format!("Hardware sink: {}", self.hardware_sink.label());

        let applications_section = self.render_applications_section();
        let visuals_section = self.render_visuals_section(&visuals_snapshot);

        let content = Column::new()
            .spacing(16)
            .push(text(sink_label).size(14))
            .push(applications_section)
            .push(visuals_section);

        container(content)
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }

    fn render_applications_section(&self) -> Column<'_, ConfigMessage> {
        let status_suffix = if self.applications.is_empty() {
            if self.registry_updates.is_some() {
                if self.registry_ready {
                    " - none detected"
                } else {
                    " - waiting..."
                }
            } else {
                " - unavailable"
            }
        } else {
            &format!(" - {} total", self.applications.len())
        };

        let indicator = if self.applications_expanded {
            "▾"
        } else {
            "▸"
        };
        let summary_label = format!("{indicator} Applications{status_suffix}");

        let summary_button = button(
            text(summary_label)
                .width(Length::Fill)
                .align_x(alignment::Horizontal::Left),
        )
        .padding(8)
        .width(Length::Fill)
        .style(header_button_style)
        .on_press(ConfigMessage::ToggleApplicationsVisibility);

        let mut section = Column::new().spacing(8).push(summary_button);

        if self.applications_expanded {
            let content = if self.applications.is_empty() {
                text(self.applications_empty_message()).into()
            } else {
                self.render_applications_grid()
            };
            section = section.push(scrollable(content).height(Length::Shrink));
        }

        section
    }

    fn applications_empty_message(&self) -> &'static str {
        if self.registry_updates.is_some() {
            if self.registry_ready {
                "No audio applications detected. Launch something to see it here."
            } else {
                "Waiting for PipeWire registry snapshots..."
            }
        } else {
            "Registry unavailable; routing controls disabled."
        }
    }

    fn render_applications_grid(&self) -> Element<'_, ConfigMessage> {
        render_toggle_grid(&self.applications, |entry| {
            (
                format!(
                    "{} ({})",
                    entry.display_label(),
                    if entry.enabled { "enabled" } else { "disabled" }
                ),
                entry.enabled,
                ConfigMessage::ToggleChanged {
                    node_id: entry.node_id,
                    enabled: !entry.enabled,
                },
            )
        })
        .into()
    }

    fn render_visuals_section(
        &self,
        snapshot: &crate::ui::visualization::visual_manager::VisualSnapshot,
    ) -> Column<'_, ConfigMessage> {
        let total = snapshot.slots.len();
        let enabled = snapshot.slots.iter().filter(|slot| slot.enabled).count();
        let header = text(format!("Visual modules – {enabled}/{total} enabled")).size(14);

        let section = Column::new().spacing(8).push(header);

        if snapshot.slots.is_empty() {
            return section.push(text("No visual modules available."));
        }

        let grid = render_toggle_grid(&snapshot.slots, |slot| {
            (
                format!(
                    "{} ({})",
                    slot.metadata.display_name,
                    if slot.enabled { "enabled" } else { "disabled" }
                ),
                slot.enabled,
                ConfigMessage::VisualToggled {
                    kind: slot.kind,
                    enabled: !slot.enabled,
                },
            )
        });

        section.push(grid)
    }

    fn apply_snapshot(&mut self, snapshot: RegistrySnapshot) {
        self.hardware_sink.update(&snapshot);

        let mut entries = Vec::new();
        let mut seen = HashSet::new();

        if let Some(sink) = snapshot.find_node_by_label(VIRTUAL_SINK_NAME) {
            for node in snapshot.route_candidates(sink) {
                let enabled = self.preferences.get(&node.id).copied().unwrap_or(true);
                entries.push(ApplicationRow::from_node(node, enabled));
                seen.insert(node.id);
            }
        }

        self.preferences.retain(|node_id, _| seen.contains(node_id));

        entries.sort_by_key(|a| a.sort_key());
        self.applications = entries;
    }
}

fn render_toggle_grid<'a, T, F>(items: &[T], mut project: F) -> Column<'a, ConfigMessage>
where
    F: FnMut(&T) -> (String, bool, ConfigMessage),
{
    let mut grid = Column::new().spacing(12);

    for chunk in items.chunks(GRID_COLUMNS) {
        let mut row = Row::new().spacing(12);

        for item in chunk {
            let (label, enabled, message) = project(item);
            row = row.push(toggle_button(label, enabled, message));
        }

        for _ in chunk.len()..GRID_COLUMNS {
            row = row.push(Space::new(Length::FillPortion(1), Length::Shrink));
        }

        grid = grid.push(row);
    }

    grid
}

fn toggle_button<'a>(
    label: String,
    enabled: bool,
    message: ConfigMessage,
) -> iced::widget::Button<'a, ConfigMessage> {
    button(text(label).width(Length::Fill))
        .padding(12)
        .width(Length::FillPortion(1))
        .style(move |_theme, status| {
            let base_background = if enabled {
                theme::surface_color()
            } else {
                theme::elevated_color()
            };
            let text_color = if enabled {
                theme::text_color()
            } else {
                theme::text_secondary()
            };
            let mut style = iced::widget::button::Style {
                background: Some(iced::Background::Color(base_background)),
                text_color,
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
        })
        .on_press(message)
}

fn header_button_style(
    _theme: &iced::Theme,
    status: iced::widget::button::Status,
) -> iced::widget::button::Style {
    let mut style = iced::widget::button::Style {
        background: Some(iced::Background::Color(theme::surface_color())),
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

struct RegistrySubscription {
    receiver: Arc<AsyncReceiver<RegistrySnapshot>>,
}

impl Recipe for RegistrySubscription {
    type Output = ConfigMessage;

    fn hash(&self, state: &mut Hasher) {
        let ptr = Arc::as_ptr(&self.receiver) as usize;
        state.write(&ptr.to_ne_bytes());
    }

    fn stream(
        self: Box<Self>,
        _input: EventStream,
    ) -> futures::stream::BoxStream<'static, Self::Output> {
        futures::stream::unfold(self.receiver, |receiver| async move {
            match receiver.recv().await {
                Ok(snapshot) => Some((ConfigMessage::RegistryUpdated(snapshot), receiver)),
                Err(_) => None,
            }
        })
        .boxed()
    }
}
