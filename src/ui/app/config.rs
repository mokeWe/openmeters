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
        let mut subscriptions = Vec::new();

        if let Some(receiver) = &self.registry_updates {
            subscriptions.push(from_recipe(RegistrySubscription {
                receiver: Arc::clone(receiver),
            }));
        }

        match subscriptions.len() {
            0 => Subscription::none(),
            1 => subscriptions.into_iter().next().unwrap(),
            _ => Subscription::batch(subscriptions),
        }
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

        let mut list = Column::new().spacing(12);

        if self.applications.is_empty() {
            let message = if self.registry_updates.is_some() {
                if self.registry_ready {
                    "No audio applications detected. Launch something to see it here."
                } else {
                    "Waiting for PipeWire registry snapshots..."
                }
            } else {
                "Registry unavailable; routing controls disabled."
            };

            list = list.push(text(message));
        } else {
            const GRID_COLUMNS: usize = 2;

            for chunk in self.applications.chunks(GRID_COLUMNS) {
                let mut row = Row::new().spacing(12);

                for entry in chunk {
                    let node_id = entry.node_id;
                    let toggled = !entry.enabled;
                    let label = entry.display_label();
                    let label = if entry.enabled {
                        format!("{label} (enabled)")
                    } else {
                        format!("{label} (disabled)")
                    };
                    let button = button(text(label).width(Length::Fill))
                        .padding(12)
                        .width(Length::FillPortion(1))
                        .style(move |_theme, status| {
                            let mut style = iced::widget::button::Style::default();
                            style.background = Some(iced::Background::Color(if entry.enabled {
                                theme::surface_color()
                            } else {
                                theme::elevated_color()
                            }));
                            style.text_color = if entry.enabled {
                                theme::text_color()
                            } else {
                                theme::text_secondary()
                            };
                            style.border = theme::sharp_border();

                            match status {
                                iced::widget::button::Status::Hovered => {
                                    style.background =
                                        Some(iced::Background::Color(theme::hover_color()));
                                }
                                iced::widget::button::Status::Pressed => {
                                    style.border = theme::focus_border();
                                }
                                _ => {}
                            }

                            style
                        })
                        .on_press(ConfigMessage::ToggleChanged {
                            node_id,
                            enabled: toggled,
                        });
                    row = row.push(button);
                }

                for _ in chunk.len()..GRID_COLUMNS {
                    row = row.push(Space::new(Length::FillPortion(1), Length::Shrink));
                }

                list = list.push(row);
            }
        }

        let summary_status = if self.applications.is_empty() {
            if self.registry_updates.is_some() {
                if self.registry_ready {
                    " - none detected".to_string()
                } else {
                    " - waiting...".to_string()
                }
            } else {
                " - unavailable".to_string()
            }
        } else {
            format!(" - {} total", self.applications.len())
        };

        let indicator = if self.applications_expanded {
            "▾"
        } else {
            "▸"
        };
        let summary_label = format!("{indicator} Applications{summary_status}");

        let summary_button = button(
            text(summary_label)
                .width(Length::Fill)
                .align_x(alignment::Horizontal::Left),
        )
        .padding(8)
        .width(Length::Fill)
        .style(|_theme, status| {
            let mut style = iced::widget::button::Style::default();
            style.background = Some(iced::Background::Color(theme::surface_color()));
            style.text_color = theme::text_color();
            style.border = theme::sharp_border();

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
        .on_press(ConfigMessage::ToggleApplicationsVisibility);

        let mut applications_section = Column::new().spacing(8).push(summary_button);

        if self.applications_expanded {
            applications_section =
                applications_section.push(scrollable(list).height(Length::Shrink));
        }

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

    fn render_visuals_section(
        &self,
        snapshot: &crate::ui::visualization::visual_manager::VisualSnapshot,
    ) -> Column<'_, ConfigMessage> {
        let mut section = Column::new().spacing(8);
        let total = snapshot.slots.len();
        let enabled = snapshot.slots.iter().filter(|slot| slot.enabled).count();

        section =
            section.push(text(format!("Visual modules – {enabled}/{total} enabled")).size(14));

        if snapshot.slots.is_empty() {
            return section
                .push(text("No visual modules available."))
                .spacing(8);
        }

        let mut grid = Column::new().spacing(12);
        const GRID_COLUMNS: usize = 2;

        for chunk in snapshot.slots.chunks(GRID_COLUMNS) {
            let mut row = Row::new().spacing(12);

            for slot in chunk {
                let kind = slot.kind;
                let enabled = slot.enabled;
                let toggled = !enabled;
                let display_name = slot.metadata.display_name;

                let label = if enabled {
                    format!("{display_name} (enabled)")
                } else {
                    format!("{display_name} (disabled)")
                };

                let button = button(text(label).width(Length::Fill))
                    .padding(12)
                    .width(Length::FillPortion(1))
                    .style(move |_theme, status| {
                        let mut style = iced::widget::button::Style::default();
                        style.background = Some(iced::Background::Color(if enabled {
                            theme::surface_color()
                        } else {
                            theme::elevated_color()
                        }));
                        style.text_color = if enabled {
                            theme::text_color()
                        } else {
                            theme::text_secondary()
                        };
                        style.border = theme::sharp_border();

                        match status {
                            iced::widget::button::Status::Hovered => {
                                style.background =
                                    Some(iced::Background::Color(theme::hover_color()));
                            }
                            iced::widget::button::Status::Pressed => {
                                style.border = theme::focus_border();
                            }
                            _ => {}
                        }

                        style
                    })
                    .on_press(ConfigMessage::VisualToggled {
                        kind,
                        enabled: toggled,
                    });

                row = row.push(button);
            }

            for _ in chunk.len()..GRID_COLUMNS {
                row = row.push(Space::new(Length::FillPortion(1), Length::Shrink));
            }

            grid = grid.push(row);
        }

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
