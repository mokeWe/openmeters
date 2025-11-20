//! Configuration page for application and visual settings.

use crate::audio::VIRTUAL_SINK_NAME;
use crate::audio::pw_registry::RegistrySnapshot;
use crate::ui::app::visuals::settings::palette::{PaletteEditor, PaletteEvent};
use crate::ui::application_row::ApplicationRow;
use crate::ui::channel_subscription::channel_subscription;
use crate::ui::hardware_sink::HardwareSinkCache;
use crate::ui::settings::SettingsHandle;
use crate::ui::theme;
use crate::ui::visualization::visual_manager::{VisualKind, VisualManagerHandle};
use async_channel::Receiver as AsyncReceiver;
use iced::alignment;
use iced::widget::text::Style as TextStyle;
use iced::widget::{
    Column, Row, Rule, Space, button, container, pick_list, radio, rule, scrollable, text,
};
use iced::{Element, Length, Subscription, Task};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, mpsc};

const GRID_COLUMNS: usize = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
struct DeviceOption(String, DeviceSelection);

impl std::fmt::Display for DeviceOption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CaptureMode {
    #[default]
    Applications,
    Device,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DeviceSelection {
    #[default]
    Default,
    Node(u32),
}

#[derive(Debug, Clone)]
pub enum RoutingCommand {
    SetApplicationEnabled { node_id: u32, enabled: bool },
    SetCaptureMode(CaptureMode),
    SelectCaptureDevice(DeviceSelection),
}

#[derive(Debug, Clone)]
pub enum ConfigMessage {
    RegistryUpdated(RegistrySnapshot),
    ToggleChanged { node_id: u32, enabled: bool },
    ToggleApplicationsVisibility,
    VisualToggled { kind: VisualKind, enabled: bool },
    CaptureModeChanged(CaptureMode),
    CaptureDeviceChanged(DeviceSelection),
    BgPalette(PaletteEvent),
    DecorationsToggled(bool),
}

#[derive(Debug)]
pub struct ConfigPage {
    routing_sender: mpsc::Sender<RoutingCommand>,
    registry_updates: Option<Arc<AsyncReceiver<RegistrySnapshot>>>,
    visual_manager: VisualManagerHandle,
    settings: SettingsHandle,
    preferences: HashMap<u32, bool>,
    applications: Vec<ApplicationRow>,
    hardware_sink: HardwareSinkCache,
    registry_ready: bool,
    applications_expanded: bool,
    capture_mode: CaptureMode,
    device_choices: Vec<DeviceOption>,
    selected_device: DeviceSelection,
    bg_palette: PaletteEditor,
}

impl ConfigPage {
    pub fn new(
        routing_sender: mpsc::Sender<RoutingCommand>,
        registry_updates: Option<Arc<AsyncReceiver<RegistrySnapshot>>>,
        visual_manager: VisualManagerHandle,
        settings: SettingsHandle,
    ) -> Self {
        let current_bg = settings
            .borrow()
            .settings()
            .background_color
            .map(|c| c.to_color())
            .unwrap_or(theme::BG_BASE);
        let defaults = [theme::BG_BASE];
        let bg_palette = PaletteEditor::new(&[current_bg], &defaults);

        let ret = Self {
            routing_sender,
            registry_updates,
            visual_manager,
            settings,
            preferences: HashMap::new(),
            applications: Vec::new(),
            hardware_sink: HardwareSinkCache::new(),
            registry_ready: false,
            applications_expanded: false,
            capture_mode: CaptureMode::Applications,
            device_choices: Vec::new(),
            selected_device: DeviceSelection::Default,
            bg_palette,
        };
        ret.dispatch_capture_state();
        ret
    }

    pub fn subscription(&self) -> Subscription<ConfigMessage> {
        self.registry_updates
            .as_ref()
            .map(|receiver| {
                channel_subscription(Arc::clone(receiver)).map(ConfigMessage::RegistryUpdated)
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
                self.settings
                    .update(|settings| settings.set_visual_enabled(kind, enabled));
            }
            ConfigMessage::CaptureModeChanged(mode) => {
                if self.capture_mode != mode {
                    self.capture_mode = mode;
                    self.dispatch_capture_state();
                }
            }
            ConfigMessage::CaptureDeviceChanged(selection) => {
                if self.selected_device != selection {
                    self.selected_device = selection;
                    self.dispatch_capture_state();
                }
            }
            ConfigMessage::BgPalette(event) => {
                if self.bg_palette.update(event) {
                    let color = self.bg_palette.colors().first().copied();
                    self.settings.update(|s| s.set_background_color(color));
                }
            }
            ConfigMessage::DecorationsToggled(enabled) => {
                self.settings.update(|s| s.set_decorations(enabled));
            }
        }

        Task::none()
    }

    pub fn view(&self) -> Element<'_, ConfigMessage> {
        let visuals_snapshot = self.visual_manager.snapshot();

        let capture_section = self.render_capture_section();
        let visuals_section = self.render_visuals_section(&visuals_snapshot);
        let bg_section = self.render_bg_section();

        let content = Column::new()
            .spacing(20)
            .push(capture_section)
            .push(self.divider())
            .push(visuals_section)
            .push(self.divider())
            .push(bg_section);

        container(scrollable(content))
            .width(Length::Fill)
            .height(Length::Fill)
            .padding(16)
            .into()
    }

    fn render_capture_section(&self) -> Column<'_, ConfigMessage> {
        let status_label = self.capture_status_label();
        let capture_controls = self.render_capture_mode_controls();
        let primary_section: Element<'_, ConfigMessage> = match self.capture_mode {
            CaptureMode::Applications => self.render_applications_section().into(),
            CaptureMode::Device => self.render_device_section().into(),
        };

        let content = Column::new()
            .spacing(12)
            .push(
                text(status_label)
                    .size(12)
                    .style(|theme: &iced::Theme| TextStyle {
                        color: Some(theme.extended_palette().secondary.weak.text),
                    }),
            )
            .push(capture_controls)
            .push(primary_section);

        self.section("Audio Capture", content)
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
                let msg = if self.registry_updates.is_none() {
                    "Registry unavailable; routing controls disabled."
                } else if self.registry_ready {
                    "No audio applications detected. Launch something to see it here."
                } else {
                    "Waiting for PipeWire registry snapshots..."
                };
                text(msg).into()
            } else {
                self.render_applications_grid()
            };
            section = section.push(scrollable(content).height(Length::Shrink));
        }

        section
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

    fn capture_status_label(&self) -> String {
        match self.capture_mode {
            CaptureMode::Applications => {
                format!("Default hardware sink: {}", self.hardware_sink.label())
            }
            CaptureMode::Device => format!("Capturing from: {}", self.selected_device_label()),
        }
    }

    fn render_capture_mode_controls(&self) -> Row<'_, ConfigMessage> {
        Row::new()
            .spacing(12)
            .push(radio(
                "Applications",
                CaptureMode::Applications,
                Some(self.capture_mode),
                ConfigMessage::CaptureModeChanged,
            ))
            .push(radio(
                "Devices",
                CaptureMode::Device,
                Some(self.capture_mode),
                ConfigMessage::CaptureModeChanged,
            ))
    }

    fn render_device_section(&self) -> Column<'_, ConfigMessage> {
        let selected = self
            .device_choices
            .iter()
            .find(|opt| opt.1 == self.selected_device)
            .cloned();
        let mut picker = pick_list(
            self.device_choices.clone(),
            selected,
            |opt: DeviceOption| ConfigMessage::CaptureDeviceChanged(opt.1),
        );
        if self.device_choices.len() <= 1 {
            picker = picker.placeholder("No devices available");
        }

        Column::new().spacing(8).push(picker).push(
            text("Direct device capture. Application routing disabled.")
                .size(12)
                .style(|theme: &iced::Theme| TextStyle {
                    color: Some(theme.extended_palette().secondary.weak.text),
                }),
        )
    }

    fn selected_device_label(&self) -> String {
        self.device_choices
            .iter()
            .find(|opt| opt.1 == self.selected_device)
            .map(|opt| opt.0.clone())
            .unwrap_or_else(|| match self.selected_device {
                DeviceSelection::Default => format!("Default ({})", self.hardware_sink.label()),
                DeviceSelection::Node(id) => format!("Node #{id}"),
            })
    }

    fn build_device_choices(&self, snapshot: &RegistrySnapshot) -> Vec<DeviceOption> {
        let mut choices = Vec::new();

        // Use the cached hardware sink label which has fallback to last known value
        let default_label = format!("Default sink - {}", self.hardware_sink.label());
        choices.push(DeviceOption(default_label, DeviceSelection::Default));

        let mut nodes: Vec<_> = snapshot
            .nodes
            .iter()
            .filter(|node| Self::is_capture_candidate(node))
            .map(|node| (node.display_name(), node.id))
            .collect();
        nodes.sort_by(|a, b| a.0.to_ascii_lowercase().cmp(&b.0.to_ascii_lowercase()));

        for (label, id) in nodes {
            choices.push(DeviceOption(label, DeviceSelection::Node(id)));
        }
        choices
    }

    fn is_capture_candidate(node: &crate::audio::pw_registry::NodeInfo) -> bool {
        if node.is_virtual || node.app_name().is_some() {
            return false;
        }

        let contains = |opt: Option<&String>, pattern: &str| {
            opt.is_some_and(|s| s.to_ascii_lowercase().contains(pattern))
        };

        contains(node.media_class.as_ref(), "audio")
            || contains(node.name.as_ref(), "monitor")
            || contains(node.description.as_ref(), "monitor")
    }

    fn render_bg_section(&self) -> Column<'_, ConfigMessage> {
        let content = self.bg_palette.view().map(ConfigMessage::BgPalette);
        let decorations_enabled = self.settings.borrow().settings().decorations;

        let decorations_toggle =
            iced::widget::checkbox("Enable Window Decorations", decorations_enabled)
                .on_toggle(ConfigMessage::DecorationsToggled);

        self.section(
            "Global",
            Column::new()
                .spacing(12)
                .push(content)
                .push(decorations_toggle),
        )
    }

    fn render_visuals_section(
        &self,
        snapshot: &crate::ui::visualization::visual_manager::VisualSnapshot,
    ) -> Column<'_, ConfigMessage> {
        let total = snapshot.slots.len();
        let enabled = snapshot.slots.iter().filter(|slot| slot.enabled).count();
        let title = format!("Visual Modules ({enabled}/{total})");

        let content: Element<'_, ConfigMessage> = if snapshot.slots.is_empty() {
            text("No visual modules available.").into()
        } else {
            render_toggle_grid(&snapshot.slots, |slot| {
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
            })
            .into()
        };

        self.section(&title, content)
    }

    fn section<'a>(
        &self,
        title: impl Into<String>,
        content: impl Into<Element<'a, ConfigMessage>>,
    ) -> Column<'a, ConfigMessage> {
        Column::new()
            .spacing(12)
            .push(text(title.into()).size(14))
            .push(content)
    }

    fn divider<'a>(&self) -> Rule<'a> {
        Rule::horizontal(1).style(|theme: &iced::Theme| rule::Style {
            color: theme::with_alpha(theme.extended_palette().secondary.weak.text, 0.2),
            width: 1,
            radius: 0.0.into(),
            fill_mode: rule::FillMode::Percent(100.0),
        })
    }

    fn apply_snapshot(&mut self, snapshot: RegistrySnapshot) {
        self.hardware_sink.update(&snapshot);
        let choices = self.build_device_choices(&snapshot);
        if !choices.iter().any(|opt| opt.1 == self.selected_device) {
            self.selected_device = DeviceSelection::Default;
            self.dispatch_capture_state();
        }
        self.device_choices = choices;

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

impl ConfigPage {
    fn dispatch_capture_state(&self) {
        let _ = self
            .routing_sender
            .send(RoutingCommand::SetCaptureMode(self.capture_mode));
        let _ = self
            .routing_sender
            .send(RoutingCommand::SelectCaptureDevice(self.selected_device));
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
        .style(move |theme: &iced::Theme, status| {
            let palette = theme.extended_palette();
            let base_background = if enabled {
                palette.background.weak.color
            } else {
                palette.background.strong.color
            };

            let mut style = theme::button_style(theme, base_background, status);

            if !enabled {
                style.text_color = palette.secondary.weak.text;
            }

            style
        })
        .on_press(message)
}

fn header_button_style(
    theme: &iced::Theme,
    status: iced::widget::button::Status,
) -> iced::widget::button::Style {
    theme::surface_button_style(theme, status)
}
