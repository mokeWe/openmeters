use crate::ui::pane_grid::{self, Content as PaneContent, Pane};
use crate::ui::settings::SettingsHandle;
use crate::ui::theme;
use crate::ui::visualization::visual_manager::{
    VisualContent, VisualId, VisualKind, VisualLayoutHint, VisualManagerHandle, VisualMetadata,
    VisualSlotSnapshot, VisualSnapshot,
};
use crate::ui::visualization::{lufs_meter, oscilloscope, spectrogram, spectrum};
mod settings;

use settings::{ActiveSettings, SettingsMessage};

use iced::alignment::{Horizontal, Vertical};
use iced::widget::{button, column, container, row, text};
use iced::{Background, Element, Length, Subscription, Task};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum VisualsMessage {
    PaneDragged(pane_grid::DragEvent),
    PaneContextRequested(Pane),
    CloseSettings,
    Settings(SettingsMessage),
}

#[derive(Debug, Clone)]
struct VisualPane {
    id: VisualId,
    kind: VisualKind,
    metadata: VisualMetadata,
    content: VisualContent,
    layout_hint: VisualLayoutHint,
}

impl VisualPane {
    fn from_snapshot(snapshot: &VisualSlotSnapshot) -> Self {
        Self {
            id: snapshot.id,
            kind: snapshot.kind,
            metadata: snapshot.metadata,
            content: snapshot.content.clone(),
            layout_hint: snapshot.layout_hint,
        }
    }

    fn view(&self) -> PaneContent<'_, VisualsMessage> {
        let content: Element<'_, VisualsMessage> = match &self.content {
            VisualContent::LufsMeter { state } => {
                let meter = lufs_meter::widget_with_layout(
                    state,
                    self.layout_hint.preferred_width,
                    self.layout_hint.preferred_height,
                );
                Element::from(
                    container(meter)
                        .width(Length::Fill)
                        .height(Length::Fill)
                        .align_x(Horizontal::Center)
                        .align_y(Vertical::Bottom),
                )
            }
            VisualContent::Oscilloscope { state } => {
                let scope = oscilloscope::widget(state);
                Element::from(
                    container(scope)
                        .width(Length::Fill)
                        .height(Length::Fill)
                        .center_x(Length::Fill),
                )
            }
            VisualContent::Spectrogram { state } => {
                let spec = spectrogram::widget(state.as_ref());
                Element::from(
                    container(spec)
                        .width(Length::Fill)
                        .height(Length::Fill)
                        .center_x(Length::Fill),
                )
            }
            VisualContent::Spectrum { state } => {
                let spec = spectrum::widget(state);
                Element::from(
                    container(spec)
                        .width(Length::Fill)
                        .height(Length::Fill)
                        .center_x(Length::Fill),
                )
            }
            VisualContent::Placeholder { message } => {
                let placeholder = text(message.as_ref())
                    .size(14)
                    .width(Length::Fill)
                    .align_x(Horizontal::Center);
                Element::from(placeholder)
            }
        };

        let target_width = self.layout_hint.preferred_width;
        let target_height = self.layout_hint.preferred_height;

        let width = if self.layout_hint.fill_horizontal {
            Length::Fill
        } else {
            Length::Fixed(target_width)
        };

        let height = if self.layout_hint.fill_vertical {
            Length::Fill
        } else {
            Length::Fixed(target_height)
        };

        let framed = container(content)
            .width(width)
            .height(height)
            .align_x(Horizontal::Center)
            .align_y(Vertical::Center);

        let element = container(framed)
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(Horizontal::Center)
            .align_y(Vertical::Center);

        PaneContent::new(element).with_width_hint(
            self.layout_hint.min_width,
            self.layout_hint.preferred_width,
            self.layout_hint.max_width,
        )
    }
}

#[derive(Debug)]
pub struct VisualsPage {
    visual_manager: VisualManagerHandle,
    settings: SettingsHandle,
    panes: Option<pane_grid::State<VisualPane>>,
    order: Vec<VisualId>,
    active_settings: Option<ActiveSettings>,
}

impl VisualsPage {
    pub fn new(visual_manager: VisualManagerHandle, settings: SettingsHandle) -> Self {
        let mut page = Self {
            visual_manager,
            settings,
            panes: None,
            order: Vec::new(),
            active_settings: None,
        };
        page.sync_with_manager();
        page
    }

    pub fn subscription(&self) -> Subscription<VisualsMessage> {
        Subscription::none()
    }

    pub fn update(&mut self, message: VisualsMessage) -> Task<VisualsMessage> {
        match message {
            VisualsMessage::PaneDragged(event) => {
                if let Some(panes) = self.panes.as_mut()
                    && let pane_grid::DragEvent::Dropped {
                        pane,
                        target: pane_grid::Target::Pane(target),
                    } = event
                {
                    panes.swap(pane, target);
                    let new_order: Vec<VisualId> = panes.iter().map(|(_, pane)| pane.id).collect();
                    self.order = new_order.clone();
                    self.visual_manager.borrow_mut().reorder(&new_order);

                    let kind_order: Vec<VisualKind> =
                        panes.iter().map(|(_, pane)| pane.kind).collect();
                    self.settings
                        .update(|settings| settings.set_visual_order(&kind_order));
                }
            }
            VisualsMessage::PaneContextRequested(pane) => {
                self.open_settings_for_pane(pane);
            }
            VisualsMessage::CloseSettings => {
                self.active_settings = None;
            }
            VisualsMessage::Settings(message) => {
                if let Some(active) = self.active_settings.as_mut()
                    && active.kind() == message.kind()
                {
                    active.handle_message(&message, &self.visual_manager, &self.settings);
                }
            }
        }

        Task::none()
    }

    pub fn view(&self) -> Element<'_, VisualsMessage> {
        let body: Element<'_, VisualsMessage> = if let Some(panes) = &self.panes {
            let grid = pane_grid::PaneGrid::new(panes, |_, pane_state| pane_state.view())
                .width(Length::Fill)
                .height(Length::Fill)
                .spacing(16.0)
                .on_drag(VisualsMessage::PaneDragged)
                .on_context_request(VisualsMessage::PaneContextRequested);

            container(grid)
                .width(Length::Fill)
                .height(Length::Fill)
                .into()
        } else {
            container(text("enable one or more visual modules to get started"))
                .width(Length::Fill)
                .height(Length::Fill)
                .center_x(Length::Fill)
                .center_y(Length::Fill)
                .into()
        };

        let mut layout = column![body].spacing(16).width(Length::Fill);

        if let Some(active) = &self.active_settings {
            layout = layout.push(self.render_settings_panel(active));
        }

        container(layout)
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }

    pub fn sync_with_manager(&mut self) {
        let snapshot = self.visual_manager.snapshot();
        self.apply_snapshot(snapshot);
    }

    pub fn apply_snapshot(&mut self, snapshot: VisualSnapshot) {
        let enabled_slots: Vec<_> = snapshot.slots.iter().filter(|slot| slot.enabled).collect();
        let new_order: Vec<_> = enabled_slots.iter().map(|slot| slot.id).collect();

        if let Some(active) = &self.active_settings {
            let still_visible = enabled_slots
                .iter()
                .any(|slot| slot.id == active.visual_id());
            if !still_visible {
                self.active_settings = None;
            }
        }

        if enabled_slots.is_empty() {
            self.order.clear();
            self.panes = None;
            self.active_settings = None;
            return;
        }

        if self.panes.is_none() || new_order != self.order {
            self.order = new_order.clone();
            self.panes = Self::build_panes(&enabled_slots);
            if let Some(active) = self.active_settings.as_mut()
                && let Some(slot) = enabled_slots
                    .iter()
                    .find(|slot| slot.id == active.visual_id())
            {
                active.set_title(slot.metadata.display_name.to_string());
            }
            return;
        }

        if let Some(panes) = self.panes.as_mut() {
            let slot_map: HashMap<VisualId, &VisualSlotSnapshot> =
                enabled_slots.iter().map(|slot| (slot.id, *slot)).collect();

            panes.for_each_mut(|_, pane_state| {
                if let Some(slot) = slot_map.get(&pane_state.id) {
                    pane_state.content = slot.content.clone();
                    pane_state.layout_hint = slot.layout_hint;
                }
            });
        }

        if let Some(active) = self.active_settings.as_mut()
            && let Some(slot) = snapshot
                .slots
                .iter()
                .find(|slot| slot.id == active.visual_id())
        {
            active.set_title(slot.metadata.display_name.to_string());
        }
    }

    fn build_panes(slots: &[&VisualSlotSnapshot]) -> Option<pane_grid::State<VisualPane>> {
        let mut iter = slots.iter();
        let first = *iter.next()?;

        let (mut state, mut last_pane) = pane_grid::State::new(VisualPane::from_snapshot(first));

        for slot in iter {
            if let Some(pane) = state.insert_after(last_pane, VisualPane::from_snapshot(slot)) {
                last_pane = pane;
            }
        }

        Some(state)
    }

    fn open_settings_for_pane(&mut self, pane: Pane) {
        let Some(panes) = self.panes.as_ref() else {
            return;
        };

        let Some(pane_state) = panes.get(pane) else {
            return;
        };

        let title = pane_state.metadata.display_name.to_string();
        let panel =
            settings::create_panel(title, pane_state.id, pane_state.kind, &self.visual_manager);
        self.active_settings = Some(panel);
    }

    fn render_settings_panel<'a>(
        &'a self,
        active: &'a ActiveSettings,
    ) -> Element<'a, VisualsMessage> {
        let header = row![
            text(format!("{} settings", active.title()))
                .size(16)
                .width(Length::Fill),
            button(text("Close"))
                .padding([8, 12])
                .style(settings_button_style)
                .on_press(VisualsMessage::CloseSettings)
        ]
        .spacing(12);

        let body: Element<'a, VisualsMessage> = active.view().map(VisualsMessage::Settings);

        container(column![header, body].spacing(16))
            .width(Length::Fill)
            .padding(16)
            .style(settings_panel_style)
            .into()
    }
}

fn settings_button_style(
    _theme: &iced::Theme,
    status: iced::widget::button::Status,
) -> iced::widget::button::Style {
    let mut style = iced::widget::button::Style {
        background: Some(Background::Color(theme::surface_color())),
        text_color: theme::text_color(),
        border: theme::sharp_border(),
        ..Default::default()
    };

    match status {
        iced::widget::button::Status::Hovered => {
            style.background = Some(Background::Color(theme::hover_color()));
        }
        iced::widget::button::Status::Pressed => {
            style.border = theme::focus_border();
        }
        _ => {}
    }

    style
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
