use crate::ui::pane_grid::{self, Content as PaneContent, Pane};
use crate::ui::settings::SettingsHandle;
use crate::ui::visualization::visual_manager::{
    VisualContent, VisualId, VisualKind, VisualManagerHandle, VisualMetadata, VisualSlotSnapshot,
    VisualSnapshot,
};
pub mod settings;
pub use settings::{ActiveSettings, SettingsMessage, create_panel as create_settings_panel};

use iced::widget::{container, mouse_area, text};
use iced::{Element, Length, Subscription, Task};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum VisualsMessage {
    PaneDragged(pane_grid::DragEvent),
    PaneContextRequested(Pane),
    PaneHovered(Option<Pane>),
    SettingsRequested {
        visual_id: VisualId,
        kind: VisualKind,
    },
    WindowDragRequested,
}

#[derive(Debug, Clone)]
struct VisualPane {
    id: VisualId,
    kind: VisualKind,
    metadata: VisualMetadata,
    content: VisualContent,
}

impl VisualPane {
    fn from_snapshot(s: &VisualSlotSnapshot) -> Self {
        Self {
            id: s.id,
            kind: s.kind,
            metadata: s.metadata,
            content: s.content.clone(),
        }
    }
    fn view(&self) -> PaneContent<'_, VisualsMessage> {
        PaneContent::new(self.content.render(self.metadata)).with_width_hint(
            self.metadata.min_width,
            self.metadata.preferred_width,
            self.metadata.max_width,
        )
    }
}

#[derive(Debug)]
pub struct VisualsPage {
    visual_manager: VisualManagerHandle,
    settings: SettingsHandle,
    panes: Option<pane_grid::State<VisualPane>>,
    order: Vec<VisualId>,
    hovered_pane: Option<Pane>,
}

impl VisualsPage {
    pub fn new(visual_manager: VisualManagerHandle, settings: SettingsHandle) -> Self {
        let mut page = Self {
            visual_manager,
            settings,
            panes: None,
            order: Vec::new(),
            hovered_pane: None,
        };
        page.sync_with_manager();
        page
    }

    pub fn subscription(&self) -> Subscription<VisualsMessage> {
        Subscription::none()
    }

    pub fn update(&mut self, message: VisualsMessage) -> Task<VisualsMessage> {
        match message {
            VisualsMessage::PaneDragged(pane_grid::DragEvent::Dropped {
                pane,
                target: pane_grid::Target::Pane(target),
            }) => {
                if let Some(panes) = self.panes.as_mut() {
                    panes.swap(pane, target);
                    self.order = panes.iter().map(|(_, p)| p.id).collect();
                    self.visual_manager.borrow_mut().reorder(&self.order);
                    let kinds: Vec<_> = panes.iter().map(|(_, p)| p.kind).collect();
                    self.settings.update(|s| s.set_visual_order(&kinds));
                }
            }
            VisualsMessage::PaneDragged(_) => {}
            VisualsMessage::PaneContextRequested(pane) => {
                if let Some(p) = self.panes.as_ref().and_then(|ps| ps.get(pane)) {
                    return Task::done(VisualsMessage::SettingsRequested {
                        visual_id: p.id,
                        kind: p.kind,
                    });
                }
            }
            VisualsMessage::PaneHovered(pane) => self.hovered_pane = pane,
            VisualsMessage::SettingsRequested { .. } | VisualsMessage::WindowDragRequested => {}
        }
        Task::none()
    }

    pub fn hovered_visual(&self) -> Option<(VisualId, VisualKind)> {
        self.panes
            .as_ref()?
            .get(self.hovered_pane?)
            .map(|p| (p.id, p.kind))
    }

    pub fn view(&self, controls_visible: bool) -> Element<'_, VisualsMessage> {
        let spacing = if controls_visible { 16.0 } else { 0.0 };
        let Some(panes) = &self.panes else {
            return container(text("enable one or more visual modules to get started"))
                .width(Length::Fill)
                .height(Length::Fill)
                .center_x(Length::Fill)
                .center_y(Length::Fill)
                .into();
        };

        let mut grid = pane_grid::PaneGrid::new(panes, |_, p| p.view())
            .width(Length::Fill)
            .height(Length::Fill)
            .spacing(spacing)
            .on_context_request(VisualsMessage::PaneContextRequested)
            .on_hover(VisualsMessage::PaneHovered);

        if controls_visible {
            grid = grid.on_drag(VisualsMessage::PaneDragged);
            container(grid)
                .width(Length::Fill)
                .height(Length::Fill)
                .into()
        } else {
            mouse_area(container(grid).width(Length::Fill).height(Length::Fill))
                .on_press(VisualsMessage::WindowDragRequested)
                .interaction(iced::mouse::Interaction::Grab)
                .into()
        }
    }

    pub fn sync_with_manager(&mut self) {
        self.apply_snapshot_excluding(self.visual_manager.snapshot(), &[]);
    }

    pub fn apply_snapshot_excluding(&mut self, snapshot: VisualSnapshot, exclude: &[VisualId]) {
        let exclude_set: std::collections::HashSet<_> = exclude.iter().copied().collect();
        let slots: Vec<_> = snapshot
            .slots
            .iter()
            .filter(|s| s.enabled && !exclude_set.contains(&s.id))
            .collect();
        let new_order: Vec<_> = slots.iter().map(|s| s.id).collect();

        if slots.is_empty() {
            self.order.clear();
            self.panes = None;
            return;
        }
        if self.panes.is_none() || new_order != self.order {
            self.order = new_order;
            self.panes = Self::build_panes(&slots);
            return;
        }
        if let Some(panes) = self.panes.as_mut() {
            let map: HashMap<_, _> = slots.iter().map(|s| (s.id, *s)).collect();
            panes.for_each_mut(|_, p| {
                if let Some(s) = map.get(&p.id) {
                    p.content = s.content.clone();
                    p.metadata = s.metadata;
                }
            });
        }
    }

    fn build_panes(slots: &[&VisualSlotSnapshot]) -> Option<pane_grid::State<VisualPane>> {
        let (first, rest) = slots.split_first()?;
        let (mut state, mut last) = pane_grid::State::new(VisualPane::from_snapshot(first));
        for s in rest {
            if let Some(p) = state.insert_after(last, VisualPane::from_snapshot(s)) {
                last = p;
            }
        }
        Some(state)
    }
}
