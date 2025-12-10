//! Color palette editor.

use crate::ui::theme;
use iced::alignment::{Horizontal, Vertical};
use iced::widget::text::Wrapping;
use iced::widget::{Button, Column, Row, Space, container, slider, text};
use iced::{Background, Color, Element, Length};

const SWATCH_SIZE: (f32, f32) = (56.0, 28.0);
const LABEL_SIZE: u32 = 11;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PaletteEvent {
    Open(usize),
    Close,
    Adjust { index: usize, color: Color },
    Reset,
}

#[derive(Debug, Clone)]
pub struct PaletteEditor {
    colors: Vec<Color>,
    defaults: Vec<Color>,
    labels: Vec<&'static str>,
    active: Option<usize>,
}

impl PaletteEditor {
    pub fn new(current: &[Color], defaults: &[Color]) -> Self {
        Self::with_labels(current, defaults, &[])
    }

    pub fn with_labels(current: &[Color], defaults: &[Color], labels: &[&'static str]) -> Self {
        Self {
            colors: if current.len() == defaults.len() {
                current.to_vec()
            } else {
                defaults.to_vec()
            },
            defaults: defaults.to_vec(),
            labels: labels.to_vec(),
            active: None,
        }
    }

    fn label_for(&self, index: usize) -> String {
        if let Some(&label) = self.labels.get(index) {
            label.to_string()
        } else {
            format!("Color {}", index + 1)
        }
    }

    pub fn update(&mut self, event: PaletteEvent) -> bool {
        match event {
            PaletteEvent::Open(i) if i < self.colors.len() => {
                self.active = if self.active == Some(i) {
                    None
                } else {
                    Some(i)
                };
                false
            }
            PaletteEvent::Close => {
                self.active = None;
                false
            }
            PaletteEvent::Adjust { index, color } => {
                if let Some(s) = self.colors.get_mut(index)
                    && !theme::colors_equal(*s, color)
                {
                    *s = color;
                    true
                } else {
                    false
                }
            }
            PaletteEvent::Reset => {
                self.active = None;
                if !self.is_default() {
                    self.colors.clone_from(&self.defaults);
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    pub fn colors(&self) -> &[Color] {
        &self.colors
    }

    pub fn is_default(&self) -> bool {
        self.colors.len() == self.defaults.len()
            && self
                .colors
                .iter()
                .zip(&self.defaults)
                .all(|(c, d)| theme::colors_equal(*c, *d))
    }

    pub fn view(&self) -> Element<'_, PaletteEvent> {
        let mut row = Row::new().spacing(12.0);
        for (i, &c) in self.colors.iter().enumerate() {
            row = row.push(self.color_picker(i, c));
        }

        let mut col = Column::new().spacing(12.0).push(row);
        if let Some(i) = self.active
            && let Some(&c) = self.colors.get(i)
        {
            col = col.push(self.color_controls(i, c));
        }

        col.push(
            Button::new(
                container(text("Reset to defaults").size(12).wrapping(Wrapping::None)).clip(true),
            )
            .padding([6, 10])
            .style(|theme, status| theme::tab_button_style(theme, false, status))
            .on_press_maybe((!self.is_default()).then_some(PaletteEvent::Reset)),
        )
        .into()
    }

    fn color_picker(&self, i: usize, c: Color) -> Element<'_, PaletteEvent> {
        let (w, h) = SWATCH_SIZE;
        let is_active = self.active == Some(i);
        let label = self.label_for(i);
        Button::new(
            Column::new()
                .width(Length::Shrink)
                .spacing(4.0)
                .align_x(Horizontal::Center)
                .push(container(text(label).size(LABEL_SIZE).wrapping(Wrapping::None)).clip(true))
                .push(
                    container(Space::new().width(Length::Fill).height(Length::Fill))
                        .width(Length::Fixed(w))
                        .height(Length::Fixed(h))
                        .style(move |_| swatch_style(c, is_active)),
                )
                .push(
                    container(text(to_hex(c)).size(LABEL_SIZE).wrapping(Wrapping::None)).clip(true),
                ),
        )
        .padding([6, 8])
        .style(|theme, status| theme::tab_button_style(theme, false, status))
        .on_press(PaletteEvent::Open(i))
        .into()
    }

    fn color_controls(&self, i: usize, c: Color) -> Element<'_, PaletteEvent> {
        let label = self.label_for(i);
        let mut col = Column::new().spacing(8.0).push(
            Row::new()
                .spacing(8.0)
                .align_y(Vertical::Center)
                .push(container(text(label).size(12).wrapping(Wrapping::None)).clip(true))
                .push(Space::new().width(Length::Fill).height(Length::Shrink))
                .push(
                    Button::new(
                        container(text("Done").size(12).wrapping(Wrapping::None)).clip(true),
                    )
                    .padding([6, 10])
                    .style(|theme, status| theme::tab_button_style(theme, false, status))
                    .on_press(PaletteEvent::Close),
                ),
        );

        // RGB+A channels
        for (channel_label, value, setter, display_fn) in [
            (
                "R",
                c.r,
                set_r as fn(Color, f32) -> Color,
                rgb_display as fn(f32) -> String,
            ),
            (
                "G",
                c.g,
                set_g as fn(Color, f32) -> Color,
                rgb_display as fn(f32) -> String,
            ),
            (
                "B",
                c.b,
                set_b as fn(Color, f32) -> Color,
                rgb_display as fn(f32) -> String,
            ),
            (
                "A",
                c.a,
                set_a as fn(Color, f32) -> Color,
                alpha_display as fn(f32) -> String,
            ),
        ] {
            col = col.push(
                Row::new()
                    .spacing(8.0)
                    .align_y(Vertical::Center)
                    .push(
                        container(text(channel_label).size(12).wrapping(Wrapping::None))
                            .width(Length::Fixed(32.0))
                            .clip(true),
                    )
                    .push(
                        slider::Slider::new(0.0..=1.0, value, move |nv| PaletteEvent::Adjust {
                            index: i,
                            color: setter(c, nv),
                        })
                        .step(0.01)
                        .style(theme::slider_style)
                        .width(Length::Fill),
                    )
                    .push(
                        container(text(display_fn(value)).size(12).wrapping(Wrapping::None))
                            .clip(true),
                    ),
            );
        }

        container(col)
            .padding(12)
            .style(theme::weak_container)
            .into()
    }
}

fn swatch_style(color: Color, active: bool) -> container::Style {
    // Premultiply alpha for the swatch display to match the window renderer
    let display_color = Color {
        r: color.r * color.a,
        g: color.g * color.a,
        b: color.b * color.a,
        a: color.a,
    };

    container::Style::default()
        .background(Background::Color(display_color))
        .border(if active {
            theme::focus_border()
        } else {
            theme::sharp_border()
        })
}

fn to_hex(c: Color) -> String {
    let r = (c.r.clamp(0.0, 1.0) * 255.0).round() as u8;
    let g = (c.g.clamp(0.0, 1.0) * 255.0).round() as u8;
    let b = (c.b.clamp(0.0, 1.0) * 255.0).round() as u8;
    let a = (c.a.clamp(0.0, 1.0) * 255.0).round() as u8;

    if a == 255 {
        format!("#{r:02X}{g:02X}{b:02X}")
    } else {
        format!("#{r:02X}{g:02X}{b:02X}{a:02X}")
    }
}

#[inline]
fn rgb_display(v: f32) -> String {
    format!("{:>3}", (v.clamp(0.0, 1.0) * 255.0).round() as u8)
}

#[inline]
fn alpha_display(v: f32) -> String {
    format!("{:>3}%", (v.clamp(0.0, 1.0) * 100.0).round() as u8)
}

fn set_r(mut c: Color, v: f32) -> Color {
    c.r = v;
    c
}
fn set_g(mut c: Color, v: f32) -> Color {
    c.g = v;
    c
}
fn set_b(mut c: Color, v: f32) -> Color {
    c.b = v;
    c
}
fn set_a(mut c: Color, v: f32) -> Color {
    // Snap to 0.0 if very close to ensure clean transparency
    c.a = if v < 0.005 { 0.0 } else { v };
    c
}
