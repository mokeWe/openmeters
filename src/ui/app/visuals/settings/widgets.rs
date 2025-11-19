//! Shared widgets and utilities for the settings panes

use std::borrow::Cow;
use std::fmt;

use iced::alignment::Vertical;
use iced::widget::text::Style as TextStyle;
use iced::widget::{column, pick_list, row, slider, text};

use super::SettingsMessage;

pub const CONTROL_SPACING: f32 = 8.0;
pub const LABEL_SIZE: u16 = 12;
pub const VALUE_SIZE: u16 = 11;
const VALUE_GAP: f32 = 6.0;
pub struct SliderRange {
    pub min: f32,
    pub max: f32,
    pub step: f32,
}

impl SliderRange {
    pub const fn new(min: f32, max: f32, step: f32) -> Self {
        Self { min, max, step }
    }

    #[inline]
    pub fn snap(self, value: f32) -> f32 {
        debug_assert!(self.step > 0.0, "SliderRange::snap expects a positive step");
        if self.step <= 0.0 {
            return value.clamp(self.min, self.max);
        }

        let steps_from_min = ((value - self.min) / self.step).round();
        (self.min + steps_from_min * self.step).clamp(self.min, self.max)
    }
}

// State Update Helpers

#[inline]
pub fn set_if_changed<T>(target: &mut T, value: T) -> bool
where
    T: PartialEq,
{
    if target != &value {
        *target = value;
        true
    } else {
        false
    }
}

#[inline]
pub fn set_f32(target: &mut f32, value: f32) -> bool {
    if (*target).to_bits() != value.to_bits() {
        *target = value;
        true
    } else {
        false
    }
}

#[inline]
pub fn update_f32_range(target: &mut f32, value: f32, range: SliderRange) -> bool {
    set_f32(target, range.snap(value))
}

#[inline]
pub fn update_usize_from_f32(target: &mut usize, value: f32, range: SliderRange) -> bool {
    debug_assert!(
        [range.min, range.max, range.step]
            .into_iter()
            .all(|v| v.fract().abs() <= f32::EPSILON),
        "update_usize_from_f32 expects integral slider bounds"
    );

    let snapped = range.snap(value);
    set_if_changed(target, snapped.round() as usize)
}

// Widget Constructors

pub fn labeled_slider<'a>(
    label: &'static str,
    value: f32,
    formatted: String,
    range: SliderRange,
    on_change: impl Fn(f32) -> SettingsMessage + 'a,
) -> iced::widget::Column<'a, SettingsMessage> {
    let SliderRange { min, max, step } = range;
    column![
        row![
            text(label).size(LABEL_SIZE),
            text(formatted)
                .size(VALUE_SIZE)
                .style(|theme: &iced::Theme| TextStyle {
                    color: Some(theme.extended_palette().secondary.weak.text),
                }),
        ]
        .spacing(VALUE_GAP),
        slider::Slider::new(min..=max, value, on_change).step(step),
    ]
    .spacing(CONTROL_SPACING)
}

pub fn labeled_pick_list<'a, T>(
    label: &'static str,
    options: impl Into<Cow<'a, [T]>>,
    selected: Option<T>,
    on_select: impl Fn(T) -> SettingsMessage + 'a,
) -> iced::widget::Row<'a, SettingsMessage>
where
    T: Clone + PartialEq + fmt::Display + 'static,
{
    row![
        text(label).size(LABEL_SIZE),
        pick_list(options.into(), selected, on_select),
    ]
    .align_y(Vertical::Center)
}
