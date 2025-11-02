//! Monochrome Iced theme.

use iced::border::Border;
use iced::theme::palette::{self, Extended, Pair};
use iced::widget::button;
use iced::{Background, Color, Theme};

// Core palette stops
const BG_BASE: Color = Color::from_rgb(0.059, 0.063, 0.071);
const BG_SURFACE: Color = Color::from_rgb(0.090, 0.094, 0.106);
const BG_ELEVATED: Color = Color::from_rgb(0.122, 0.129, 0.145);
const BG_HOVER: Color = Color::from_rgb(0.157, 0.165, 0.184);

const TEXT_PRIMARY: Color = Color::from_rgb(0.902, 0.910, 0.925);
const TEXT_SECONDARY: Color = Color::from_rgb(0.655, 0.671, 0.698);

const BORDER_SUBTLE: Color = Color::from_rgb(0.196, 0.204, 0.224);
const BORDER_FOCUS: Color = Color::from_rgb(0.416, 0.424, 0.443);

// Accent colors
const ACCENT_PRIMARY: Color = Color::from_rgb(0.529, 0.549, 0.584);
const ACCENT_SUCCESS: Color = Color::from_rgb(0.478, 0.557, 0.502);
const ACCENT_DANGER: Color = Color::from_rgb(0.557, 0.478, 0.478);

pub const DEFAULT_SPECTROGRAM_PALETTE: [Color; 5] = [
    Color::from_rgba(0.000, 0.000, 0.000, 0.0),
    Color::from_rgba(0.039, 0.011, 0.09, 1.0),
    Color::from_rgba(0.329, 0.000, 0.000, 1.0),
    Color::from_rgba(1.000, 0.502, 0.102, 1.0),
    Color::from_rgba(1.000, 1.000, 1.000, 1.0),
];

pub const DEFAULT_SPECTRUM_PALETTE: [Color; 5] = [
    Color::from_rgba(0.039, 0.011, 0.09, 1.0), // #0A0317 - dark purple (quiet)
    Color::from_rgba(0.329, 0.000, 0.000, 1.0), // #540000 - dark red
    Color::from_rgba(0.800, 0.200, 0.000, 1.0), // #CC3300 - red-orange
    Color::from_rgba(1.000, 0.502, 0.102, 1.0), // #FF801A - orange
    Color::from_rgba(1.000, 1.000, 0.000, 1.0), // #FFFF00 - yellow (loud)
];

pub const DEFAULT_WAVEFORM_PALETTE: [Color; 5] = [
    Color::from_rgb(0.400, 0.000, 0.000), // dark red
    Color::from_rgb(1.000, 0.000, 0.000), // bright red
    Color::from_rgb(1.000, 0.400, 0.000), // orange
    Color::from_rgb(0.600, 0.200, 0.800), // purple-magenta
    Color::from_rgb(0.400, 0.000, 0.800), // deep purple
];

pub const DEFAULT_OSCILLOSCOPE_PALETTE: [Color; 2] = [
    Color::from_rgb(0.529, 0.549, 0.584), // left channel - primary accent
    Color::from_rgb(0.529, 0.549, 0.584), // right channel - same
];

pub fn theme() -> Theme {
    Theme::custom_with_fn(
        "OpenMeters Monochrome".to_string(),
        palette(),
        extended_palette,
    )
}

fn palette() -> palette::Palette {
    palette::Palette {
        background: BG_BASE,
        text: TEXT_PRIMARY,
        primary: ACCENT_PRIMARY,
        success: ACCENT_SUCCESS,
        danger: ACCENT_DANGER,
    }
}

fn extended_palette(base: palette::Palette) -> Extended {
    Extended {
        background: palette::Background {
            base: Pair::new(base.background, base.text),
            weak: Pair::new(BG_SURFACE, base.text),
            strong: Pair::new(BG_ELEVATED, base.text),
        },
        primary: palette::Primary {
            base: Pair::new(base.primary, TEXT_PRIMARY),
            weak: Pair::new(
                Color::new(
                    base.primary.r * 0.7,
                    base.primary.g * 0.7,
                    base.primary.b * 0.7,
                    1.0,
                ),
                TEXT_SECONDARY,
            ),
            strong: Pair::new(
                Color::new(
                    base.primary.r * 1.2,
                    base.primary.g * 1.2,
                    base.primary.b * 1.2,
                    1.0,
                ),
                TEXT_PRIMARY,
            ),
        },
        secondary: palette::Secondary {
            base: Pair::new(BG_SURFACE, base.text),
            weak: Pair::new(BG_BASE, TEXT_SECONDARY),
            strong: Pair::new(BG_ELEVATED, base.text),
        },
        success: palette::Success {
            base: Pair::new(base.success, TEXT_PRIMARY),
            weak: Pair::new(
                Color::new(
                    base.success.r * 0.7,
                    base.success.g * 0.7,
                    base.success.b * 0.7,
                    1.0,
                ),
                TEXT_SECONDARY,
            ),
            strong: Pair::new(
                Color::new(
                    base.success.r * 1.2,
                    base.success.g * 1.2,
                    base.success.b * 1.2,
                    1.0,
                ),
                TEXT_PRIMARY,
            ),
        },
        danger: palette::Danger {
            base: Pair::new(base.danger, TEXT_PRIMARY),
            weak: Pair::new(
                Color::new(
                    base.danger.r * 0.7,
                    base.danger.g * 0.7,
                    base.danger.b * 0.7,
                    1.0,
                ),
                TEXT_SECONDARY,
            ),
            strong: Pair::new(
                Color::new(
                    base.danger.r * 1.2,
                    base.danger.g * 1.2,
                    base.danger.b * 1.2,
                    1.0,
                ),
                TEXT_PRIMARY,
            ),
        },
        is_dark: true,
    }
}

// styling helpers

/// Standard sharp border for buttons and containers.
pub fn sharp_border() -> Border {
    Border {
        color: BORDER_SUBTLE,
        width: 1.0,
        radius: 0.0.into(),
    }
}

pub fn focus_border() -> Border {
    Border {
        color: BORDER_FOCUS,
        width: 1.0,
        radius: 0.0.into(),
    }
}

pub fn button_style(base: Color, status: button::Status) -> button::Style {
    let mut style = button::Style {
        background: Some(Background::Color(base)),
        text_color: text_color(),
        border: sharp_border(),
        ..Default::default()
    };

    match status {
        button::Status::Hovered => {
            style.background = Some(Background::Color(hover_color()));
        }
        button::Status::Pressed => {
            style.border = focus_border();
        }
        _ => {}
    }

    style
}

pub fn surface_button_style(status: button::Status) -> button::Style {
    button_style(surface_color(), status)
}

pub fn surface_color() -> Color {
    BG_SURFACE
}

pub fn base_color() -> Color {
    BG_BASE
}

pub fn elevated_color() -> Color {
    BG_ELEVATED
}

pub fn hover_color() -> Color {
    BG_HOVER
}

pub fn text_color() -> Color {
    TEXT_PRIMARY
}

pub fn text_secondary() -> Color {
    TEXT_SECONDARY
}

pub fn accent_primary() -> Color {
    ACCENT_PRIMARY
}

pub fn accent_success() -> Color {
    ACCENT_SUCCESS
}

pub fn mix_colors(a: Color, b: Color, factor: f32) -> Color {
    let t = factor.clamp(0.0, 1.0);
    Color::new(
        a.r + (b.r - a.r) * t,
        a.g + (b.g - a.g) * t,
        a.b + (b.b - a.b) * t,
        a.a + (b.a - a.a) * t,
    )
}

pub fn with_alpha(color: Color, alpha: f32) -> Color {
    Color::new(color.r, color.g, color.b, alpha.clamp(0.0, 1.0))
}

/// Converts a color into an `[f32; 4]` RGBA array for GPU pipelines.
pub fn color_to_rgba(color: Color) -> [f32; 4] {
    [color.r, color.g, color.b, color.a]
}

/// Converts a color to RGBA with custom opacity override.
pub fn color_to_rgba_with_opacity(color: Color, opacity: f32) -> [f32; 4] {
    let mut rgba = color_to_rgba(color);
    rgba[3] = rgba[3].clamp(0.0, 1.0) * opacity.clamp(0.0, 1.0);
    rgba
}

/// Compares two colors for approximate equality within a small epsilon.
pub fn colors_equal(a: Color, b: Color) -> bool {
    const EPSILON: f32 = 1e-4;
    (a.r - b.r).abs() <= EPSILON
        && (a.g - b.g).abs() <= EPSILON
        && (a.b - b.b).abs() <= EPSILON
        && (a.a - b.a).abs() <= EPSILON
}
