//! Monochrome Iced theme.

use iced::border::Border;
use iced::theme::palette::{self, Extended, Pair};
use iced::widget::button;
use iced::{Background, Color, Theme};

// Core palette stops
pub const BG_BASE: Color = Color::from_rgb(0.059, 0.063, 0.071);

const TEXT_PRIMARY: Color = Color::from_rgb(0.902, 0.910, 0.925);
const TEXT_DARK: Color = Color::from_rgb(0.10, 0.10, 0.10);
const TEXT_SECONDARY: Color = Color::from_rgb(0.655, 0.671, 0.698);
const TEXT_SECONDARY_DARK: Color = Color::from_rgb(0.40, 0.40, 0.40);

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

pub const DEFAULT_OSCILLOSCOPE_PALETTE: [Color; 1] = [
    Color::from_rgb(0.529, 0.549, 0.584), // trace color - primary accent
];

pub const DEFAULT_STEREOMETER_PALETTE: [Color; 1] = [
    Color::from_rgb(0.529, 0.549, 0.584), // XY trace - primary accent
];

pub fn luminance(color: Color) -> f32 {
    0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b
}

fn lighten(color: Color, amount: f32) -> Color {
    Color {
        r: (color.r + amount).min(1.0),
        g: (color.g + amount).min(1.0),
        b: (color.b + amount).min(1.0),
        a: color.a,
    }
}

fn darken(color: Color, amount: f32) -> Color {
    Color {
        r: (color.r - amount).max(0.0),
        g: (color.g - amount).max(0.0),
        b: (color.b - amount).max(0.0),
        a: color.a,
    }
}

pub fn theme(custom_bg: Option<Color>) -> Theme {
    Theme::custom_with_fn(
        "OpenMeters Monochrome".to_string(),
        palette(custom_bg),
        extended_palette,
    )
}

fn palette(custom_bg: Option<Color>) -> palette::Palette {
    let background = custom_bg.unwrap_or(BG_BASE);
    let is_light = luminance(background) > 0.5;
    let text = if is_light { TEXT_DARK } else { TEXT_PRIMARY };

    palette::Palette {
        background,
        text,
        primary: ACCENT_PRIMARY,
        success: ACCENT_SUCCESS,
        danger: ACCENT_DANGER,
    }
}

fn extended_palette(base: palette::Palette) -> Extended {
    let is_light = luminance(base.background) > 0.5;
    let text_secondary = if is_light {
        TEXT_SECONDARY_DARK
    } else {
        TEXT_SECONDARY
    };

    // Adjust surface colors based on background luminance
    // If light, darken. If dark, lighten.
    let (surface, elevated) = if is_light {
        (darken(base.background, 0.05), darken(base.background, 0.10))
    } else {
        (
            lighten(base.background, 0.05),
            lighten(base.background, 0.10),
        )
    };

    Extended {
        background: palette::Background {
            base: Pair::new(base.background, base.text),
            weak: Pair::new(surface, base.text),
            strong: Pair::new(elevated, base.text),
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
                text_secondary,
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
            base: Pair::new(surface, base.text),
            weak: Pair::new(base.background, text_secondary),
            strong: Pair::new(elevated, base.text),
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
                text_secondary,
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
                text_secondary,
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
        is_dark: !is_light,
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

pub fn button_style(theme: &Theme, base: Color, status: button::Status) -> button::Style {
    let palette = theme.extended_palette();
    let mut style = button::Style {
        background: Some(Background::Color(base)),
        text_color: palette.background.base.text,
        border: sharp_border(),
        ..Default::default()
    };

    match status {
        button::Status::Hovered => {
            // We need a hover color that contrasts with the base.
            // If base is light, darken. If dark, lighten.
            let is_light = luminance(base) > 0.5;
            let hover = if is_light {
                darken(base, 0.05)
            } else {
                lighten(base, 0.05)
            };
            style.background = Some(Background::Color(hover));
        }
        button::Status::Pressed => {
            style.border = focus_border();
        }
        _ => {}
    }

    style
}

pub fn surface_button_style(theme: &Theme, status: button::Status) -> button::Style {
    let palette = theme.extended_palette();
    button_style(theme, palette.background.weak.color, status)
}

pub fn accent_primary() -> Color {
    ACCENT_PRIMARY
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
