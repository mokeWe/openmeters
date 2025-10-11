//! custom monochrome Iced theme

use iced::border::Border;
use iced::theme::palette::{self, Extended, Pair};
use iced::{Color, Theme};

// Core palette stops
const BG_BASE: Color = Color::from_rgb(0.059, 0.063, 0.071); // #0f1012
const BG_SURFACE: Color = Color::from_rgb(0.090, 0.094, 0.106); // #17181b
const BG_ELEVATED: Color = Color::from_rgb(0.122, 0.129, 0.145); // #1f2125
const BG_HOVER: Color = Color::from_rgb(0.157, 0.165, 0.184); // #28293a

const TEXT_PRIMARY: Color = Color::from_rgb(0.902, 0.910, 0.925); // #e6e8ec
const TEXT_SECONDARY: Color = Color::from_rgb(0.655, 0.671, 0.698); // #a7abb2

const BORDER_SUBTLE: Color = Color::from_rgb(0.196, 0.204, 0.224); // #323438
const BORDER_FOCUS: Color = Color::from_rgb(0.416, 0.424, 0.443); // #6a6c71

// Accent colors
const ACCENT_PRIMARY: Color = Color::from_rgb(0.529, 0.549, 0.584); // #878c95
const ACCENT_SUCCESS: Color = Color::from_rgb(0.478, 0.557, 0.502); // #7a8e80
const ACCENT_DANGER: Color = Color::from_rgb(0.557, 0.478, 0.478); // #8e7a7a

// Visualization palettes
pub const DEFAULT_SPECTROGRAM_PALETTE: [Color; 5] = [
    Color::from_rgba(0.05, 0.08, 0.18, 0.0),
    Color::from_rgba(0.13, 0.20, 0.46, 1.0),
    Color::from_rgba(0.11, 0.48, 0.63, 1.0),
    Color::from_rgba(0.94, 0.75, 0.29, 1.0),
    Color::from_rgba(0.98, 0.93, 0.65, 1.0),
];

/// Returns the default spectrogram color palette.
pub fn spectrogram_palette() -> [Color; 5] {
    DEFAULT_SPECTROGRAM_PALETTE
}

/// Returns the monochrome theme for the application.
pub fn theme() -> Theme {
    Theme::custom_with_fn(
        "OpenMeters Monochrome".to_string(),
        palette(),
        extended_palette,
    )
}

/// Base palette mapping.
fn palette() -> palette::Palette {
    palette::Palette {
        background: BG_BASE,
        text: TEXT_PRIMARY,
        primary: ACCENT_PRIMARY,
        success: ACCENT_SUCCESS,
        danger: ACCENT_DANGER,
    }
}

/// Extended palette generator for widget states.
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

/// Focused/active border for interactive elements.
pub fn focus_border() -> Border {
    Border {
        color: BORDER_FOCUS,
        width: 1.0,
        radius: 0.0.into(),
    }
}

// Color utilities

/// Returns the surface background color.
pub fn surface_color() -> Color {
    BG_SURFACE
}

/// Returns the base background color.
pub fn base_color() -> Color {
    BG_BASE
}

/// Returns the elevated background color for layering.
pub fn elevated_color() -> Color {
    BG_ELEVATED
}

/// Returns the hover state background.
pub fn hover_color() -> Color {
    BG_HOVER
}

/// Returns the primary text color.
pub fn text_color() -> Color {
    TEXT_PRIMARY
}

/// Returns the secondary/muted text color.
pub fn text_secondary() -> Color {
    TEXT_SECONDARY
}

/// Returns the primary accent color.
pub fn accent_primary() -> Color {
    ACCENT_PRIMARY
}

/// Returns the success accent color.
pub fn accent_success() -> Color {
    ACCENT_SUCCESS
}

/// Returns the danger accent color.
pub fn accent_danger() -> Color {
    ACCENT_DANGER
}

/// Linearly interpolates between two colors.
pub fn mix_colors(a: Color, b: Color, factor: f32) -> Color {
    let t = factor.clamp(0.0, 1.0);
    Color::new(
        a.r + (b.r - a.r) * t,
        a.g + (b.g - a.g) * t,
        a.b + (b.b - a.b) * t,
        a.a + (b.a - a.a) * t,
    )
}

/// Applies a new alpha value to a color, clamped to [0, 1].
pub fn with_alpha(color: Color, alpha: f32) -> Color {
    Color::new(color.r, color.g, color.b, alpha.clamp(0.0, 1.0))
}

/// Converts a color into an `[f32; 4]` RGBA array for GPU pipelines.
pub fn color_to_rgba(color: Color) -> [f32; 4] {
    [color.r, color.g, color.b, color.a]
}
