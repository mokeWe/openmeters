//! Monochrome Iced theme.
//!
//! GPU palette colors are defined in sRGB. The sRGB framebuffer format handles
//! gamma correction automatically, so colors are passed through without conversion.

use iced::border::Border;
use iced::theme::palette::{self, Extended};
use iced::widget::{button, container, slider, text};
use iced::{Background, Color, Theme};

// Core palette stops
// Slightly neutralized base to reduce blue cast in derived weak tones.
pub const BG_BASE: Color = Color::from_rgba(0.065, 0.065, 0.065, 1.0);

const TEXT_PRIMARY: Color = Color::from_rgba(0.902, 0.910, 0.925, 1.0);
const TEXT_DARK: Color = Color::from_rgba(0.10, 0.10, 0.10, 1.0);

const BORDER_SUBTLE: Color = Color::from_rgba(0.280, 0.288, 0.304, 1.0);
const BORDER_FOCUS: Color = Color::from_rgba(0.520, 0.536, 0.560, 1.0);

// Accent colors
const ACCENT_PRIMARY: Color = Color::from_rgba(0.157, 0.157, 0.157, 1.0);
const ACCENT_SUCCESS: Color = Color::from_rgba(0.478, 0.557, 0.502, 1.0);
const ACCENT_DANGER: Color = Color::from_rgba(0.557, 0.478, 0.478, 1.0);

// GPU palettes

pub const DEFAULT_SPECTROGRAM_PALETTE: [Color; 5] = [
    Color::from_rgba(0.000, 0.000, 0.000, 0.0),
    Color::from_rgba(0.218, 0.106, 0.332, 1.0),
    Color::from_rgba(0.609, 0.000, 0.000, 1.0),
    Color::from_rgba(1.000, 0.737, 0.353, 1.0),
    Color::from_rgba(1.000, 1.000, 1.000, 1.0),
];

pub const DEFAULT_SPECTRUM_PALETTE: [Color; 5] = [
    Color::from_rgba(0.218, 0.106, 0.332, 1.0),
    Color::from_rgba(0.609, 0.000, 0.000, 1.0),
    Color::from_rgba(0.906, 0.485, 0.000, 1.0),
    Color::from_rgba(1.000, 0.737, 0.353, 1.0),
    Color::from_rgba(1.000, 1.000, 0.000, 1.0),
];

pub const DEFAULT_WAVEFORM_PALETTE: [Color; 5] = [
    Color::from_rgba(0.665, 0.000, 0.000, 1.0),
    Color::from_rgba(1.000, 0.000, 0.000, 1.0),
    Color::from_rgba(1.000, 0.665, 0.000, 1.0),
    Color::from_rgba(0.798, 0.485, 0.906, 1.0),
    Color::from_rgba(0.665, 0.000, 0.906, 1.0),
];

pub const DEFAULT_OSCILLOSCOPE_PALETTE: [Color; 1] = [Color::from_rgba(1.000, 1.000, 1.000, 1.0)];

pub const DEFAULT_STEREOMETER_PALETTE: [Color; 2] = [
    Color::from_rgba(1.000, 1.000, 1.000, 1.0),
    Color::from_rgba(0.537, 0.547, 0.566, 1.0),
];

pub const DEFAULT_LOUDNESS_PALETTE: [Color; 5] = [
    Color::from_rgba(0.161, 0.161, 0.161, 1.0),
    Color::from_rgba(0.626, 0.665, 0.680, 1.0),
    Color::from_rgba(0.584, 0.618, 0.650, 1.0),
    Color::from_rgba(0.701, 0.767, 0.735, 1.0),
    Color::from_rgba(0.735, 0.748, 0.774, 0.88),
];

pub fn theme(custom_bg: Option<Color>) -> Theme {
    Theme::custom_with_fn(
        "OpenMeters Monochrome".to_string(),
        palette(custom_bg),
        Extended::generate,
    )
}

fn palette(custom_bg: Option<Color>) -> palette::Palette {
    let background = custom_bg.unwrap_or(BG_BASE);
    let text = if palette::is_dark(background) {
        TEXT_PRIMARY
    } else {
        TEXT_DARK
    };

    palette::Palette {
        background,
        text,
        primary: ACCENT_PRIMARY,
        success: ACCENT_SUCCESS,
        warning: ACCENT_SUCCESS,
        danger: ACCENT_DANGER,
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
            let hover = palette::deviate(base, 0.05);
            style.background = Some(Background::Color(hover));
        }
        button::Status::Pressed => {
            style.border = focus_border();
        }
        _ => {}
    }

    style
}

pub fn tab_button_style(theme: &Theme, active: bool, status: button::Status) -> button::Style {
    let palette = theme.extended_palette();
    let mut base = if active {
        palette.primary.base.color
    } else {
        // Hand-tuned neutral lift to avoid the bluish tint of the generated weak background.
        mix_colors(palette.background.base.color, Color::WHITE, 0.08)
    };
    base.a = 1.0;
    button_style(theme, base, status)
}

pub fn weak_container(theme: &Theme) -> container::Style {
    let palette = theme.extended_palette();
    container::Style {
        background: Some(Background::Color(palette.background.weak.color)),
        text_color: Some(palette.background.base.text),
        border: sharp_border(),
        ..Default::default()
    }
}

pub fn weak_text_style(theme: &Theme) -> text::Style {
    text::Style {
        color: Some(theme.extended_palette().secondary.weak.text),
    }
}

pub fn opaque_container(theme: &Theme) -> container::Style {
    let palette = theme.extended_palette();
    let mut bg = palette.background.base.color;
    bg.a = 1.0;
    container::Style {
        background: Some(Background::Color(bg)),
        ..Default::default()
    }
}

pub fn accent_primary() -> Color {
    ACCENT_PRIMARY
}

pub fn mix_colors(a: Color, b: Color, factor: f32) -> Color {
    let t = factor.clamp(0.0, 1.0);
    Color::from_rgba(
        a.r + (b.r - a.r) * t,
        a.g + (b.g - a.g) * t,
        a.b + (b.b - a.b) * t,
        a.a + (b.a - a.a) * t,
    )
}

pub fn with_alpha(color: Color, alpha: f32) -> Color {
    Color {
        a: alpha.clamp(0.0, 1.0),
        ..color
    }
}

pub fn slider_style(_theme: &Theme, status: slider::Status) -> slider::Style {
    // Neutral track sits above the base surface; filled rail leans slightly into the accent.
    let palette = _theme.extended_palette();
    let track = mix_colors(palette.background.base.color, Color::WHITE, 0.16);
    let filled = mix_colors(palette.primary.base.color, Color::WHITE, 0.10);

    let (handle_color, border_color, border_width) = match status {
        slider::Status::Hovered => (filled, BORDER_FOCUS, 1.0),
        slider::Status::Dragged => (filled, BORDER_FOCUS, 1.0),
        _ => (filled, BORDER_SUBTLE, 1.0),
    };

    slider::Style {
        rail: slider::Rail {
            backgrounds: (Background::Color(filled), Background::Color(track)),
            border: sharp_border(),
            width: 2.0,
        },
        handle: slider::Handle {
            shape: slider::HandleShape::Circle { radius: 7.0 },
            background: Background::Color(handle_color),
            border_color,
            border_width,
        },
    }
}

/// Converts a color to `[f32; 4]` RGBA array for GPU pipelines.
#[inline]
pub fn color_to_rgba(color: Color) -> [f32; 4] {
    [color.r, color.g, color.b, color.a]
}

/// Compares two colors for approximate equality.
#[inline]
pub fn colors_equal(a: Color, b: Color) -> bool {
    const EPSILON: f32 = 1e-4;
    (a.r - b.r).abs() <= EPSILON
        && (a.g - b.g).abs() <= EPSILON
        && (a.b - b.b).abs() <= EPSILON
        && (a.a - b.a).abs() <= EPSILON
}
