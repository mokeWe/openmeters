use super::{ModuleSettingsPane, SettingsMessage};
use crate::dsp::spectrogram::{FrequencyScale, SpectrogramConfig, WindowKind};
use crate::ui::settings::{ModuleSettings, SettingsHandle};
use crate::ui::theme;
use crate::ui::visualization::visual_manager::{VisualId, VisualKind, VisualManagerHandle};
use iced::Element;
use iced::Length;
use iced::alignment::Vertical;
use iced::widget::rule;
use iced::widget::text::Style as TextStyle;
use iced::widget::{Rule, column, container, pick_list, row, slider, text, toggler};
use std::fmt;

const FFT_OPTIONS: [usize; 5] = [1024, 2048, 4096, 8192, 16384];
const ZERO_PADDING_OPTIONS: [usize; 6] = [1, 2, 4, 8, 16, 32];
const FREQUENCY_SCALE_OPTIONS: [FrequencyScale; 3] = [
    FrequencyScale::Linear,
    FrequencyScale::Logarithmic,
    FrequencyScale::Mel,
];

// Slider ranges: (min, max, step)
const HISTORY_RANGE: (f32, f32, f32) = (120.0, 960.0, 30.0);
const REASSIGNMENT_FLOOR_RANGE: (f32, f32, f32) = (-120.0, -30.0, 1.0);
const TEMPORAL_SMOOTHING_RANGE: (f32, f32, f32) = (0.0, 0.99, 0.01);
const SYNCHRO_BINS_RANGE: (f32, f32, f32) = (64.0, 4096.0, 64.0);
const TEMPORAL_MAX_HZ_RANGE: (f32, f32, f32) = (0.0, 4000.0, 1.0);
const TEMPORAL_BLEND_HZ_RANGE: (f32, f32, f32) = (0.0, 4000.0, 1.0);
const FREQUENCY_SMOOTHING_RANGE: (f32, f32, f32) = (0.0, 20.0, 1.0);
const FREQUENCY_MAX_HZ_RANGE: (f32, f32, f32) = (0.0, 4000.0, 1.0);
const FREQUENCY_BLEND_HZ_RANGE: (f32, f32, f32) = (0.0, 4000.0, 1.0);
const SECTION_PADDING: f32 = 12.0;
const SECTION_SPACING: f32 = 10.0;
const ROW_SPACING: f32 = 10.0;
const CONTROL_SPACING: f32 = 8.0;
const OUTER_SPACING: f32 = 14.0;
const RULE_HEIGHT: f32 = 1.0;
const RULE_THICKNESS: u16 = 1;
const RULE_FILL_PERCENT: f32 = 82.0;

const TITLE_SIZE: u16 = 14;
const LABEL_SIZE: u16 = 12;
const VALUE_SIZE: u16 = 11;
const TOGGLER_TEXT_SIZE: u16 = 11;

fn clamp_synchro_bins(value: usize) -> usize {
    let (min, max, step) = SYNCHRO_BINS_RANGE;
    let snapped = ((value as f32 / step).round() * step).clamp(min, max);
    snapped as usize
}

#[derive(Debug)]
pub struct SpectrogramSettingsPane {
    visual_id: VisualId,
    config: SpectrogramConfig,
    window: WindowPreset,
    frequency_scale: FrequencyScale,
    hop_ratio: HopRatio,
}

impl SpectrogramSettingsPane {
    fn core_section(&self) -> container::Container<'_, SettingsMessage> {
        let fft_row = labeled_pick_list(
            "FFT size",
            FFT_OPTIONS.to_vec(),
            Some(self.config.fft_size),
            |size| SettingsMessage::Spectrogram(Message::FftSize(size)),
        );

        let hop_row = labeled_pick_list(
            "Hop overlap",
            HopRatio::ALL.to_vec(),
            Some(self.hop_ratio),
            |ratio| SettingsMessage::Spectrogram(Message::HopRatio(ratio)),
        );

        let window_row = labeled_pick_list(
            "Window",
            WindowPreset::ALL.to_vec(),
            Some(self.window),
            |preset| SettingsMessage::Spectrogram(Message::Window(preset)),
        );

        let frequency_scale_row = labeled_pick_list(
            "Frequency scale",
            FREQUENCY_SCALE_OPTIONS.to_vec(),
            Some(self.frequency_scale),
            |scale| SettingsMessage::Spectrogram(Message::FrequencyScale(scale)),
        );

        let zero_padding_row = labeled_pick_list(
            "Zero padding",
            ZERO_PADDING_OPTIONS.to_vec(),
            Some(self.config.zero_padding_factor),
            |value| SettingsMessage::Spectrogram(Message::ZeroPadding(value)),
        );

        let history_slider = labeled_slider(
            "History length",
            self.config.history_length as f32,
            format!("{} columns", self.config.history_length),
            HISTORY_RANGE,
            |value| SettingsMessage::Spectrogram(Message::HistoryLength(value)),
        );

        let paired_pick_lists = row![
            column![fft_row, hop_row].spacing(CONTROL_SPACING),
            column![window_row, frequency_scale_row, zero_padding_row].spacing(CONTROL_SPACING),
        ]
        .spacing(ROW_SPACING)
        .width(Length::Fill);

        section_container(
            "Core controls",
            column![paired_pick_lists, history_slider].spacing(CONTROL_SPACING),
        )
    }

    fn advanced_section(&self) -> container::Container<'_, SettingsMessage> {
        let reassignment_floor = labeled_slider(
            "Reassignment floor",
            self.config.reassignment_power_floor_db,
            format!("{:.0} dB", self.config.reassignment_power_floor_db),
            REASSIGNMENT_FLOOR_RANGE,
            |value| SettingsMessage::Spectrogram(Message::ReassignmentFloor(value)),
        );

        let temporal_smoothing = labeled_slider(
            "Temporal smoothing",
            self.config.temporal_smoothing,
            format!("{:.2}", self.config.temporal_smoothing),
            TEMPORAL_SMOOTHING_RANGE,
            |value| SettingsMessage::Spectrogram(Message::TemporalSmoothing(value)),
        );

        let synchro_bins = labeled_slider(
            "Synchrosqueezing bins",
            self.config.synchrosqueezing_bin_count as f32,
            format!("{} bins", self.config.synchrosqueezing_bin_count),
            SYNCHRO_BINS_RANGE,
            |value| {
                let (min, max, step) = SYNCHRO_BINS_RANGE;
                let bins = ((value / step).round() * step).clamp(min, max) as usize;
                SettingsMessage::Spectrogram(Message::SynchroBinCount(bins))
            },
        );

        let temporal_smoothing_max = labeled_slider(
            "Temporal smoothing max",
            self.config.temporal_smoothing_max_hz,
            format!("{:.0} Hz", self.config.temporal_smoothing_max_hz),
            TEMPORAL_MAX_HZ_RANGE,
            |value| SettingsMessage::Spectrogram(Message::TemporalSmoothingMaxHz(value)),
        );

        let temporal_smoothing_blend = labeled_slider(
            "Temporal smoothing blend",
            self.config.temporal_smoothing_blend_hz,
            format!("{:.0} Hz", self.config.temporal_smoothing_blend_hz),
            TEMPORAL_BLEND_HZ_RANGE,
            |value| SettingsMessage::Spectrogram(Message::TemporalSmoothingBlendHz(value)),
        );

        let frequency_smoothing = labeled_slider(
            "Frequency smoothing",
            self.config.frequency_smoothing_radius as f32,
            format!("{} bins", self.config.frequency_smoothing_radius),
            FREQUENCY_SMOOTHING_RANGE,
            |value| SettingsMessage::Spectrogram(Message::FrequencySmoothing(value)),
        );

        let frequency_smoothing_max = labeled_slider(
            "Frequency smoothing max",
            self.config.frequency_smoothing_max_hz,
            format!("{:.0} Hz", self.config.frequency_smoothing_max_hz),
            FREQUENCY_MAX_HZ_RANGE,
            |value| SettingsMessage::Spectrogram(Message::FrequencySmoothingMaxHz(value)),
        );

        let frequency_smoothing_blend = labeled_slider(
            "Frequency smoothing blend",
            self.config.frequency_smoothing_blend_hz,
            format!("{:.0} Hz", self.config.frequency_smoothing_blend_hz),
            FREQUENCY_BLEND_HZ_RANGE,
            |value| SettingsMessage::Spectrogram(Message::FrequencySmoothingBlendHz(value)),
        );

        let reassignment_block = column![reassignment_floor].spacing(CONTROL_SPACING);
        let synchro_block = column![synchro_bins].spacing(CONTROL_SPACING);

        let smoothing_columns = row![
            column![
                temporal_smoothing,
                temporal_smoothing_max,
                temporal_smoothing_blend
            ]
            .spacing(CONTROL_SPACING)
            .width(Length::FillPortion(1)),
            column![
                frequency_smoothing,
                frequency_smoothing_max,
                frequency_smoothing_blend,
            ]
            .spacing(CONTROL_SPACING)
            .width(Length::FillPortion(1)),
        ]
        .spacing(ROW_SPACING)
        .width(Length::Fill);

        let mut advanced_content = column![
            toggler(self.config.use_reassignment)
                .label("Time-frequency reassignment")
                .text_size(TOGGLER_TEXT_SIZE)
                .spacing(CONTROL_SPACING - 4.0)
                .on_toggle(|value| {
                    SettingsMessage::Spectrogram(Message::UseReassignment(value))
                }),
        ]
        .spacing(CONTROL_SPACING);

        if self.config.use_reassignment {
            advanced_content = advanced_content.push(reassignment_block).push(
                toggler(self.config.use_synchrosqueezing)
                    .label("Synchrosqueezed accumulation")
                    .text_size(TOGGLER_TEXT_SIZE)
                    .spacing(CONTROL_SPACING - 4.0)
                    .on_toggle(|value| {
                        SettingsMessage::Spectrogram(Message::UseSynchrosqueezing(value))
                    }),
            );

            if self.config.use_synchrosqueezing {
                advanced_content = advanced_content.push(synchro_block).push(smoothing_columns);
            }
        }

        section_container("Advanced signal processing", advanced_content)
    }
}

#[derive(Debug, Clone)]
pub enum Message {
    FftSize(usize),
    HopRatio(HopRatio),
    HistoryLength(f32),
    Window(WindowPreset),
    FrequencyScale(FrequencyScale),
    UseReassignment(bool),
    ReassignmentFloor(f32),
    ZeroPadding(usize),
    UseSynchrosqueezing(bool),
    SynchroBinCount(usize),
    TemporalSmoothing(f32),
    TemporalSmoothingMaxHz(f32),
    TemporalSmoothingBlendHz(f32),
    FrequencySmoothing(f32),
    FrequencySmoothingMaxHz(f32),
    FrequencySmoothingBlendHz(f32),
}

pub fn create(
    visual_id: VisualId,
    visual_manager: &VisualManagerHandle,
) -> SpectrogramSettingsPane {
    let stored_config = visual_manager
        .borrow()
        .module_settings(VisualKind::SPECTROGRAM)
        .and_then(|stored| stored.spectrogram_config())
        .unwrap_or_default();

    let mut config = stored_config;

    if !config.use_reassignment {
        config.use_synchrosqueezing = false;
    }

    let window = WindowPreset::from_kind(config.window);
    let frequency_scale = config.frequency_scale;
    let hop_ratio = HopRatio::from_config(config.fft_size, config.hop_size);

    SpectrogramSettingsPane {
        visual_id,
        config,
        window,
        frequency_scale,
        hop_ratio,
    }
}

// Helper function to create a labeled slider
fn labeled_slider<'a>(
    label: &'static str,
    value: f32,
    format: String,
    range: (f32, f32, f32), // (min, max, step)
    on_change: impl Fn(f32) -> SettingsMessage + 'a,
) -> iced::widget::Column<'a, SettingsMessage> {
    let (min, max, step) = range;
    let muted = muted_text_style();

    column![
        row![
            text(label).size(LABEL_SIZE),
            text(format).size(VALUE_SIZE).style(muted.clone()),
        ]
        .spacing(6.0),
        slider::Slider::new(min..=max, value, on_change).step(step),
    ]
    .spacing(CONTROL_SPACING)
}

fn labeled_pick_list<'a, T>(
    label: &'static str,
    options: Vec<T>,
    selected: Option<T>,
    on_select: impl Fn(T) -> SettingsMessage + 'a,
) -> iced::widget::Row<'a, SettingsMessage>
where
    T: Clone + PartialEq + fmt::Display + 'static,
{
    row![
        text(label).size(LABEL_SIZE),
        pick_list(options, selected, on_select),
    ]
    .spacing(ROW_SPACING)
    .align_y(Vertical::Center)
}

fn muted_text_style() -> impl Fn(&iced::Theme) -> TextStyle + Clone {
    let color = theme::text_secondary();
    move |_| TextStyle { color: Some(color) }
}

fn section_container<'a>(
    title: &'static str,
    body: iced::widget::Column<'a, SettingsMessage>,
) -> container::Container<'a, SettingsMessage> {
    container(column![text(title).size(TITLE_SIZE), body].spacing(SECTION_SPACING))
        .padding(SECTION_PADDING)
}

impl ModuleSettingsPane for SpectrogramSettingsPane {
    fn visual_id(&self) -> VisualId {
        self.visual_id
    }

    fn view(&self) -> Element<'_, SettingsMessage> {
        let divider_color = theme::with_alpha(theme::text_secondary(), 0.35);
        let section_divider = Rule::horizontal(RULE_HEIGHT).style(move |_| rule::Style {
            color: divider_color,
            width: RULE_THICKNESS,
            radius: 0.0.into(),
            fill_mode: rule::FillMode::Percent(RULE_FILL_PERCENT),
        });

        column![
            self.core_section(),
            section_divider,
            self.advanced_section()
        ]
        .spacing(OUTER_SPACING)
        .into()
    }

    fn handle(
        &mut self,
        message: &SettingsMessage,
        visual_manager: &VisualManagerHandle,
        settings: &SettingsHandle,
    ) {
        let SettingsMessage::Spectrogram(msg) = message else {
            return;
        };

        let mut changed = false;
        match msg {
            Message::FftSize(size) => {
                if self.config.fft_size != *size {
                    self.config.fft_size = *size;
                    self.config.hop_size = self.hop_ratio.to_hop_size(*size);
                    changed = true;
                }
            }
            Message::HopRatio(ratio) => {
                if self.hop_ratio != *ratio {
                    self.hop_ratio = *ratio;
                    self.config.hop_size = ratio.to_hop_size(self.config.fft_size);
                    changed = true;
                }
            }
            Message::HistoryLength(value) => {
                let (min, max, step) = HISTORY_RANGE;
                let columns = ((value / step).round() * step).clamp(min, max) as usize;
                if self.config.history_length != columns {
                    self.config.history_length = columns;
                    changed = true;
                }
            }
            Message::Window(preset) => {
                if self.window != *preset {
                    self.window = *preset;
                    self.config.window = preset.to_window_kind();
                    changed = true;
                }
            }
            Message::FrequencyScale(scale) => {
                if self.frequency_scale != *scale {
                    self.frequency_scale = *scale;
                    self.config.frequency_scale = *scale;
                    changed = true;
                }
            }
            Message::UseReassignment(value) => {
                if self.config.use_reassignment != *value {
                    self.config.use_reassignment = *value;
                    if !self.config.use_reassignment {
                        self.config.use_synchrosqueezing = false;
                    }
                    changed = true;
                }
            }
            Message::ReassignmentFloor(value) => {
                let (min, max, _) = REASSIGNMENT_FLOOR_RANGE;
                let clamped = value.clamp(min, max);
                if (self.config.reassignment_power_floor_db - clamped).abs() > f32::EPSILON {
                    self.config.reassignment_power_floor_db = clamped;
                    changed = true;
                }
            }
            Message::ZeroPadding(value) => {
                if self.config.zero_padding_factor != *value {
                    self.config.zero_padding_factor = *value;
                    changed = true;
                }
            }
            Message::UseSynchrosqueezing(value) => {
                let desired = *value && self.config.use_reassignment;
                if self.config.use_synchrosqueezing != desired {
                    self.config.use_synchrosqueezing = desired;
                    changed = true;
                }
            }
            Message::SynchroBinCount(value) => {
                if self.config.use_synchrosqueezing {
                    let bins = clamp_synchro_bins(*value);
                    if self.config.synchrosqueezing_bin_count != bins {
                        self.config.synchrosqueezing_bin_count = bins;
                        changed = true;
                    }
                }
            }
            Message::TemporalSmoothing(value) => {
                if self.config.use_synchrosqueezing {
                    let (min, max, _) = TEMPORAL_SMOOTHING_RANGE;
                    let clamped = value.clamp(min, max);
                    if (self.config.temporal_smoothing - clamped).abs() > f32::EPSILON {
                        self.config.temporal_smoothing = clamped;
                        changed = true;
                    }
                }
            }
            Message::TemporalSmoothingMaxHz(value) => {
                if self.config.use_synchrosqueezing {
                    let (min, max, _) = TEMPORAL_MAX_HZ_RANGE;
                    let clamped = value.clamp(min, max);
                    if (self.config.temporal_smoothing_max_hz - clamped).abs() > f32::EPSILON {
                        self.config.temporal_smoothing_max_hz = clamped;
                        changed = true;
                    }
                }
            }
            Message::TemporalSmoothingBlendHz(value) => {
                if self.config.use_synchrosqueezing {
                    let (min, max, _) = TEMPORAL_BLEND_HZ_RANGE;
                    let clamped = value.clamp(min, max);
                    if (self.config.temporal_smoothing_blend_hz - clamped).abs() > f32::EPSILON {
                        self.config.temporal_smoothing_blend_hz = clamped;
                        changed = true;
                    }
                }
            }
            Message::FrequencySmoothing(value) => {
                if self.config.use_synchrosqueezing {
                    let (min, max, _) = FREQUENCY_SMOOTHING_RANGE;
                    let radius = value.clamp(min, max).round() as usize;
                    if self.config.frequency_smoothing_radius != radius {
                        self.config.frequency_smoothing_radius = radius;
                        changed = true;
                    }
                }
            }
            Message::FrequencySmoothingMaxHz(value) => {
                if self.config.use_synchrosqueezing {
                    let (min, max, _) = FREQUENCY_MAX_HZ_RANGE;
                    let clamped = value.clamp(min, max);
                    if (self.config.frequency_smoothing_max_hz - clamped).abs() > f32::EPSILON {
                        self.config.frequency_smoothing_max_hz = clamped;
                        changed = true;
                    }
                }
            }
            Message::FrequencySmoothingBlendHz(value) => {
                if self.config.use_synchrosqueezing {
                    let (min, max, _) = FREQUENCY_BLEND_HZ_RANGE;
                    let clamped = value.clamp(min, max);
                    if (self.config.frequency_smoothing_blend_hz - clamped).abs() > f32::EPSILON {
                        self.config.frequency_smoothing_blend_hz = clamped;
                        changed = true;
                    }
                }
            }
        }

        if changed {
            apply_changes(self, visual_manager, settings);
        }
    }
}

fn apply_changes(
    pane: &mut SpectrogramSettingsPane,
    visual_manager: &VisualManagerHandle,
    settings: &SettingsHandle,
) {
    let config = pane.config;

    let module_settings = ModuleSettings::with_spectrogram_config(&config);

    if visual_manager
        .borrow_mut()
        .apply_module_settings(VisualKind::SPECTROGRAM, &module_settings)
    {
        settings
            .update(|settings| settings.set_spectrogram_settings(VisualKind::SPECTROGRAM, &config));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WindowPreset {
    Rectangular,
    Hann,
    Hamming,
    Blackman,
    PlanckBessel,
}

impl WindowPreset {
    const ALL: [WindowPreset; 5] = [
        WindowPreset::Rectangular,
        WindowPreset::Hann,
        WindowPreset::Hamming,
        WindowPreset::Blackman,
        WindowPreset::PlanckBessel,
    ];

    fn from_kind(kind: WindowKind) -> Self {
        match kind {
            WindowKind::Rectangular => WindowPreset::Rectangular,
            WindowKind::Hann => WindowPreset::Hann,
            WindowKind::Hamming => WindowPreset::Hamming,
            WindowKind::Blackman => WindowPreset::Blackman,
            WindowKind::PlanckBessel { .. } => WindowPreset::PlanckBessel,
        }
    }

    fn to_window_kind(self) -> WindowKind {
        match self {
            WindowPreset::Rectangular => WindowKind::Rectangular,
            WindowPreset::Hann => WindowKind::Hann,
            WindowPreset::Hamming => WindowKind::Hamming,
            WindowPreset::Blackman => WindowKind::Blackman,
            WindowPreset::PlanckBessel => WindowKind::PlanckBessel {
                epsilon: 0.1,
                beta: 5.5,
            },
        }
    }
}

impl fmt::Display for WindowPreset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            WindowPreset::Rectangular => "Rectangular",
            WindowPreset::Hann => "Hann",
            WindowPreset::Hamming => "Hamming",
            WindowPreset::Blackman => "Blackman",
            WindowPreset::PlanckBessel => "Planck-Bessel",
        };
        write!(f, "{}", label)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HopRatio {
    Quarter,
    Sixth,
    Eighth,
    Sixteenth,
}

impl HopRatio {
    const ALL: [HopRatio; 4] = [
        HopRatio::Quarter,
        HopRatio::Sixth,
        HopRatio::Eighth,
        HopRatio::Sixteenth,
    ];

    fn from_config(fft_size: usize, hop_size: usize) -> Self {
        if fft_size == 0 || hop_size == 0 {
            return HopRatio::Eighth;
        }

        let ratio = fft_size as f32 / hop_size as f32;
        let mut best = HopRatio::Eighth;
        let mut best_err = f32::MAX;
        for candidate in Self::ALL {
            let target = candidate.divisor() as f32;
            let err = (ratio - target).abs();
            if err < best_err {
                best = candidate;
                best_err = err;
            }
        }
        best
    }

    fn divisor(self) -> usize {
        match self {
            HopRatio::Quarter => 4,
            HopRatio::Sixth => 6,
            HopRatio::Eighth => 8,
            HopRatio::Sixteenth => 16,
        }
    }

    fn to_hop_size(self, fft_size: usize) -> usize {
        let divisor = self.divisor().max(1);
        (fft_size / divisor).max(1)
    }
}

impl fmt::Display for HopRatio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            HopRatio::Quarter => "75% overlap",
            HopRatio::Sixth => "83% overlap",
            HopRatio::Eighth => "87% overlap",
            HopRatio::Sixteenth => "94% overlap",
        };
        write!(f, "{}", label)
    }
}

impl fmt::Display for FrequencyScale {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FrequencyScale::Linear => write!(f, "Linear"),
            FrequencyScale::Logarithmic => write!(f, "Logarithmic"),
            FrequencyScale::Mel => write!(f, "Mel"),
        }
    }
}
