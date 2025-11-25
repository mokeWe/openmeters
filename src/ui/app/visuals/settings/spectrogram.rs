use super::palette::{PaletteEditor, PaletteEvent};
use super::widgets::{
    CONTROL_SPACING, SliderRange, labeled_pick_list, labeled_slider, set_if_changed,
    update_f32_range, update_usize_from_f32,
};
use super::{ModuleSettingsPane, SettingsMessage};
use crate::dsp::spectrogram::{
    FrequencyScale, PLANCK_BESSEL_DEFAULT_BETA, PLANCK_BESSEL_DEFAULT_EPSILON, SpectrogramConfig,
    WindowKind,
};
use crate::ui::render::spectrogram::SPECTROGRAM_PALETTE_SIZE;
use crate::ui::settings::{ModuleSettings, PaletteSettings, SettingsHandle, SpectrogramSettings};
use crate::ui::theme;
use crate::ui::visualization::visual_manager::{VisualId, VisualKind, VisualManagerHandle};
use iced::Element;
use iced::Length;
use iced::widget::rule;
use iced::widget::{Rule, column, container, row, text, toggler};
use std::fmt;

const FFT_OPTIONS: [usize; 5] = [1024, 2048, 4096, 8192, 16384];
const ZERO_PADDING_OPTIONS: [usize; 6] = [1, 2, 4, 8, 16, 32];
const FREQUENCY_SCALE_OPTIONS: [FrequencyScale; 3] = [
    FrequencyScale::Linear,
    FrequencyScale::Logarithmic,
    FrequencyScale::Mel,
];

// Slider ranges: min, max, step
const HISTORY_RANGE: SliderRange = SliderRange::new(120.0, 960.0, 30.0);
const REASSIGNMENT_FLOOR_RANGE: SliderRange = SliderRange::new(-120.0, -30.0, 1.0);
const TEMPORAL_SMOOTHING_RANGE: SliderRange = SliderRange::new(0.0, 0.99, 0.01);
const SYNCHRO_BINS_RANGE: SliderRange = SliderRange::new(64.0, 4096.0, 64.0);
const TEMPORAL_MAX_HZ_RANGE: SliderRange = SliderRange::new(0.0, 4000.0, 1.0);
const TEMPORAL_BLEND_HZ_RANGE: SliderRange = SliderRange::new(0.0, 4000.0, 1.0);
const FREQUENCY_SMOOTHING_RANGE: SliderRange = SliderRange::new(0.0, 20.0, 1.0);
const FREQUENCY_MAX_HZ_RANGE: SliderRange = SliderRange::new(0.0, 4000.0, 1.0);
const FREQUENCY_BLEND_HZ_RANGE: SliderRange = SliderRange::new(0.0, 4000.0, 1.0);
const PLANCK_BESSEL_EPSILON_RANGE: SliderRange = SliderRange::new(0.01, 0.5, 0.01);
const PLANCK_BESSEL_BETA_RANGE: SliderRange = SliderRange::new(0.0, 20.0, 0.25);
const SECTION_PADDING: f32 = 12.0;
const SECTION_SPACING: f32 = 10.0;
const ROW_SPACING: f32 = 10.0;
const OUTER_SPACING: f32 = 14.0;
const RULE_HEIGHT: f32 = 1.0;
const RULE_THICKNESS: u16 = 1;
const RULE_FILL_PERCENT: f32 = 82.0;

const TITLE_SIZE: u16 = 14;
const TOGGLER_TEXT_SIZE: u16 = 11;

#[derive(Debug)]
pub struct SpectrogramSettingsPane {
    visual_id: VisualId,
    config: SpectrogramConfig,
    window: WindowPreset,
    frequency_scale: FrequencyScale,
    hop_ratio: HopRatio,
    palette: PaletteEditor,
    planck_bessel: PlanckBesselParams,
}

#[derive(Debug, Clone, Copy)]
struct PlanckBesselParams {
    epsilon: f32,
    beta: f32,
}

impl Default for PlanckBesselParams {
    fn default() -> Self {
        Self {
            epsilon: PLANCK_BESSEL_DEFAULT_EPSILON,
            beta: PLANCK_BESSEL_DEFAULT_BETA,
        }
    }
}

impl SpectrogramSettingsPane {
    fn persist(&self, visual_manager: &VisualManagerHandle, settings: &SettingsHandle) {
        let mut stored = SpectrogramSettings::from_config(&self.config);
        stored.palette = PaletteSettings::maybe_from_colors(
            self.palette.colors(),
            &theme::DEFAULT_SPECTROGRAM_PALETTE,
        );

        let module_settings = ModuleSettings::with_config(&stored);

        if visual_manager
            .borrow_mut()
            .apply_module_settings(VisualKind::SPECTROGRAM, &module_settings)
        {
            settings
                .update(|settings| settings.set_module_config(VisualKind::SPECTROGRAM, &stored));
        }
    }

    fn core_section(&self) -> container::Container<'_, SettingsMessage> {
        let fft_row = labeled_pick_list(
            "FFT size",
            FFT_OPTIONS.as_slice(),
            Some(self.config.fft_size),
            |size| SettingsMessage::Spectrogram(Message::FftSize(size)),
        )
        .spacing(ROW_SPACING);

        let hop_row = labeled_pick_list(
            "Hop overlap",
            HopRatio::ALL.as_slice(),
            Some(self.hop_ratio),
            |ratio| SettingsMessage::Spectrogram(Message::HopRatio(ratio)),
        )
        .spacing(ROW_SPACING);

        let window_row = labeled_pick_list(
            "Window",
            WindowPreset::ALL.as_slice(),
            Some(self.window),
            |preset| SettingsMessage::Spectrogram(Message::Window(preset)),
        )
        .spacing(ROW_SPACING);

        let frequency_scale_row = labeled_pick_list(
            "Frequency scale",
            FREQUENCY_SCALE_OPTIONS.as_slice(),
            Some(self.frequency_scale),
            |scale| SettingsMessage::Spectrogram(Message::FrequencyScale(scale)),
        )
        .spacing(ROW_SPACING);

        let zero_padding_row = labeled_pick_list(
            "Zero padding",
            ZERO_PADDING_OPTIONS.as_slice(),
            Some(self.config.zero_padding_factor),
            |value| SettingsMessage::Spectrogram(Message::ZeroPadding(value)),
        )
        .spacing(ROW_SPACING);

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

        let mut body = column![paired_pick_lists].spacing(CONTROL_SPACING);

        if self.window == WindowPreset::PlanckBessel {
            let (epsilon, beta) = self.planck_bessel_params();
            let epsilon_slider = labeled_slider(
                "Planck-Bessel epsilon",
                epsilon,
                format!("{epsilon:.3}"),
                PLANCK_BESSEL_EPSILON_RANGE,
                |value| SettingsMessage::Spectrogram(Message::PlanckBesselEpsilon(value)),
            );

            let beta_slider = labeled_slider(
                "Planck-Bessel beta",
                beta,
                format!("{beta:.2}"),
                PLANCK_BESSEL_BETA_RANGE,
                |value| SettingsMessage::Spectrogram(Message::PlanckBesselBeta(value)),
            );

            body = body.push(column![epsilon_slider, beta_slider].spacing(CONTROL_SPACING));
        }

        body = body.push(history_slider);

        section_container("Core controls", body)
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
            |value| SettingsMessage::Spectrogram(Message::SynchroBinCount(value)),
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

    fn colors_section(&self) -> container::Container<'_, SettingsMessage> {
        let controls = self
            .palette
            .view()
            .map(|event| SettingsMessage::Spectrogram(Message::Palette(event)));

        section_container("Colors", column![controls].spacing(CONTROL_SPACING))
    }

    fn planck_bessel_params(&self) -> (f32, f32) {
        (self.planck_bessel.epsilon, self.planck_bessel.beta)
    }
}

#[derive(Debug, Clone)]
pub enum Message {
    FftSize(usize),
    HopRatio(HopRatio),
    HistoryLength(f32),
    Window(WindowPreset),
    PlanckBesselEpsilon(f32),
    PlanckBesselBeta(f32),
    FrequencyScale(FrequencyScale),
    UseReassignment(bool),
    ReassignmentFloor(f32),
    ZeroPadding(usize),
    UseSynchrosqueezing(bool),
    SynchroBinCount(f32),
    TemporalSmoothing(f32),
    TemporalSmoothingMaxHz(f32),
    TemporalSmoothingBlendHz(f32),
    FrequencySmoothing(f32),
    FrequencySmoothingMaxHz(f32),
    FrequencySmoothingBlendHz(f32),
    Palette(PaletteEvent),
}

pub fn create(
    visual_id: VisualId,
    visual_manager: &VisualManagerHandle,
) -> SpectrogramSettingsPane {
    let stored_settings = visual_manager
        .borrow()
        .module_settings(VisualKind::SPECTROGRAM)
        .and_then(|stored| stored.config::<SpectrogramSettings>());

    let mut config = stored_settings
        .as_ref()
        .map(|settings| settings.to_config())
        .unwrap_or_default();

    if !config.use_reassignment {
        config.use_synchrosqueezing = false;
    }

    let palette = stored_settings
        .as_ref()
        .and_then(|settings| settings.palette_array::<SPECTROGRAM_PALETTE_SIZE>())
        .unwrap_or(theme::DEFAULT_SPECTROGRAM_PALETTE);

    let window = WindowPreset::from_kind(config.window);
    let frequency_scale = config.frequency_scale;
    let hop_ratio = HopRatio::from_config(config.fft_size, config.hop_size);

    let mut planck_bessel = PlanckBesselParams::default();
    if let WindowKind::PlanckBessel { epsilon, beta } = config.window {
        planck_bessel = PlanckBesselParams { epsilon, beta };
    }

    SpectrogramSettingsPane {
        visual_id,
        config,
        window,
        frequency_scale,
        hop_ratio,
        palette: PaletteEditor::new(&palette, &theme::DEFAULT_SPECTROGRAM_PALETTE),
        planck_bessel,
    }
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
        let make_divider = || {
            Rule::horizontal(RULE_HEIGHT).style(move |theme: &iced::Theme| rule::Style {
                color: theme::with_alpha(theme.extended_palette().secondary.weak.text, 0.35),
                width: RULE_THICKNESS,
                radius: 0.0.into(),
                fill_mode: rule::FillMode::Percent(RULE_FILL_PERCENT),
            })
        };

        column![
            self.core_section(),
            make_divider(),
            self.advanced_section(),
            make_divider(),
            self.colors_section()
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
                if set_if_changed(&mut self.config.fft_size, *size) {
                    self.config.hop_size = self.hop_ratio.to_hop_size(*size);
                    changed = true;
                }
            }
            Message::HopRatio(ratio) => {
                if self.hop_ratio != *ratio {
                    self.hop_ratio = *ratio;
                    let new_hop = ratio.to_hop_size(self.config.fft_size);
                    if set_if_changed(&mut self.config.hop_size, new_hop) {
                        changed = true;
                    }
                }
            }
            Message::HistoryLength(value) => {
                changed |=
                    update_usize_from_f32(&mut self.config.history_length, *value, HISTORY_RANGE);
            }
            Message::Window(preset) => {
                if self.window != *preset {
                    if let WindowKind::PlanckBessel { epsilon, beta } = self.config.window {
                        self.planck_bessel = PlanckBesselParams { epsilon, beta };
                    }
                    self.window = *preset;
                    self.config.window = match preset {
                        WindowPreset::PlanckBessel => WindowKind::PlanckBessel {
                            epsilon: self.planck_bessel.epsilon,
                            beta: self.planck_bessel.beta,
                        },
                        _ => preset.to_window_kind(),
                    };
                    changed = true;
                }
            }
            Message::PlanckBesselEpsilon(value) => {
                if let WindowKind::PlanckBessel { epsilon, beta } = self.config.window {
                    let mut new_epsilon = epsilon;
                    if update_f32_range(&mut new_epsilon, *value, PLANCK_BESSEL_EPSILON_RANGE) {
                        self.planck_bessel.epsilon = new_epsilon;
                        self.config.window = WindowKind::PlanckBessel {
                            epsilon: new_epsilon,
                            beta,
                        };
                        changed = true;
                    }
                }
            }
            Message::PlanckBesselBeta(value) => {
                if let WindowKind::PlanckBessel { epsilon, beta } = self.config.window {
                    let mut new_beta = beta;
                    if update_f32_range(&mut new_beta, *value, PLANCK_BESSEL_BETA_RANGE) {
                        self.planck_bessel.beta = new_beta;
                        self.config.window = WindowKind::PlanckBessel {
                            epsilon,
                            beta: new_beta,
                        };
                        changed = true;
                    }
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
                if set_if_changed(&mut self.config.use_reassignment, *value) {
                    if !self.config.use_reassignment {
                        self.config.use_synchrosqueezing = false;
                    }
                    changed = true;
                }
            }
            Message::ReassignmentFloor(value) => {
                changed |= update_f32_range(
                    &mut self.config.reassignment_power_floor_db,
                    *value,
                    REASSIGNMENT_FLOOR_RANGE,
                );
            }
            Message::ZeroPadding(value) => {
                changed |= set_if_changed(&mut self.config.zero_padding_factor, *value);
            }
            Message::UseSynchrosqueezing(value) => {
                let desired = *value && self.config.use_reassignment;
                if set_if_changed(&mut self.config.use_synchrosqueezing, desired) {
                    changed = true;
                }
            }
            Message::SynchroBinCount(value) => {
                changed |= self.config.use_synchrosqueezing
                    && update_usize_from_f32(
                        &mut self.config.synchrosqueezing_bin_count,
                        *value,
                        SYNCHRO_BINS_RANGE,
                    );
            }
            Message::TemporalSmoothing(value) => {
                changed |= self.config.use_synchrosqueezing
                    && update_f32_range(
                        &mut self.config.temporal_smoothing,
                        *value,
                        TEMPORAL_SMOOTHING_RANGE,
                    );
            }
            Message::TemporalSmoothingMaxHz(value) => {
                changed |= self.config.use_synchrosqueezing
                    && update_f32_range(
                        &mut self.config.temporal_smoothing_max_hz,
                        *value,
                        TEMPORAL_MAX_HZ_RANGE,
                    );
            }
            Message::TemporalSmoothingBlendHz(value) => {
                changed |= self.config.use_synchrosqueezing
                    && update_f32_range(
                        &mut self.config.temporal_smoothing_blend_hz,
                        *value,
                        TEMPORAL_BLEND_HZ_RANGE,
                    );
            }
            Message::FrequencySmoothing(value) => {
                changed |= self.config.use_synchrosqueezing
                    && update_usize_from_f32(
                        &mut self.config.frequency_smoothing_radius,
                        *value,
                        FREQUENCY_SMOOTHING_RANGE,
                    );
            }
            Message::FrequencySmoothingMaxHz(value) => {
                changed |= self.config.use_synchrosqueezing
                    && update_f32_range(
                        &mut self.config.frequency_smoothing_max_hz,
                        *value,
                        FREQUENCY_MAX_HZ_RANGE,
                    );
            }
            Message::FrequencySmoothingBlendHz(value) => {
                changed |= self.config.use_synchrosqueezing
                    && update_f32_range(
                        &mut self.config.frequency_smoothing_blend_hz,
                        *value,
                        FREQUENCY_BLEND_HZ_RANGE,
                    );
            }
            Message::Palette(event) => {
                changed |= self.palette.update(*event);
            }
        }

        if changed {
            self.persist(visual_manager, settings);
        }
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
    const ALL: [Self; 5] = [
        Self::Rectangular,
        Self::Hann,
        Self::Hamming,
        Self::Blackman,
        Self::PlanckBessel,
    ];

    fn from_kind(kind: WindowKind) -> Self {
        match kind {
            WindowKind::Rectangular => Self::Rectangular,
            WindowKind::Hann => Self::Hann,
            WindowKind::Hamming => Self::Hamming,
            WindowKind::Blackman => Self::Blackman,
            WindowKind::PlanckBessel { .. } => Self::PlanckBessel,
        }
    }

    fn to_window_kind(self) -> WindowKind {
        match self {
            Self::Rectangular => WindowKind::Rectangular,
            Self::Hann => WindowKind::Hann,
            Self::Hamming => WindowKind::Hamming,
            Self::Blackman => WindowKind::Blackman,
            Self::PlanckBessel => WindowKind::PlanckBessel {
                epsilon: PLANCK_BESSEL_DEFAULT_EPSILON,
                beta: PLANCK_BESSEL_DEFAULT_BETA,
            },
        }
    }
}

impl fmt::Display for WindowPreset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Rectangular => "Rectangular",
            Self::Hann => "Hann",
            Self::Hamming => "Hamming",
            Self::Blackman => "Blackman",
            Self::PlanckBessel => "Planck-Bessel",
        })
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
    const ALL: [Self; 4] = [Self::Quarter, Self::Sixth, Self::Eighth, Self::Sixteenth];

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
            Self::Quarter => 4,
            Self::Sixth => 6,
            Self::Eighth => 8,
            Self::Sixteenth => 16,
        }
    }

    fn to_hop_size(self, fft_size: usize) -> usize {
        (fft_size / self.divisor().max(1)).max(1)
    }
}

impl fmt::Display for HopRatio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Quarter => "75% overlap",
            Self::Sixth => "83% overlap",
            Self::Eighth => "87% overlap",
            Self::Sixteenth => "94% overlap",
        })
    }
}

impl fmt::Display for FrequencyScale {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            FrequencyScale::Linear => "Linear",
            FrequencyScale::Logarithmic => "Logarithmic",
            FrequencyScale::Mel => "Mel",
        })
    }
}
