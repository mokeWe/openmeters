use super::palette::{PaletteEditor, PaletteEvent};
use super::widgets::{
    SliderRange, labeled_pick_list, labeled_slider, set_if_changed, update_f32_range,
    update_usize_from_f32,
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
use iced::widget::text::Wrapping;
use iced::widget::{column, container, row, rule, text, toggler};
use iced::{Element, Length};
use std::fmt;

const FFT_OPTIONS: [usize; 5] = [1024, 2048, 4096, 8192, 16384];
const ZERO_PAD_OPTIONS: [usize; 6] = [1, 2, 4, 8, 16, 32];
const FREQ_SCALE_OPTIONS: [FrequencyScale; 3] = [
    FrequencyScale::Linear,
    FrequencyScale::Logarithmic,
    FrequencyScale::Mel,
];
const HISTORY_RANGE: SliderRange = SliderRange::new(120.0, 960.0, 30.0);
const REASSIGN_FLOOR_RANGE: SliderRange = SliderRange::new(-120.0, -30.0, 1.0);
const DISPLAY_BINS_RANGE: SliderRange = SliderRange::new(64.0, 4096.0, 64.0);
const PB_EPSILON_RANGE: SliderRange = SliderRange::new(0.01, 0.5, 0.01);
const PB_BETA_RANGE: SliderRange = SliderRange::new(0.0, 20.0, 0.25);

#[derive(Debug)]
pub struct SpectrogramSettingsPane {
    visual_id: VisualId,
    config: SpectrogramConfig,
    palette: PaletteEditor,
    planck_bessel: (f32, f32),
}

impl SpectrogramSettingsPane {
    fn persist(&self, visual_manager: &VisualManagerHandle, settings: &SettingsHandle) {
        let mut stored = SpectrogramSettings::from_config(&self.config);
        stored.palette = PaletteSettings::maybe_from_colors(
            self.palette.colors(),
            &theme::DEFAULT_SPECTROGRAM_PALETTE,
        );
        if visual_manager.borrow_mut().apply_module_settings(
            VisualKind::SPECTROGRAM,
            &ModuleSettings::with_config(&stored),
        ) {
            settings.update(|s| s.set_module_config(VisualKind::SPECTROGRAM, &stored));
        }
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
    DisplayBinCount(f32),
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

    let config = stored_settings
        .as_ref()
        .map(|settings| settings.to_config())
        .unwrap_or_default();

    let palette = stored_settings
        .as_ref()
        .and_then(|settings| settings.palette_array::<SPECTROGRAM_PALETTE_SIZE>())
        .unwrap_or(theme::DEFAULT_SPECTROGRAM_PALETTE);

    let planck_bessel = match config.window {
        WindowKind::PlanckBessel { epsilon, beta } => (epsilon, beta),
        _ => (PLANCK_BESSEL_DEFAULT_EPSILON, PLANCK_BESSEL_DEFAULT_BETA),
    };

    SpectrogramSettingsPane {
        visual_id,
        config,
        palette: PaletteEditor::new(&palette, &theme::DEFAULT_SPECTROGRAM_PALETTE),
        planck_bessel,
    }
}

fn divider<'a>() -> rule::Rule<'a, iced::Theme> {
    rule::horizontal(1).style(|_| rule::Style {
        color: theme::with_alpha(theme::accent_primary(), 0.35),
        radius: 0.0.into(),
        fill_mode: rule::FillMode::Percent(82.0),
        snap: true,
    })
}

fn section<'a>(
    title: &'static str,
    body: iced::widget::Column<'a, SettingsMessage>,
) -> container::Container<'a, SettingsMessage> {
    container(
        column![
            container(text(title).size(14).wrapping(Wrapping::None)).clip(true),
            body
        ]
        .spacing(10),
    )
    .padding(12)
}

impl ModuleSettingsPane for SpectrogramSettingsPane {
    fn visual_id(&self) -> VisualId {
        self.visual_id
    }

    fn view(&self) -> Element<'_, SettingsMessage> {
        let window = WindowPreset::from_kind(self.config.window);
        let hop_ratio = HopRatio::from_config(self.config.fft_size, self.config.hop_size);

        let left_col = column![
            labeled_pick_list("FFT size", &FFT_OPTIONS, Some(self.config.fft_size), |v| {
                SettingsMessage::Spectrogram(Message::FftSize(v))
            })
            .spacing(10),
            labeled_pick_list("Hop overlap", &HopRatio::ALL, Some(hop_ratio), |v| {
                SettingsMessage::Spectrogram(Message::HopRatio(v))
            })
            .spacing(10),
        ]
        .spacing(8);

        let right_col = column![
            labeled_pick_list("Window", &WindowPreset::ALL, Some(window), |v| {
                SettingsMessage::Spectrogram(Message::Window(v))
            })
            .spacing(10),
            labeled_pick_list(
                "Freq scale",
                &FREQ_SCALE_OPTIONS,
                Some(self.config.frequency_scale),
                |v| SettingsMessage::Spectrogram(Message::FrequencyScale(v))
            )
            .spacing(10),
            labeled_pick_list(
                "Zero pad",
                &ZERO_PAD_OPTIONS,
                Some(self.config.zero_padding_factor),
                |v| SettingsMessage::Spectrogram(Message::ZeroPadding(v))
            )
            .spacing(10),
        ]
        .spacing(8);

        let mut core =
            column![row![left_col, right_col].spacing(10).width(Length::Fill)].spacing(8);
        if let WindowKind::PlanckBessel { epsilon, beta } = self.config.window {
            core = core.push(labeled_slider(
                "PB epsilon",
                epsilon,
                format!("{epsilon:.3}"),
                PB_EPSILON_RANGE,
                |v| SettingsMessage::Spectrogram(Message::PlanckBesselEpsilon(v)),
            ));
            core = core.push(labeled_slider(
                "PB beta",
                beta,
                format!("{beta:.2}"),
                PB_BETA_RANGE,
                |v| SettingsMessage::Spectrogram(Message::PlanckBesselBeta(v)),
            ));
        }
        core = core.push(labeled_slider(
            "History length",
            self.config.history_length as f32,
            format!("{} cols", self.config.history_length),
            HISTORY_RANGE,
            |v| SettingsMessage::Spectrogram(Message::HistoryLength(v)),
        ));

        let reassign_toggle = toggler(self.config.use_reassignment)
            .label("Time-frequency reassignment")
            .text_size(11)
            .spacing(4)
            .on_toggle(|v| SettingsMessage::Spectrogram(Message::UseReassignment(v)));
        let mut adv = column![reassign_toggle].spacing(8);
        if self.config.use_reassignment {
            adv = adv.push(labeled_slider(
                "Reassign floor",
                self.config.reassignment_power_floor_db,
                format!("{:.0} dB", self.config.reassignment_power_floor_db),
                REASSIGN_FLOOR_RANGE,
                |v| SettingsMessage::Spectrogram(Message::ReassignmentFloor(v)),
            ));
            adv = adv.push(labeled_slider(
                "Display bins",
                self.config.display_bin_count as f32,
                format!("{} bins", self.config.display_bin_count),
                DISPLAY_BINS_RANGE,
                |v| SettingsMessage::Spectrogram(Message::DisplayBinCount(v)),
            ));
        }

        let colors = column![
            self.palette
                .view()
                .map(|e| SettingsMessage::Spectrogram(Message::Palette(e)))
        ]
        .spacing(8);

        column![
            section("Core controls", core),
            divider(),
            section("Advanced", adv),
            divider(),
            section("Colors", colors)
        ]
        .spacing(14)
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
                let hop_ratio = HopRatio::from_config(self.config.fft_size, self.config.hop_size);
                if set_if_changed(&mut self.config.fft_size, *size) {
                    self.config.hop_size = hop_ratio.to_hop_size(*size);
                    changed = true;
                }
            }
            Message::HopRatio(ratio) => {
                let new_hop = ratio.to_hop_size(self.config.fft_size);
                changed |= set_if_changed(&mut self.config.hop_size, new_hop);
            }
            Message::HistoryLength(value) => {
                changed |=
                    update_usize_from_f32(&mut self.config.history_length, *value, HISTORY_RANGE);
            }
            Message::Window(preset) => {
                let current = WindowPreset::from_kind(self.config.window);
                if current != *preset {
                    if let WindowKind::PlanckBessel { epsilon, beta } = self.config.window {
                        self.planck_bessel = (epsilon, beta);
                    }
                    self.config.window = match preset {
                        WindowPreset::PlanckBessel => WindowKind::PlanckBessel {
                            epsilon: self.planck_bessel.0,
                            beta: self.planck_bessel.1,
                        },
                        _ => preset.to_window_kind(),
                    };
                    changed = true;
                }
            }
            Message::PlanckBesselEpsilon(value) => {
                if let WindowKind::PlanckBessel { epsilon, beta } = self.config.window {
                    let mut new_epsilon = epsilon;
                    if update_f32_range(&mut new_epsilon, *value, PB_EPSILON_RANGE) {
                        self.planck_bessel.0 = new_epsilon;
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
                    if update_f32_range(&mut new_beta, *value, PB_BETA_RANGE) {
                        self.planck_bessel.1 = new_beta;
                        self.config.window = WindowKind::PlanckBessel {
                            epsilon,
                            beta: new_beta,
                        };
                        changed = true;
                    }
                }
            }
            Message::FrequencyScale(scale) => {
                changed |= set_if_changed(&mut self.config.frequency_scale, *scale);
            }
            Message::UseReassignment(value) => {
                changed |= set_if_changed(&mut self.config.use_reassignment, *value);
            }
            Message::ReassignmentFloor(value) => {
                changed |= update_f32_range(
                    &mut self.config.reassignment_power_floor_db,
                    *value,
                    REASSIGN_FLOOR_RANGE,
                );
            }
            Message::ZeroPadding(value) => {
                changed |= set_if_changed(&mut self.config.zero_padding_factor, *value);
            }
            Message::DisplayBinCount(value) => {
                changed |= self.config.use_reassignment
                    && update_usize_from_f32(
                        &mut self.config.display_bin_count,
                        *value,
                        DISPLAY_BINS_RANGE,
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
        (fft_size / self.divisor()).max(1)
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
        write!(f, "{self:?}")
    }
}
