use super::{ModuleSettingsPane, SettingsMessage};
use crate::dsp::spectrogram::{SpectrogramConfig, WindowKind};
use crate::ui::settings::{ModuleSettings, SettingsHandle, SpectrogramSettings};
use crate::ui::visualization::visual_manager::{VisualId, VisualKind, VisualManagerHandle};
use iced::Element;
use iced::widget::{checkbox, column, pick_list, row, slider, text};
use std::fmt;

const FFT_OPTIONS: [usize; 4] = [1024, 2048, 4096, 8192];
const ZERO_PADDING_OPTIONS: [usize; 3] = [1, 2, 4];
const HISTORY_MIN: f32 = 120.0;
const HISTORY_MAX: f32 = 960.0;
const HISTORY_STEP: f32 = 30.0;
const REASSIGNMENT_FLOOR_MIN: f32 = -120.0;
const REASSIGNMENT_FLOOR_MAX: f32 = -30.0;
const TEMPORAL_SMOOTHING_MIN: f32 = 0.0;
const TEMPORAL_SMOOTHING_MAX: f32 = 0.99;
const FREQUENCY_SMOOTHING_MIN: f32 = 0.0;
const FREQUENCY_SMOOTHING_MAX: f32 = 20.0;
const REASSIGNMENT_LOW_BIN_MIN: f32 = 0.0;
const REASSIGNMENT_LOW_BIN_MAX: f32 = 4096.0;
const SYNCHRO_BINS_MIN: f32 = 64.0;
const SYNCHRO_BINS_MAX: f32 = 4096.0;
const SYNCHRO_BINS_STEP: f32 = 64.0;
const TEMPORAL_MAX_HZ_MIN: f32 = 0.0;
const TEMPORAL_MAX_HZ_MAX: f32 = 4000.0;
const TEMPORAL_BLEND_HZ_MIN: f32 = 0.0;
const TEMPORAL_BLEND_HZ_MAX: f32 = 4000.0;
const FREQUENCY_MAX_HZ_MIN: f32 = 0.0;
const FREQUENCY_MAX_HZ_MAX: f32 = 4000.0;
const FREQUENCY_BLEND_HZ_MIN: f32 = 0.0;
const FREQUENCY_BLEND_HZ_MAX: f32 = 4000.0;

#[derive(Debug)]
pub struct SpectrogramSettingsPane {
    visual_id: VisualId,
    config: SpectrogramConfig,
    window: WindowPreset,
    hop_ratio: HopRatio,
}

#[derive(Debug, Clone)]
pub enum Message {
    FftSize(usize),
    HopRatio(HopRatio),
    HistoryLength(f32),
    Window(WindowPreset),
    UseReassignment(bool),
    ReassignmentFloor(f32),
    ReassignmentLowBinLimit(usize),
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
    let config = visual_manager
        .borrow()
        .module_settings(VisualKind::SPECTROGRAM)
        .and_then(|stored| stored.spectrogram().cloned())
        .map_or_else(SpectrogramConfig::default, |stored| {
            let mut config = SpectrogramConfig::default();
            stored.apply_to(&mut config);
            config
        });

    let window = WindowPreset::from_kind(config.window);
    let hop_ratio = HopRatio::from_config(config.fft_size, config.hop_size);

    SpectrogramSettingsPane {
        visual_id,
        config,
        window,
        hop_ratio,
    }
}

impl ModuleSettingsPane for SpectrogramSettingsPane {
    fn visual_id(&self) -> VisualId {
        self.visual_id
    }

    fn view(&self) -> Element<'_, SettingsMessage> {
        let fft_pick = pick_list(FFT_OPTIONS.to_vec(), Some(self.config.fft_size), |size| {
            SettingsMessage::Spectrogram(Message::FftSize(size))
        });

        let hop_pick = pick_list(HopRatio::ALL.to_vec(), Some(self.hop_ratio), |ratio| {
            SettingsMessage::Spectrogram(Message::HopRatio(ratio))
        });

        let zero_padding_pick = pick_list(
            ZERO_PADDING_OPTIONS.to_vec(),
            Some(self.config.zero_padding_factor),
            |value| SettingsMessage::Spectrogram(Message::ZeroPadding(value)),
        );

        let window_pick = pick_list(WindowPreset::ALL.to_vec(), Some(self.window), |preset| {
            SettingsMessage::Spectrogram(Message::Window(preset))
        });

        let history_slider = column![
            row![
                text("History length"),
                text(format!("{} columns", self.config.history_length)).size(12)
            ]
            .spacing(8),
            slider::Slider::new(
                HISTORY_MIN..=HISTORY_MAX,
                self.config.history_length as f32,
                |value| SettingsMessage::Spectrogram(Message::HistoryLength(value)),
            )
            .step(HISTORY_STEP),
        ]
        .spacing(8);

        let reassignment_floor = column![
            row![
                text("Reassignment floor"),
                text(format!("{:.0} dB", self.config.reassignment_power_floor_db)).size(12)
            ]
            .spacing(8),
            slider::Slider::new(
                REASSIGNMENT_FLOOR_MIN..=REASSIGNMENT_FLOOR_MAX,
                self.config.reassignment_power_floor_db,
                |value| SettingsMessage::Spectrogram(Message::ReassignmentFloor(value)),
            )
            .step(1.0),
        ]
        .spacing(8);

        let reassignment_low_bin = column![
            row![
                text("Reassignment low-bin limit"),
                text(format!("{} bins", self.config.reassignment_low_bin_limit)).size(12)
            ]
            .spacing(8),
            slider::Slider::new(
                REASSIGNMENT_LOW_BIN_MIN..=REASSIGNMENT_LOW_BIN_MAX,
                self.config.reassignment_low_bin_limit as f32,
                |value| {
                    let bins = value
                        .round()
                        .clamp(REASSIGNMENT_LOW_BIN_MIN, REASSIGNMENT_LOW_BIN_MAX)
                        as usize;
                    SettingsMessage::Spectrogram(Message::ReassignmentLowBinLimit(bins))
                },
            )
            .step(1.0),
        ]
        .spacing(8);

        let temporal_smoothing = column![
            row![
                text("Temporal smoothing"),
                text(format!("{:.2}", self.config.temporal_smoothing)).size(12)
            ]
            .spacing(8),
            slider::Slider::new(
                TEMPORAL_SMOOTHING_MIN..=TEMPORAL_SMOOTHING_MAX,
                self.config.temporal_smoothing,
                |value| SettingsMessage::Spectrogram(Message::TemporalSmoothing(value)),
            )
            .step(0.01),
        ]
        .spacing(8);

        let synchro_bins = column![
            row![
                text("Synchrosqueezing bins"),
                text(format!("{} bins", self.config.synchrosqueezing_bin_count)).size(12)
            ]
            .spacing(8),
            slider::Slider::new(
                SYNCHRO_BINS_MIN..=SYNCHRO_BINS_MAX,
                self.config.synchrosqueezing_bin_count as f32,
                |value| {
                    let bins = (value / SYNCHRO_BINS_STEP).round() * SYNCHRO_BINS_STEP;
                    let bins = bins.clamp(SYNCHRO_BINS_MIN, SYNCHRO_BINS_MAX) as usize;
                    SettingsMessage::Spectrogram(Message::SynchroBinCount(bins))
                },
            )
            .step(SYNCHRO_BINS_STEP),
        ]
        .spacing(8);

        let temporal_smoothing_max = column![
            row![
                text("Temporal smoothing max"),
                text(format!("{:.0} Hz", self.config.temporal_smoothing_max_hz)).size(12)
            ]
            .spacing(8),
            slider::Slider::new(
                TEMPORAL_MAX_HZ_MIN..=TEMPORAL_MAX_HZ_MAX,
                self.config.temporal_smoothing_max_hz,
                |value| SettingsMessage::Spectrogram(Message::TemporalSmoothingMaxHz(value)),
            )
            .step(1.0),
        ]
        .spacing(8);

        let temporal_smoothing_blend = column![
            row![
                text("Temporal smoothing blend"),
                text(format!("{:.0} Hz", self.config.temporal_smoothing_blend_hz)).size(12)
            ]
            .spacing(8),
            slider::Slider::new(
                TEMPORAL_BLEND_HZ_MIN..=TEMPORAL_BLEND_HZ_MAX,
                self.config.temporal_smoothing_blend_hz,
                |value| SettingsMessage::Spectrogram(Message::TemporalSmoothingBlendHz(value)),
            )
            .step(1.0),
        ]
        .spacing(8);

        let frequency_smoothing = column![
            row![
                text("Frequency smoothing"),
                text(format!("{} bins", self.config.frequency_smoothing_radius)).size(12)
            ]
            .spacing(8),
            slider::Slider::new(
                FREQUENCY_SMOOTHING_MIN..=FREQUENCY_SMOOTHING_MAX,
                self.config.frequency_smoothing_radius as f32,
                |value| SettingsMessage::Spectrogram(Message::FrequencySmoothing(value)),
            )
            .step(1.0),
        ]
        .spacing(8);

        let frequency_smoothing_max = column![
            row![
                text("Frequency smoothing max"),
                text(format!("{:.0} Hz", self.config.frequency_smoothing_max_hz)).size(12)
            ]
            .spacing(8),
            slider::Slider::new(
                FREQUENCY_MAX_HZ_MIN..=FREQUENCY_MAX_HZ_MAX,
                self.config.frequency_smoothing_max_hz,
                |value| SettingsMessage::Spectrogram(Message::FrequencySmoothingMaxHz(value)),
            )
            .step(1.0),
        ]
        .spacing(8);

        let frequency_smoothing_blend = column![
            row![
                text("Frequency smoothing blend"),
                text(format!(
                    "{:.0} Hz",
                    self.config.frequency_smoothing_blend_hz
                ))
                .size(12)
            ]
            .spacing(8),
            slider::Slider::new(
                FREQUENCY_BLEND_HZ_MIN..=FREQUENCY_BLEND_HZ_MAX,
                self.config.frequency_smoothing_blend_hz,
                |value| SettingsMessage::Spectrogram(Message::FrequencySmoothingBlendHz(value)),
            )
            .step(1.0),
        ]
        .spacing(8);

        column![
            row![
                column![
                    row![text("FFT size"), fft_pick].spacing(12),
                    row![text("Hop overlap"), hop_pick].spacing(12),
                    row![text("Window"), window_pick].spacing(12),
                    row![text("Zero padding"), zero_padding_pick].spacing(12),
                ]
                .spacing(12)
            ]
            .spacing(12),
            history_slider,
            checkbox("Time-frequency reassignment", self.config.use_reassignment,)
                .on_toggle(|value| SettingsMessage::Spectrogram(Message::UseReassignment(value))),
            reassignment_floor,
            reassignment_low_bin,
            checkbox(
                "Synchrosqueezed accumulation",
                self.config.use_synchrosqueezing,
            )
            .on_toggle(|value| SettingsMessage::Spectrogram(Message::UseSynchrosqueezing(value))),
            synchro_bins,
            temporal_smoothing,
            temporal_smoothing_max,
            temporal_smoothing_blend,
            frequency_smoothing,
            frequency_smoothing_max,
            frequency_smoothing_blend,
        ]
        .spacing(16)
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
                let columns = (value / HISTORY_STEP).round() * HISTORY_STEP;
                let columns = columns.clamp(HISTORY_MIN, HISTORY_MAX) as usize;
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
            Message::UseReassignment(value) => {
                if self.config.use_reassignment != *value {
                    self.config.use_reassignment = *value;
                    changed = true;
                }
            }
            Message::ReassignmentFloor(value) => {
                let clamped = value.clamp(REASSIGNMENT_FLOOR_MIN, REASSIGNMENT_FLOOR_MAX);
                if (self.config.reassignment_power_floor_db - clamped).abs() > f32::EPSILON {
                    self.config.reassignment_power_floor_db = clamped;
                    changed = true;
                }
            }
            Message::ReassignmentLowBinLimit(value) => {
                if self.config.reassignment_low_bin_limit != *value {
                    self.config.reassignment_low_bin_limit = *value;
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
                if self.config.use_synchrosqueezing != *value {
                    self.config.use_synchrosqueezing = *value;
                    changed = true;
                }
            }
            Message::SynchroBinCount(value) => {
                if self.config.synchrosqueezing_bin_count != *value {
                    self.config.synchrosqueezing_bin_count = (*value).max(1);
                    changed = true;
                }
            }
            Message::TemporalSmoothing(value) => {
                let clamped = value.clamp(TEMPORAL_SMOOTHING_MIN, TEMPORAL_SMOOTHING_MAX);
                if (self.config.temporal_smoothing - clamped).abs() > f32::EPSILON {
                    self.config.temporal_smoothing = clamped;
                    changed = true;
                }
            }
            Message::TemporalSmoothingMaxHz(value) => {
                let clamped = value.clamp(TEMPORAL_MAX_HZ_MIN, TEMPORAL_MAX_HZ_MAX);
                if (self.config.temporal_smoothing_max_hz - clamped).abs() > f32::EPSILON {
                    self.config.temporal_smoothing_max_hz = clamped;
                    changed = true;
                }
            }
            Message::TemporalSmoothingBlendHz(value) => {
                let clamped = value.clamp(TEMPORAL_BLEND_HZ_MIN, TEMPORAL_BLEND_HZ_MAX);
                if (self.config.temporal_smoothing_blend_hz - clamped).abs() > f32::EPSILON {
                    self.config.temporal_smoothing_blend_hz = clamped;
                    changed = true;
                }
            }
            Message::FrequencySmoothing(value) => {
                let radius = value
                    .clamp(FREQUENCY_SMOOTHING_MIN, FREQUENCY_SMOOTHING_MAX)
                    .round() as usize;
                if self.config.frequency_smoothing_radius != radius {
                    self.config.frequency_smoothing_radius = radius;
                    changed = true;
                }
            }
            Message::FrequencySmoothingMaxHz(value) => {
                let clamped = value.clamp(FREQUENCY_MAX_HZ_MIN, FREQUENCY_MAX_HZ_MAX);
                if (self.config.frequency_smoothing_max_hz - clamped).abs() > f32::EPSILON {
                    self.config.frequency_smoothing_max_hz = clamped;
                    changed = true;
                }
            }
            Message::FrequencySmoothingBlendHz(value) => {
                let clamped = value.clamp(FREQUENCY_BLEND_HZ_MIN, FREQUENCY_BLEND_HZ_MAX);
                if (self.config.frequency_smoothing_blend_hz - clamped).abs() > f32::EPSILON {
                    self.config.frequency_smoothing_blend_hz = clamped;
                    changed = true;
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

    let mut module_settings = ModuleSettings::default();
    module_settings.set_spectrogram(SpectrogramSettings::from_config(&config));

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
}

impl HopRatio {
    const ALL: [HopRatio; 3] = [HopRatio::Quarter, HopRatio::Sixth, HopRatio::Eighth];

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
        };
        write!(f, "{}", label)
    }
}
