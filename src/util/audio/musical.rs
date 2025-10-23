/// Standard A440 tuning reference frequency.
const A440_HZ: f32 = 440.0;
/// MIDI note number for A440 (A4).
const A440_MIDI: i32 = 69;
/// Number of semitones in one octave (12-tone equal temperament).
const SEMITONES_PER_OCTAVE: i32 = 12;
/// MIDI octave offset (C0 starts at MIDI 12, so octave = midi/12 - 1).
const MIDI_OCTAVE_OFFSET: i32 = 1;

/// Note names in one octave starting from C.
const NOTE_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/// Musical note representation with name and octave.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MusicalNote {
    /// MIDI note number (0-127, where 69 = A440).
    pub midi_number: i32,
    /// Note name with sharp notation (e.g., "C#", "A", "F#").
    pub name: &'static str,
    /// Octave number (e.g., 4 for middle C = C4).
    pub octave: i32,
}

impl MusicalNote {
    /// Convert a frequency in Hz to the nearest musical note using 12-tone equal temperament.
    /// Returns `None` for non-positive or non-finite frequencies.
    pub fn from_frequency(freq_hz: f32) -> Option<Self> {
        if freq_hz <= 0.0 || !freq_hz.is_finite() {
            return None;
        }

        // Calculate MIDI note number using 12-TET formula: midi = 69 + 12 * log2(freq / 440)
        let midi_float =
            A440_MIDI as f32 + SEMITONES_PER_OCTAVE as f32 * (freq_hz / A440_HZ).log2();
        let midi_number = midi_float.round() as i32;

        // Calculate note index (0-11) with proper negative wrap-around
        let note_index = ((midi_number % SEMITONES_PER_OCTAVE + SEMITONES_PER_OCTAVE)
            % SEMITONES_PER_OCTAVE) as usize;
        let octave = (midi_number / SEMITONES_PER_OCTAVE) - MIDI_OCTAVE_OFFSET;

        Some(Self {
            midi_number,
            name: NOTE_NAMES[note_index],
            octave,
        })
    }

    /// Format the note as a string (e.g., "A4", "C#5").
    pub fn format(&self) -> String {
        format!("{}{}", self.name, self.octave)
    }
}
