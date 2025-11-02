const A440_HZ: f32 = 440.0;
const A440_MIDI: i32 = 69;
const SEMITONES_PER_OCTAVE: i32 = 12;
const MIDI_OCTAVE_OFFSET: i32 = 1;

const NOTE_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MusicalNote {
    pub midi_number: i32,
    pub name: &'static str,
    pub octave: i32,
}

impl MusicalNote {
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

    pub fn format(&self) -> String {
        format!("{}{}", self.name, self.octave)
    }
}
