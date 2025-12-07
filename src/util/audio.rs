pub mod batcher;
pub mod musical;

use pipewire::spa;
use std::convert::TryInto;
use tracing::warn;

pub use batcher::SampleBatcher;

/// Default sample rate (Hz) used throughout the audio pipeline.
pub const DEFAULT_SAMPLE_RATE: f32 = 48_000.0;

// decibel conversion constants/utils

/// Floor value (dB) below which magnitudes are clamped.
pub const DB_FLOOR: f32 = -140.0;

/// Minimum power value to avoid log(0) in dB conversions.
const POWER_EPSILON: f32 = 1.0e-20;

/// Natural log to decibel conversion factor: 10 / ln(10) ~= 4.342944819.
const LN_TO_DB: f32 = 4.342_944_8;

/// Convert power (magnitude squared) to decibels.
#[inline(always)]
pub fn power_to_db(power: f32) -> f32 {
    if power > POWER_EPSILON {
        (power.ln() * LN_TO_DB).max(DB_FLOOR)
    } else {
        DB_FLOOR
    }
}

#[inline(always)]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

pub fn bytes_per_sample(format: spa::param::audio::AudioFormat) -> Option<usize> {
    use spa::param::audio::AudioFormat as Fmt;

    match format {
        Fmt::F32LE
        | Fmt::F32BE
        | Fmt::S24_32LE
        | Fmt::S24_32BE
        | Fmt::S32LE
        | Fmt::S32BE
        | Fmt::U32LE
        | Fmt::U32BE => Some(4),
        Fmt::F64LE | Fmt::F64BE => Some(8),
        Fmt::S16LE | Fmt::S16BE | Fmt::U16LE | Fmt::U16BE => Some(2),
        Fmt::S8 | Fmt::U8 => Some(1),
        _ => None,
    }
}

pub fn convert_samples_to_f32(
    bytes: &[u8],
    format: spa::param::audio::AudioFormat,
) -> Option<Vec<f32>> {
    use spa::param::audio::AudioFormat as Fmt;

    let sample_bytes = bytes_per_sample(format)?;
    if !bytes.len().is_multiple_of(sample_bytes) {
        warn!(
            "[virtual-sink] buffer length {} is not aligned to {:?}",
            bytes.len(),
            format
        );
        return None;
    }

    let sample_count = bytes.len() / sample_bytes;
    let mut samples = Vec::with_capacity(sample_count);

    /// Helper macro to reduce duplication for integer format conversions.
    macro_rules! convert_int {
        ($ty:ty, $endian:ident, $divisor:expr, $unsigned_offset:expr) => {{
            for chunk in bytes.chunks_exact(std::mem::size_of::<$ty>()) {
                let raw = <$ty>::$endian(chunk.try_into().unwrap());
                let normalized = if $unsigned_offset != 0.0 {
                    (raw as f32 - $unsigned_offset) / $unsigned_offset
                } else {
                    raw as f32 / $divisor
                };
                samples.push(normalized);
            }
        }};
    }

    match format {
        Fmt::F32LE => {
            for chunk in bytes.chunks_exact(4) {
                samples.push(f32::from_le_bytes(chunk.try_into().unwrap()));
            }
        }
        Fmt::F32BE => {
            for chunk in bytes.chunks_exact(4) {
                samples.push(f32::from_be_bytes(chunk.try_into().unwrap()));
            }
        }
        Fmt::F64LE => {
            for chunk in bytes.chunks_exact(8) {
                samples.push(f64::from_le_bytes(chunk.try_into().unwrap()) as f32);
            }
        }
        Fmt::F64BE => {
            for chunk in bytes.chunks_exact(8) {
                samples.push(f64::from_be_bytes(chunk.try_into().unwrap()) as f32);
            }
        }
        Fmt::S16LE => convert_int!(i16, from_le_bytes, i16::MAX as f32, 0.0),
        Fmt::S16BE => convert_int!(i16, from_be_bytes, i16::MAX as f32, 0.0),
        Fmt::S32LE | Fmt::S24_32LE => convert_int!(i32, from_le_bytes, i32::MAX as f32, 0.0),
        Fmt::S32BE | Fmt::S24_32BE => convert_int!(i32, from_be_bytes, i32::MAX as f32, 0.0),
        Fmt::U16LE => convert_int!(u16, from_le_bytes, 32_768.0, 32_768.0),
        Fmt::U16BE => convert_int!(u16, from_be_bytes, 32_768.0, 32_768.0),
        Fmt::U8 => {
            for &byte in bytes {
                samples.push((byte as f32 - 128.0) / 128.0);
            }
        }
        Fmt::S8 => {
            for &byte in bytes {
                samples.push((byte as i8) as f32 / i8::MAX as f32);
            }
        }
        _ => return None,
    }

    Some(samples)
}

pub fn mixdown_into_deque(
    buffer: &mut std::collections::VecDeque<f32>,
    samples: &[f32],
    channels: usize,
) {
    if channels == 0 || samples.is_empty() {
        return;
    }

    if channels == 1 {
        buffer.extend(samples);
        return;
    }

    let frame_count = samples.len() / channels;
    buffer.reserve(frame_count);

    let inv = 1.0 / channels as f32;
    for frame in samples.chunks_exact(channels) {
        let sum: f32 = frame.iter().sum();
        buffer.push_back(sum * inv);
    }
}

#[inline]
pub fn apply_window(buffer: &mut [f32], window: &[f32]) {
    debug_assert_eq!(buffer.len(), window.len());
    for (sample, coeff) in buffer.iter_mut().zip(window.iter()) {
        *sample *= *coeff;
    }
}

pub fn remove_dc(buffer: &mut [f32]) {
    if buffer.is_empty() {
        return;
    }

    let mean = buffer.iter().sum::<f32>() / buffer.len() as f32;
    if mean.abs() <= f32::EPSILON {
        return;
    }

    for sample in buffer.iter_mut() {
        *sample -= mean;
    }
}

/// Convert dB to linear power: 10^(db/10).
#[inline(always)]
pub fn db_to_power(db: f32) -> f32 {
    const DB_TO_LOG2: f32 = 0.1 * core::f32::consts::LOG2_10;
    (db * DB_TO_LOG2).exp2()
}

/// Convert frequency in Hz to mel scale.
#[inline(always)]
pub fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert mel scale to frequency in Hz.
#[inline(always)]
pub fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

/// Copy from VecDeque to a contiguous slice, handling wraparound.
#[inline]
pub fn copy_from_deque(dst: &mut [f32], src: &std::collections::VecDeque<f32>) {
    let len = dst.len().min(src.len());
    let (head, tail) = src.as_slices();
    if head.len() >= len {
        dst[..len].copy_from_slice(&head[..len]);
    } else {
        let split = head.len();
        dst[..split].copy_from_slice(head);
        dst[split..len].copy_from_slice(&tail[..len - split]);
    }
}

pub fn compute_fft_bin_normalization(window: &[f32], fft_size: usize) -> Vec<f32> {
    let bins = fft_size / 2 + 1;
    if bins == 0 {
        return Vec::new();
    }

    let window_sum: f32 = window.iter().sum();
    let inv_sum = if window_sum.abs() > f32::EPSILON {
        1.0 / window_sum
    } else if fft_size > 0 {
        1.0 / fft_size as f32
    } else {
        0.0
    };

    let dc_scale = inv_sum * inv_sum;
    let ac_scale = 4.0 * dc_scale;
    let mut norms = vec![ac_scale; bins];
    norms[0] = dc_scale;
    if bins > 1 {
        norms[bins - 1] = dc_scale;
    }
    norms
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_per_sample_matches_expected_widths() {
        assert_eq!(
            bytes_per_sample(spa::param::audio::AudioFormat::F32LE),
            Some(4)
        );
        assert_eq!(
            bytes_per_sample(spa::param::audio::AudioFormat::F64LE),
            Some(8)
        );
        assert_eq!(
            bytes_per_sample(spa::param::audio::AudioFormat::S16LE),
            Some(2)
        );
        assert_eq!(
            bytes_per_sample(spa::param::audio::AudioFormat::S8),
            Some(1)
        );
        assert_eq!(
            bytes_per_sample(spa::param::audio::AudioFormat::Unknown),
            None
        );
    }

    #[test]
    fn convert_s16le_samples_to_f32() {
        let bytes = [0x00, 0x80, 0xFF, 0x7F]; // -32768 (min), 32767 (max)
        let converted = convert_samples_to_f32(&bytes, spa::param::audio::AudioFormat::S16LE)
            .expect("conversion should succeed");
        assert_eq!(converted.len(), 2);
        assert!((converted[0] + 1.000_030_5).abs() < 1e-5);
        assert!((converted[1] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn conversion_fails_for_unsupported_format() {
        let bytes = [0u8; 4];
        assert!(convert_samples_to_f32(&bytes, spa::param::audio::AudioFormat::Unknown).is_none());
    }
}
