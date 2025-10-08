pub mod batcher;

use pipewire::spa;
use std::convert::TryInto;

pub use batcher::SampleBatcher;

/// Default sample rate (Hz) used throughout the audio pipeline.
pub const DEFAULT_SAMPLE_RATE_HZ: u32 = 48_000;
/// Default sample rate represented as `f32` for DSP usage.
pub const DEFAULT_SAMPLE_RATE: f32 = DEFAULT_SAMPLE_RATE_HZ as f32;

/// Return the number of bytes per sample for the given PipeWire audio format.
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

/// Convert audio samples in any supported format into normalised `f32` samples.
pub fn convert_samples_to_f32(
    bytes: &[u8],
    format: spa::param::audio::AudioFormat,
) -> Option<Vec<f32>> {
    use spa::param::audio::AudioFormat as Fmt;

    let sample_bytes = bytes_per_sample(format)?;
    if bytes.len() % sample_bytes != 0 {
        eprintln!(
            "[virtual-sink] buffer length {} is not aligned to {:?}",
            bytes.len(),
            format
        );
        return None;
    }

    let mut samples = Vec::with_capacity(bytes.len() / sample_bytes);
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
        Fmt::S16LE => {
            for chunk in bytes.chunks_exact(2) {
                let sample = i16::from_le_bytes(chunk.try_into().unwrap());
                samples.push(sample as f32 / i16::MAX as f32);
            }
        }
        Fmt::S16BE => {
            for chunk in bytes.chunks_exact(2) {
                let sample = i16::from_be_bytes(chunk.try_into().unwrap());
                samples.push(sample as f32 / i16::MAX as f32);
            }
        }
        Fmt::S32LE | Fmt::S24_32LE => {
            for chunk in bytes.chunks_exact(4) {
                let sample = i32::from_le_bytes(chunk.try_into().unwrap());
                samples.push(sample as f32 / i32::MAX as f32);
            }
        }
        Fmt::S32BE | Fmt::S24_32BE => {
            for chunk in bytes.chunks_exact(4) {
                let sample = i32::from_be_bytes(chunk.try_into().unwrap());
                samples.push(sample as f32 / i32::MAX as f32);
            }
        }
        Fmt::U16LE => {
            for chunk in bytes.chunks_exact(2) {
                let sample = u16::from_le_bytes(chunk.try_into().unwrap());
                samples.push((sample as f32 - 32_768.0) / 32_768.0);
            }
        }
        Fmt::U16BE => {
            for chunk in bytes.chunks_exact(2) {
                let sample = u16::from_be_bytes(chunk.try_into().unwrap());
                samples.push((sample as f32 - 32_768.0) / 32_768.0);
            }
        }
        Fmt::U8 => {
            for &byte in bytes {
                samples.push((byte as f32 - 128.0) / 128.0);
            }
        }
        Fmt::S8 => {
            for &byte in bytes {
                let sample = byte as i8;
                samples.push(sample as f32 / i8::MAX as f32);
            }
        }
        _ => return None,
    }

    Some(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_sample_rate_constants_align() {
        assert_eq!(DEFAULT_SAMPLE_RATE_HZ as f32, DEFAULT_SAMPLE_RATE);
    }

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
    }

    #[test]
    fn convert_s16le_samples_to_f32() {
        let bytes = [0x00, 0x80, 0xFF, 0x7F];
        let converted = convert_samples_to_f32(&bytes, spa::param::audio::AudioFormat::S16LE)
            .expect("conversion should succeed");
        assert_eq!(converted.len(), 2);
        assert!((converted[0] + 1.000_030_5).abs() < 1e-5);
        assert!((converted[1] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn conversion_fails_for_unsupported_format() {
        let bytes = [0u8; 4];
        assert!(convert_samples_to_f32(&bytes, spa::param::audio::AudioFormat::Unknown,).is_none());
    }
}
