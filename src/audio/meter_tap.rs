use super::pw_virtual_sink::{self, CaptureBuffer, CapturedAudio};
use crate::util::audio::{DEFAULT_SAMPLE_RATE, SampleBatcher};
use async_channel::{Receiver as AsyncReceiver, Sender as AsyncSender};
use parking_lot::RwLock;
use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{error, info, warn};

const CHANNEL_CAPACITY: usize = 64;
const POLL_BACKOFF: Duration = Duration::from_millis(50);
const TARGET_BATCH_SAMPLES: usize = 2_048;
const MAX_BATCH_LATENCY: Duration = Duration::from_millis(25);

static AUDIO_STREAM: OnceLock<Arc<AsyncReceiver<Vec<f32>>>> = OnceLock::new();
static FORMAT_STATE: OnceLock<RwLock<MeterFormat>> = OnceLock::new();

#[derive(Debug, Clone, Copy)]
pub struct MeterFormat {
    pub channels: usize,
    pub sample_rate: f32,
}

impl Default for MeterFormat {
    fn default() -> Self {
        Self {
            channels: 2,
            sample_rate: DEFAULT_SAMPLE_RATE,
        }
    }
}

fn format_state() -> &'static RwLock<MeterFormat> {
    FORMAT_STATE.get_or_init(|| RwLock::new(MeterFormat::default()))
}

fn update_format(channels: usize, sample_rate: f32) {
    *format_state().write() = MeterFormat {
        channels: channels.max(1),
        sample_rate: sample_rate.max(1.0),
    };
}

pub fn current_format() -> MeterFormat {
    *format_state().read()
}

pub fn audio_sample_stream() -> Arc<AsyncReceiver<Vec<f32>>> {
    AUDIO_STREAM
        .get_or_init(|| {
            let (sender, receiver) = async_channel::bounded(CHANNEL_CAPACITY);
            spawn_forwarder(sender, pw_virtual_sink::capture_buffer_handle());
            Arc::new(receiver)
        })
        .clone()
}

fn spawn_forwarder(sender: AsyncSender<Vec<f32>>, buffer: Arc<CaptureBuffer>) {
    thread::Builder::new()
        .name("openmeters-audio-meter-tap".into())
        .spawn(move || forward_loop(sender, buffer))
        .expect("failed to spawn audio meter tap thread");
}

fn forward_loop(sender: AsyncSender<Vec<f32>>, buffer: Arc<CaptureBuffer>) {
    let mut batcher = SampleBatcher::new(TARGET_BATCH_SAMPLES);
    let mut last_flush = Instant::now();
    let mut batch_channels: Option<usize> = None;
    let mut batch_sample_rate: Option<f32> = None;
    let mut last_drop_check = Instant::now();
    let mut drop_baseline = buffer.dropped_frames();

    loop {
        if last_drop_check.elapsed() >= Duration::from_secs(5) {
            let dropped = buffer.dropped_frames();
            if dropped > drop_baseline {
                warn!(
                    "[meter-tap] dropped {} capture frames (total {})",
                    dropped - drop_baseline,
                    dropped
                );
                drop_baseline = dropped;
            }
            last_drop_check = Instant::now();
        }

        match buffer.pop_wait_timeout(POLL_BACKOFF) {
            Ok(Some(packet)) => {
                if handle_packet(
                    packet,
                    &sender,
                    &mut batcher,
                    &mut batch_channels,
                    &mut batch_sample_rate,
                    &mut last_flush,
                ) {
                    break;
                }
            }
            Ok(None) => {
                if sender.is_closed() {
                    break;
                }

                if last_flush.elapsed() >= MAX_BATCH_LATENCY {
                    if flush_batch(&sender, &mut batcher) {
                        break;
                    }
                    batch_channels = None;
                    batch_sample_rate = None;
                    last_flush = Instant::now();
                }
            }
            Err(_) => {
                error!("[meter-tap] capture buffer unavailable; stopping tap");
                break;
            }
        }

        if sender.is_closed() {
            break;
        }
    }

    let _ = flush_batch(&sender, &mut batcher);
    let dropped = buffer.dropped_frames();

    info!(
        "[meter-tap] audio channel closed; {} dropped capture frames",
        dropped
    );
}

fn handle_packet(
    packet: CapturedAudio,
    sender: &AsyncSender<Vec<f32>>,
    batcher: &mut SampleBatcher,
    batch_channels: &mut Option<usize>,
    batch_sample_rate: &mut Option<f32>,
    last_flush: &mut Instant,
) -> bool {
    let channels = packet.channels.max(1) as usize;
    let sample_rate = packet.sample_rate.max(1) as f32;

    let format_changed = batch_channels.is_some_and(|c| c != channels)
        || batch_sample_rate.is_some_and(|r| (r - sample_rate).abs() > f32::EPSILON);

    if format_changed {
        if flush_batch(sender, batcher) {
            return true;
        }
        batcher.clear();
        *batch_channels = None;
        *batch_sample_rate = None;
        *last_flush = Instant::now();
    }

    update_format(channels, sample_rate);

    if !packet.samples.is_empty() {
        batcher.push(packet.samples);
        *batch_channels = Some(channels);
        *batch_sample_rate = Some(sample_rate);
    }

    if batcher.should_flush() || last_flush.elapsed() >= MAX_BATCH_LATENCY {
        if flush_batch(sender, batcher) {
            return true;
        }
        *batch_channels = None;
        *batch_sample_rate = None;
        *last_flush = Instant::now();
    }

    false
}

fn flush_batch(sender: &AsyncSender<Vec<f32>>, batcher: &mut SampleBatcher) -> bool {
    match batcher.take() {
        Some(batch) => sender.send_blocking(batch).is_err(),
        None => false,
    }
}
