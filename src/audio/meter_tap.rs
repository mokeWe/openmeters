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
const DROP_CHECK_INTERVAL: Duration = Duration::from_secs(5);

static AUDIO_STREAM: OnceLock<Arc<AsyncReceiver<Vec<f32>>>> = OnceLock::new();
static FORMAT_STATE: RwLock<MeterFormat> = RwLock::new(MeterFormat::new());

#[derive(Debug, Clone, Copy)]
pub struct MeterFormat {
    pub channels: usize,
    pub sample_rate: f32,
}

impl MeterFormat {
    const fn new() -> Self {
        Self {
            channels: 2,
            sample_rate: DEFAULT_SAMPLE_RATE,
        }
    }

    fn differs_from(&self, channels: usize, sample_rate: f32) -> bool {
        self.channels != channels || (self.sample_rate - sample_rate).abs() > f32::EPSILON
    }
}

pub fn current_format() -> MeterFormat {
    *FORMAT_STATE.read()
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
    let mut last_drop_check = Instant::now();
    let mut drop_baseline = buffer.dropped_frames();
    let mut has_batch_data = false;

    loop {
        if last_drop_check.elapsed() >= DROP_CHECK_INTERVAL {
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
                    &mut last_flush,
                    &mut has_batch_data,
                ) {
                    break;
                }
            }
            Ok(None) if sender.is_closed() => break,
            Ok(None) => {
                if last_flush.elapsed() >= MAX_BATCH_LATENCY && try_flush(&sender, &mut batcher) {
                    break;
                }
                has_batch_data = false;
                last_flush = Instant::now();
            }
            Err(_) => {
                error!("[meter-tap] capture buffer unavailable; stopping tap");
                break;
            }
        }
    }

    let _ = batcher.take().map(|b| sender.send_blocking(b));
    info!(
        "[meter-tap] audio channel closed; {} dropped capture frames",
        buffer.dropped_frames()
    );
}

fn handle_packet(
    packet: CapturedAudio,
    sender: &AsyncSender<Vec<f32>>,
    batcher: &mut SampleBatcher,
    last_flush: &mut Instant,
    has_batch_data: &mut bool,
) -> bool {
    let channels = packet.channels.max(1) as usize;
    let sample_rate = packet.sample_rate.max(1) as f32;

    // Flush pending batch if format changed mid-batch
    if *has_batch_data && FORMAT_STATE.read().differs_from(channels, sample_rate) {
        if try_flush(sender, batcher) {
            return true;
        }
        *has_batch_data = false;
        *last_flush = Instant::now();
    }

    *FORMAT_STATE.write() = MeterFormat {
        channels,
        sample_rate,
    };

    if !packet.samples.is_empty() {
        batcher.push(packet.samples);
        *has_batch_data = true;
    }

    if batcher.should_flush() || last_flush.elapsed() >= MAX_BATCH_LATENCY {
        if try_flush(sender, batcher) {
            return true;
        }
        *has_batch_data = false;
        *last_flush = Instant::now();
    }

    false
}

fn try_flush(sender: &AsyncSender<Vec<f32>>, batcher: &mut SampleBatcher) -> bool {
    batcher
        .take()
        .is_some_and(|batch| sender.send_blocking(batch).is_err())
}
