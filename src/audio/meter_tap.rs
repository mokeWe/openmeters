use super::pw_virtual_sink::{self, CaptureBuffer};
use crate::util::audio::SampleBatcher;
use async_channel::{Receiver as AsyncReceiver, Sender as AsyncSender};
use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

/// Capacity of the channel forwarding audio frames to the UI.
const CHANNEL_CAPACITY: usize = 64;
/// Maximum time we block while waiting for new audio before checking for shutdown.
const POLL_BACKOFF: Duration = Duration::from_millis(50);
/// Desired number of PCM samples per chunk forwarded to the UI.
const TARGET_BATCH_SAMPLES: usize = 2_048;
/// Maximum amount of time we allow audio to accumulate before forcing a partial flush.
const MAX_BATCH_LATENCY: Duration = Duration::from_millis(25);

static AUDIO_STREAM: OnceLock<Arc<AsyncReceiver<Vec<f32>>>> = OnceLock::new();

/// Obtain a shared receiver that yields captured audio frames suitable for UI visualisations.
///
/// The first caller spawns a lightweight worker thread that drains the virtual sink capture
/// buffer and forwards frames through the returned async channel. Subsequent callers reuse the
/// existing stream.
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

    loop {
        match buffer.pop_wait_timeout(POLL_BACKOFF) {
            Ok(Some(samples)) => {
                if samples.is_empty() {
                    continue;
                }

                batcher.push(samples);
                let flush_due = batcher.should_flush() || last_flush.elapsed() >= MAX_BATCH_LATENCY;
                if flush_due {
                    if flush_batch(&sender, &mut batcher) {
                        break;
                    }
                    last_flush = Instant::now();
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
                    last_flush = Instant::now();
                }
            }
            Err(_) => {
                eprintln!("[meter-tap] capture buffer unavailable; stopping tap");
                break;
            }
        }
    }

    let _ = flush_batch(&sender, &mut batcher);

    println!("[meter-tap] audio channel closed; ending tap thread");
}

fn flush_batch(sender: &AsyncSender<Vec<f32>>, batcher: &mut SampleBatcher) -> bool {
    match batcher.take() {
        Some(batch) => sender.send_blocking(batch).is_err(),
        None => false,
    }
}
