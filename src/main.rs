mod audio;
mod dsp;
mod ui;
mod util;
use audio::{pw_loopback, pw_registry, pw_virtual_sink, registry_monitor};
use std::sync::{Arc, mpsc};
use ui::{RoutingCommand, UiConfig};
use util::telemetry;

use tracing::{error, info};

fn main() {
    telemetry::init();
    info!("OpenMeters starting up");

    let (routing_tx, routing_rx) = mpsc::channel::<RoutingCommand>();
    let (snapshot_tx, snapshot_rx) = async_channel::bounded::<pw_registry::RegistrySnapshot>(64);

    let _registry_handle = registry_monitor::init_registry_monitor(routing_rx, snapshot_tx.clone());

    pw_virtual_sink::run();
    pw_loopback::run();

    let audio_stream = audio::meter_tap::audio_sample_stream();

    let ui_config =
        UiConfig::new(routing_tx, Some(Arc::new(snapshot_rx))).with_audio_stream(audio_stream);

    drop(snapshot_tx);

    if let Err(err) = ui::run(ui_config) {
        error!("[ui] failed: {err}");
    }
}
