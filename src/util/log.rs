use crate::audio::pw_registry;
use tracing::info;

pub fn registry_snapshot(kind: &str, snapshot: &pw_registry::RegistrySnapshot) {
    let sink_summary = snapshot.describe_default_target(snapshot.defaults.audio_sink.as_ref());
    let source_summary = snapshot.describe_default_target(snapshot.defaults.audio_source.as_ref());

    info!(
        "[registry] {kind}: serial={}, nodes={}, devices={}, default_sink={} (raw={}), default_source={} (raw={})",
        snapshot.serial,
        snapshot.nodes.len(),
        snapshot.devices.len(),
        sink_summary.display,
        sink_summary.raw,
        source_summary.display,
        source_summary.raw
    );
}
