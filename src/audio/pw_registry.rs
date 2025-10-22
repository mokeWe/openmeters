//! PipeWire registry observer service for OpenMeters.

use crate::util::dict_to_map;
use crate::util::pipewire::{
    DEFAULT_AUDIO_SINK_KEY, DEFAULT_AUDIO_SOURCE_KEY, GraphNode, derive_node_direction,
    parse_metadata_name,
};
pub use crate::util::pipewire::{DefaultTarget, NodeDirection};
use anyhow::{Context, Result};
use parking_lot::RwLock;
use pipewire as pw;
use pw::metadata::{Metadata, MetadataListener};
use pw::registry::{GlobalObject, RegistryRc};
use pw::spa::utils::dict::DictRef;
use pw::types::ObjectType;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, OnceLock, mpsc};
use std::thread;
use std::time::Duration;
use tracing::{error, info, warn};

const REGISTRY_THREAD_NAME: &str = "openmeters-pw-registry";

static RUNTIME: OnceLock<RegistryRuntime> = OnceLock::new();
/// Ensure the PipeWire registry observer is running and return a handle to it.
///
/// The first caller spins up the background registry thread that keeps an
/// in-memory mirror of PipeWire nodes, devices, and metadata. Subsequent calls
/// simply clone the runtime state so new consumers can subscribe to snapshots.
pub fn spawn_registry() -> Result<AudioRegistryHandle> {
    if let Some(runtime) = RUNTIME.get() {
        return Ok(AudioRegistryHandle {
            runtime: runtime.clone(),
        });
    }

    let runtime = RegistryRuntime::new();

    match RUNTIME.set(runtime.clone()) {
        Ok(()) => {
            let thread_runtime = runtime.clone();
            thread::Builder::new()
                .name(REGISTRY_THREAD_NAME.into())
                .spawn(move || {
                    if let Err(err) = registry_thread_main(thread_runtime) {
                        error!("[registry] thread terminated: {err:?}");
                    }
                })
                .context("failed to spawn PipeWire registry thread")?;

            Ok(AudioRegistryHandle { runtime })
        }
        Err(_) => {
            let runtime = RUNTIME.get().expect("registry runtime initialized");
            Ok(AudioRegistryHandle {
                runtime: runtime.clone(),
            })
        }
    }
}

/// Shared handle that exposes snapshots and subscriptions to the PipeWire registry.
///
/// Cloning the handle is cheap and safe across threads. Each clone can obtain a
/// point-in-time snapshot or subscribe to live updates without coordinating with
/// other subscribers.
#[derive(Clone)]
pub struct AudioRegistryHandle {
    runtime: RegistryRuntime,
}

impl AudioRegistryHandle {
    /// Clone a point-in-time view of all known nodes, devices, and defaults.
    pub fn snapshot(&self) -> RegistrySnapshot {
        self.runtime.snapshot()
    }

    /// Subscribe to ongoing registry snapshots; the iterator yields the initial state first.
    pub fn subscribe(&self) -> RegistryUpdates {
        let mut updates = self.runtime.subscribe();
        if updates.initial.is_none() {
            updates.initial = Some(self.snapshot());
        }
        updates
    }
}

/// Iterator that produces live snapshots of the PipeWire registry.
///
/// The first item yielded is always the current snapshot at the time the
/// subscription was created. Subsequent items represent incremental changes
/// observed by the registry worker thread.
pub struct RegistryUpdates {
    initial: Option<RegistrySnapshot>,
    receiver: mpsc::Receiver<RegistrySnapshot>,
}

impl RegistryUpdates {
    /// Block until the next snapshot is available; the first call returns the initial snapshot.
    pub fn recv(&mut self) -> Option<RegistrySnapshot> {
        if let Some(snapshot) = self.initial.take() {
            return Some(snapshot);
        }
        self.receiver.recv().ok()
    }

    /// Attempt to receive a snapshot within the provided timeout.
    ///
    /// Returns `Ok(Some(snapshot))` when an update arrives, `Ok(None)` on timeout,
    /// and propagates `RecvTimeoutError::Disconnected` when the channel is closed.
    pub fn recv_timeout(
        &mut self,
        timeout: Duration,
    ) -> Result<Option<RegistrySnapshot>, mpsc::RecvTimeoutError> {
        if let Some(snapshot) = self.initial.take() {
            return Ok(Some(snapshot));
        }

        match self.receiver.recv_timeout(timeout) {
            Ok(snapshot) => Ok(Some(snapshot)),
            Err(mpsc::RecvTimeoutError::Timeout) => Ok(None),
            Err(err) => Err(err),
        }
    }
}

impl Iterator for RegistryUpdates {
    type Item = RegistrySnapshot;

    fn next(&mut self) -> Option<Self::Item> {
        self.recv()
    }
}

/// Collection of registry state cloned for thread-safe consumption.
///
/// `RegistrySnapshot` is intentionally copy-on-write friendly: the background
/// thread owns the canonical state, while readers receive cheap clones that can
/// be sent across threads without additional locking.
#[derive(Clone, Debug, Default)]
pub struct RegistrySnapshot {
    pub serial: u64,
    pub nodes: Vec<NodeInfo>,
    pub devices: Vec<DeviceInfo>,
    pub defaults: MetadataDefaults,
}

/// Human-friendly summary of a PipeWire target, including the resolved node label.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetDescription {
    pub display: String,
    pub raw: String,
}

impl RegistrySnapshot {
    /// Produce a human-readable description of a default target, including the raw metadata value.
    pub fn describe_default_target(&self, target: Option<&DefaultTarget>) -> TargetDescription {
        let raw = target
            .and_then(|t| t.name.as_deref())
            .unwrap_or("(none)")
            .to_string();

        let display = target
            .and_then(|t| self.resolve_default_target(t))
            .map(|node| node.display_name())
            .unwrap_or_else(|| raw.clone());

        TargetDescription { display, raw }
    }

    /// Attempt to resolve a metadata default to a known node in the snapshot.
    ///
    /// Resolution prefers an explicit node ID advertised with the metadata, and
    /// falls back to case-insensitive name matching to cover daemons that only
    /// publish target names.
    pub fn resolve_default_target<'a>(&'a self, target: &DefaultTarget) -> Option<&'a NodeInfo> {
        target
            .node_id
            .and_then(|id| self.nodes.iter().find(|node| node.id == id))
            .or_else(|| {
                target
                    .name
                    .as_deref()
                    .and_then(|name| self.nodes.iter().find(|node| node.matches_label(name)))
            })
    }

    /// Locate a node by comparing its name and description (case-insensitive) to `label`.
    pub fn find_node_by_label<'a>(&'a self, label: &str) -> Option<&'a NodeInfo> {
        self.nodes.iter().find(|node| node.matches_label(label))
    }

    /// Iterate over application nodes that should be routed to the supplied sink.
    ///
    /// Candidates exclude the sink itself and focus on application-owned output
    /// nodes with audio media classes, mirroring the logic the router needs when
    /// issuing metadata overrides.
    pub fn route_candidates<'a>(
        &'a self,
        sink: &'a NodeInfo,
    ) -> impl Iterator<Item = &'a NodeInfo> + 'a {
        self.nodes
            .iter()
            .filter(move |node| node.should_route_to(sink))
    }
}

/// PipeWire node information extracted from registry announcements.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NodeInfo {
    pub id: u32,
    pub name: Option<String>,
    pub description: Option<String>,
    pub media_class: Option<String>,
    pub media_role: Option<String>,
    pub direction: NodeDirection,
    pub is_virtual: bool,
    pub parent_device: Option<u32>,
    pub properties: HashMap<String, String>,
}

impl NodeInfo {
    /// Construct a `NodeInfo` from a PipeWire registry global announcement.
    pub fn from_global(global: &GlobalObject<&DictRef>) -> Self {
        let props = dict_to_map(global.props.as_ref().copied());

        let summary = GraphNode::from_props(global.id, &props);
        let name = summary.name().map(|value| value.to_string());
        let description = summary.description().map(|value| value.to_string());

        let media_class = props.get(*pw::keys::MEDIA_CLASS).cloned();
        let media_role = props.get(*pw::keys::MEDIA_ROLE).cloned();
        let direction = derive_node_direction(media_class.as_deref(), &props);
        let parent_device = props.get("device.id").and_then(|id| id.parse::<u32>().ok());
        let is_virtual = props
            .get("node.virtual")
            .map(|value| value == "true")
            .unwrap_or_else(|| name.as_deref() == Some("openmeters.sink"));

        NodeInfo {
            id: global.id,
            name,
            description,
            media_class,
            media_role,
            direction,
            is_virtual,
            parent_device,
            properties: props,
        }
    }

    /// Best-effort display name favouring explicit node names over descriptions.
    pub fn display_name(&self) -> String {
        self.name
            .clone()
            .or(self.description.clone())
            .unwrap_or_else(|| format!("node#{}", self.id))
    }

    /// Return the application name associated with this node, if present.
    pub fn app_name(&self) -> Option<&str> {
        self.properties.get(*pw::keys::APP_NAME).map(String::as_str)
    }

    /// Retrieve the PipeWire object serial for this node, when advertised.
    pub fn object_serial(&self) -> Option<&str> {
        self.properties.get("object.serial").map(String::as_str)
    }

    /// Evaluate whether the node label or description matches a given string ignoring ASCII case.
    pub fn matches_label(&self, label: &str) -> bool {
        self.name
            .as_deref()
            .is_some_and(|value| value.eq_ignore_ascii_case(label))
            || self
                .description
                .as_deref()
                .is_some_and(|value| value.eq_ignore_ascii_case(label))
    }

    /// Determine whether this node should be routed to the provided sink.
    ///
    /// The policy mirrors WirePlumber defaults: we only target audio output
    /// nodes owned by user applications, keeping system devices and input nodes
    /// untouched.
    pub fn should_route_to(&self, sink: &Self) -> bool {
        self.id != sink.id && self.is_audio_application_output()
    }

    fn is_audio_application_output(&self) -> bool {
        self.direction == NodeDirection::Output
            && self
                .media_class
                .as_deref()
                .map(|class| class.to_ascii_lowercase().contains("audio"))
                .unwrap_or(false)
            && self.app_name().is_some()
    }
}

/// PipeWire device information extracted from registry announcements.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeviceInfo {
    pub id: u32,
    pub name: Option<String>,
    pub description: Option<String>,
    pub properties: HashMap<String, String>,
}

impl DeviceInfo {
    /// Construct a `DeviceInfo` from a PipeWire registry global announcement.
    pub fn from_global(global: &GlobalObject<&DictRef>) -> Self {
        let props = dict_to_map(global.props.as_ref().copied());
        let name = props.get("device.name").cloned();
        let description = props
            .get(*pw::keys::DEVICE_DESCRIPTION)
            .cloned()
            .or_else(|| props.get("device.product.name").cloned())
            .or_else(|| name.clone());

        DeviceInfo {
            id: global.id,
            name,
            description,
            properties: props,
        }
    }
}

/// Default targets as reported by PipeWire metadata.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MetadataDefaults {
    pub audio_sink: Option<DefaultTarget>,
    pub audio_source: Option<DefaultTarget>,
}

#[derive(Debug, Default)]
struct RegistryState {
    serial: u64,
    nodes: HashMap<u32, NodeInfo>,
    devices: HashMap<u32, DeviceInfo>,
    metadata_defaults: MetadataDefaults,
}

impl RegistryState {
    /// Capture the current registry view into a snapshot suitable for sharing.
    fn snapshot(&self) -> RegistrySnapshot {
        let mut nodes: Vec<_> = self.nodes.values().cloned().collect();
        nodes.sort_by_key(|node| node.id);
        let mut devices: Vec<_> = self.devices.values().cloned().collect();
        devices.sort_by_key(|device| device.id);

        RegistrySnapshot {
            serial: self.serial,
            nodes,
            devices,
            defaults: self.metadata_defaults.clone(),
        }
    }

    /// Insert or update a node, returning whether the state changed.
    fn upsert_node(&mut self, info: NodeInfo) -> bool {
        let needs_update = !matches!(
            self.nodes.get(&info.id),
            Some(existing) if existing == &info
        );

        if needs_update {
            self.nodes.insert(info.id, info);
            self.reconcile_defaults();
            self.bump_serial();
        }

        needs_update
    }

    /// Remove a node by ID, returning `true` when the registry view changed.
    fn remove_node(&mut self, id: u32) -> bool {
        if let Some(info) = self.nodes.remove(&id) {
            let fallback = info.name.or(info.description);
            let defaults_changed = self.metadata_defaults.clear_node(id, fallback);
            if defaults_changed {
                self.reconcile_defaults();
            }
            self.bump_serial();
            true
        } else {
            false
        }
    }

    /// Insert or update a device description.
    fn upsert_device(&mut self, info: DeviceInfo) -> bool {
        let needs_update = !matches!(
            self.devices.get(&info.id),
            Some(existing) if existing == &info
        );

        if needs_update {
            self.devices.insert(info.id, info);
            self.bump_serial();
        }

        needs_update
    }

    /// Remove a device by ID.
    fn remove_device(&mut self, id: u32) -> bool {
        if self.devices.remove(&id).is_some() {
            self.bump_serial();
            true
        } else {
            false
        }
    }

    /// Apply a metadata change to the cached defaults map.
    fn apply_metadata_property(
        &mut self,
        metadata_id: u32,
        subject: u32,
        key: Option<&str>,
        type_hint: Option<&str>,
        value: Option<&str>,
    ) -> bool {
        let changed = match key {
            Some(key) => {
                self.metadata_defaults
                    .apply_update(metadata_id, subject, key, type_hint, value)
            }
            None => self.metadata_defaults.clear_metadata(metadata_id),
        };

        if changed {
            self.reconcile_defaults();
            self.bump_serial();
        }

        changed
    }

    /// Drop all cached properties coming from a metadata object that vanished.
    fn clear_metadata_defaults(&mut self, metadata_id: u32) -> bool {
        if self.metadata_defaults.clear_metadata(metadata_id) {
            self.reconcile_defaults();
            self.bump_serial();
            true
        } else {
            false
        }
    }

    fn bump_serial(&mut self) {
        self.serial = self.serial.wrapping_add(1);
    }

    fn reconcile_defaults(&mut self) {
        self.metadata_defaults.reconcile_with_nodes(&self.nodes);
    }
}

impl MetadataDefaults {
    /// Update defaults based on a metadata property, tracking whether state changed.
    fn apply_update(
        &mut self,
        metadata_id: u32,
        subject: u32,
        key: &str,
        type_hint: Option<&str>,
        value: Option<&str>,
    ) -> bool {
        let slot = match key {
            DEFAULT_AUDIO_SINK_KEY => &mut self.audio_sink,
            DEFAULT_AUDIO_SOURCE_KEY => &mut self.audio_source,
            _ => return false,
        };

        match value {
            Some(val) => {
                let inserted = slot.is_none();
                let parsed_name = parse_metadata_name(type_hint, Some(val));
                let name_ref = parsed_name.as_deref().or(Some(val));

                let target = slot.get_or_insert_with(DefaultTarget::default);
                let updated = target.update(metadata_id, subject, type_hint, name_ref);
                inserted || updated
            }
            None => {
                if slot
                    .as_ref()
                    .is_some_and(|target| target.metadata_id == Some(metadata_id))
                {
                    *slot = None;
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Ensure metadata targets point at live nodes when possible.
    fn reconcile_with_nodes(&mut self, nodes: &HashMap<u32, NodeInfo>) {
        for target in [&mut self.audio_sink, &mut self.audio_source]
            .into_iter()
            .flatten()
        {
            if let Some(node_id) = target.node_id
                && !nodes.contains_key(&node_id)
            {
                target.node_id = None;
            }
            if target.node_id.is_none()
                && let Some(name) = &target.name
                && let Some((id, _)) = nodes
                    .iter()
                    .find(|(_, node)| node.name.as_deref() == Some(name))
            {
                target.node_id = Some(*id);
            }
        }
    }

    fn clear_metadata(&mut self, metadata_id: u32) -> bool {
        let mut changed = false;
        for slot in [&mut self.audio_sink, &mut self.audio_source] {
            if slot
                .as_ref()
                .is_some_and(|target| target.metadata_id == Some(metadata_id))
            {
                *slot = None;
                changed = true;
            }
        }
        changed
    }

    /// Clear any defaults resolved to the removed node and optionally backfill names.
    fn clear_node(&mut self, node_id: u32, fallback_name: Option<String>) -> bool {
        let mut changed = false;
        for slot in [&mut self.audio_sink, &mut self.audio_source] {
            if let Some(target) = slot
                && target.node_id == Some(node_id)
            {
                target.node_id = None;
                if target.name.is_none() {
                    target.name = fallback_name.clone();
                }
                changed = true;
            }
        }
        changed
    }
}
#[derive(Clone, Default)]
struct RegistryRuntime {
    state: Arc<RwLock<RegistryState>>,
    watchers: Arc<RwLock<Vec<mpsc::Sender<RegistrySnapshot>>>>,
}

impl RegistryRuntime {
    fn new() -> Self {
        Self::default()
    }

    fn snapshot(&self) -> RegistrySnapshot {
        self.state.read().snapshot()
    }

    fn subscribe(&self) -> RegistryUpdates {
        let (tx, rx) = mpsc::channel();
        {
            let mut watchers = self.watchers.write();
            watchers.push(tx);
        }

        RegistryUpdates {
            initial: Some(self.snapshot()),
            receiver: rx,
        }
    }

    fn mutate<F>(&self, mutate: F) -> bool
    where
        F: FnOnce(&mut RegistryState) -> bool,
    {
        let changed = {
            let mut state = self.state.write();
            mutate(&mut state)
        };

        if changed {
            self.notify_watchers();
        }

        changed
    }

    fn notify_watchers(&self) {
        let snapshot = {
            let state = self.state.read();
            state.snapshot()
        };

        let mut watchers = self.watchers.write();
        watchers.retain(|sender| sender.send(snapshot.clone()).is_ok());
    }
}

fn registry_thread_main(runtime: RegistryRuntime) -> Result<()> {
    pw::init();

    let mainloop =
        pw::main_loop::MainLoopRc::new(None).context("failed to create PipeWire main loop")?;
    let context = pw::context::ContextRc::new(&mainloop, None)
        .context("failed to create PipeWire context")?;
    let core = context
        .connect_rc(None)
        .context("failed to connect to PipeWire core")?;
    let registry = core
        .get_registry_rc()
        .context("failed to obtain PipeWire registry")?;

    let metadata_bindings: Rc<RefCell<HashMap<u32, MetadataBinding>>> =
        Rc::new(RefCell::new(HashMap::new()));

    let registry_for_added = registry.clone();
    let metadata_for_added = Rc::clone(&metadata_bindings);
    let metadata_for_removed = Rc::clone(&metadata_bindings);
    let runtime_for_added = runtime.clone();
    let runtime_for_removed = runtime.clone();

    let _registry_listener = registry
        .add_listener_local()
        .global(move |global| {
            handle_global_added(
                &registry_for_added,
                global,
                &runtime_for_added,
                &metadata_for_added,
            );
        })
        .global_remove(move |id| {
            handle_global_removed(id, &runtime_for_removed, &metadata_for_removed);
        })
        .register();

    if let Err(err) = core.sync(0) {
        error!("[registry] failed to sync core: {err}");
    }

    info!("[registry] PipeWire registry thread running");
    mainloop.run();
    info!("[registry] PipeWire registry loop exited");

    // Drop resources tied to the loop before returning.
    drop(registry);
    drop(context);

    Ok(())
}

fn handle_global_added(
    registry: &RegistryRc,
    global: &GlobalObject<&DictRef>,
    runtime: &RegistryRuntime,
    metadata_bindings: &Rc<RefCell<HashMap<u32, MetadataBinding>>>,
) {
    // Each global represents a PipeWire object entering the graph. We bind and
    // mirror only the objects we care about (nodes, devices, metadata).
    match global.type_ {
        ObjectType::Node => {
            let info = NodeInfo::from_global(global);
            runtime.mutate(move |state| state.upsert_node(info));
        }
        ObjectType::Device => {
            let info = DeviceInfo::from_global(global);
            runtime.mutate(move |state| state.upsert_device(info));
        }
        ObjectType::Metadata => {
            process_metadata_added(registry, global, runtime, metadata_bindings);
        }
        _ => {}
    }
}

fn handle_global_removed(
    id: u32,
    runtime: &RegistryRuntime,
    metadata_bindings: &Rc<RefCell<HashMap<u32, MetadataBinding>>>,
) {
    // Removal events arrive for any global object. Try nodes, then devices, and
    // finally metadata proxies that we previously bound.
    if runtime.mutate(|state| state.remove_node(id)) {
        return;
    }

    if runtime.mutate(|state| state.remove_device(id)) {
        return;
    }

    if metadata_bindings.borrow_mut().remove(&id).is_some() {
        runtime.mutate(|state| state.clear_metadata_defaults(id));
    }
}

fn process_metadata_added(
    registry: &RegistryRc,
    global: &GlobalObject<&DictRef>,
    runtime: &RegistryRuntime,
    metadata_bindings: &Rc<RefCell<HashMap<u32, MetadataBinding>>>,
) {
    let metadata_id = global.id;
    if metadata_bindings.borrow().contains_key(&metadata_id) {
        return;
    }

    let metadata = match registry.bind::<Metadata, _>(global) {
        Ok(metadata) => metadata,
        Err(err) => {
            warn!("[registry] failed to bind metadata {metadata_id}: {err}");
            return;
        }
    };

    let runtime_for_listener = runtime.clone();
    let listener = metadata
        .add_listener_local()
        .property(move |subject, key, type_, value| {
            handle_metadata_property(
                &runtime_for_listener,
                metadata_id,
                subject,
                key,
                type_,
                value,
            );
            0
        })
        .register();

    metadata_bindings.borrow_mut().insert(
        metadata_id,
        MetadataBinding {
            _proxy: metadata,
            _listener: listener,
        },
    );
}

fn handle_metadata_property(
    runtime: &RegistryRuntime,
    metadata_id: u32,
    subject: u32,
    key: Option<&str>,
    type_hint: Option<&str>,
    value: Option<&str>,
) {
    // Any change to default sink/source metadata must be reflected in the
    // cached defaults and broadcast to subscribers.
    runtime
        .mutate(|state| state.apply_metadata_property(metadata_id, subject, key, type_hint, value));
}

struct MetadataBinding {
    _proxy: Metadata,
    _listener: MetadataListener,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_defaults_reconcile_matches_by_name() {
        let mut defaults = MetadataDefaults {
            audio_sink: Some(DefaultTarget {
                metadata_id: Some(7),
                node_id: None,
                name: Some("node.main".into()),
                type_hint: None,
            }),
            audio_source: None,
        };

        let mut nodes = HashMap::new();
        nodes.insert(
            42,
            NodeInfo {
                id: 42,
                name: Some("node.main".into()),
                ..Default::default()
            },
        );

        defaults.reconcile_with_nodes(&nodes);
        assert_eq!(
            defaults.audio_sink.as_ref().and_then(|t| t.node_id),
            Some(42)
        );
    }
}
