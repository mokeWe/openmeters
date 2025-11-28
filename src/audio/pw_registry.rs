//! PipeWire registry observer service.
//!
//! This module provides a unified PipeWire connection that:
//! - Observes the graph (nodes, devices, ports, metadata)
//! - Creates and manages audio links between ports
//! - Sets routing metadata on application nodes

use crate::util::dict_to_map;
use crate::util::pipewire::{
    DEFAULT_AUDIO_SINK_KEY, DEFAULT_AUDIO_SOURCE_KEY, GraphPort, PortDirection,
    create_passive_audio_link, derive_node_direction, format_target_metadata, parse_metadata_name,
};
pub use crate::util::pipewire::{DefaultTarget, NodeDirection};
use anyhow::{Context, Result};
use parking_lot::RwLock;
use pipewire as pw;
use pw::metadata::{Metadata, MetadataListener};
use pw::registry::{GlobalObject, RegistryRc};
use pw::spa::utils::dict::DictRef;
use pw::types::ObjectType;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, OnceLock, mpsc};
use std::thread;
use std::time::Duration;
use tracing::{debug, error, info, warn};

const REGISTRY_THREAD_NAME: &str = "openmeters-pw-registry";

/// Metadata key instructing PipeWire where to route a node by object serial.
const TARGET_OBJECT_KEY: &str = "target.object";
/// Metadata key instructing PipeWire where to route a node by numeric node id.
const TARGET_NODE_KEY: &str = "target.node";

/// A single port-to-port link specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LinkSpec {
    pub output_node: u32,
    pub output_port: u32,
    pub input_node: u32,
    pub input_port: u32,
}

/// Command sent to the registry thread to perform actions.
#[derive(Debug, Clone)]
pub enum RegistryCommand {
    /// Set the desired audio links. Links not in this set will be removed.
    SetLinks(Vec<LinkSpec>),
    /// Route a node to a target using metadata properties.
    RouteNode {
        subject: u32,
        target_object: String,
        target_node: String,
    },
}

static RUNTIME: OnceLock<RegistryRuntime> = OnceLock::new();
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

    /// Send a command to the registry thread for execution.
    ///
    /// Commands are processed asynchronously by the registry thread. This method
    /// returns immediately after enqueueing the command.
    pub fn send_command(&self, command: RegistryCommand) -> bool {
        self.runtime.send_command(command)
    }

    /// Convenience method to set the desired audio links.
    pub fn set_links(&self, links: Vec<LinkSpec>) -> bool {
        self.send_command(RegistryCommand::SetLinks(links))
    }

    /// Convenience method to route a node to a specific target.
    pub fn route_node(&self, application: &NodeInfo, sink: &NodeInfo) -> bool {
        let metadata =
            format_target_metadata(sink.object_serial(), sink.id, sink.display_name().as_str());
        self.send_command(RegistryCommand::RouteNode {
            subject: application.id,
            target_object: metadata.target_object,
            target_node: metadata.target_node,
        })
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
    pub device_count: usize,
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
        let raw = target.and_then(|t| t.name.as_deref()).unwrap_or("(none)");

        let display = target
            .and_then(|t| self.resolve_default_target(t))
            .map(|node| node.display_name())
            .unwrap_or_else(|| raw.to_string());

        TargetDescription {
            display,
            raw: raw.to_string(),
        }
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
    pub ports: Vec<GraphPort>,
}

impl NodeInfo {
    /// Construct a `NodeInfo` from a PipeWire registry global announcement.
    pub fn from_global(global: &GlobalObject<&DictRef>) -> Self {
        let props = dict_to_map(global.props.as_ref().copied());

        let name = props.get(*pw::keys::NODE_NAME).cloned();
        let description = props
            .get(*pw::keys::NODE_DESCRIPTION)
            .cloned()
            .or_else(|| props.get("media.name").cloned())
            .or_else(|| name.clone());

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
            ports: Vec::new(),
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

    /// Get output ports suitable for loopback (prefer monitors, fall back to regular outputs).
    pub fn output_ports_for_loopback(&self) -> Vec<GraphPort> {
        self.ports_for_loopback(
            |p| p.direction == PortDirection::Output && p.is_monitor,
            |p| p.direction == PortDirection::Output,
        )
    }

    /// Get input ports suitable for loopback (prefer non-monitors, fall back to any inputs).
    pub fn input_ports_for_loopback(&self) -> Vec<GraphPort> {
        self.ports_for_loopback(
            |p| p.direction == PortDirection::Input && !p.is_monitor,
            |p| p.direction == PortDirection::Input,
        )
    }

    fn ports_for_loopback<F, G>(&self, primary: F, secondary: G) -> Vec<GraphPort>
    where
        F: Fn(&GraphPort) -> bool,
        G: Fn(&GraphPort) -> bool,
    {
        let primary_ports: Vec<_> = self.ports.iter().filter(|p| primary(p)).cloned().collect();
        if !primary_ports.is_empty() {
            return primary_ports;
        }

        let secondary_ports: Vec<_> = self
            .ports
            .iter()
            .filter(|p| secondary(p))
            .cloned()
            .collect();
        if !secondary_ports.is_empty() {
            return secondary_ports;
        }

        self.ports.clone()
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
    device_count: usize,
    port_index: HashMap<u32, (u32, u32)>, // global_id -> (node_id, port_id)
    metadata_defaults: MetadataDefaults,
}

impl RegistryState {
    /// Capture the current registry view into a snapshot suitable for sharing.
    fn snapshot(&self) -> RegistrySnapshot {
        let mut nodes: Vec<_> = self.nodes.values().cloned().collect();
        nodes.sort_by_key(|node| node.id);

        RegistrySnapshot {
            serial: self.serial,
            nodes,
            device_count: self.device_count,
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

    /// Increment device count for a newly announced device.
    fn add_device(&mut self) {
        self.device_count += 1;
        self.bump_serial();
    }

    /// Insert or update a port, attaching it to its parent node.
    fn upsert_port(&mut self, port: GraphPort) -> bool {
        let node_id = port.node_id;
        let port_id = port.port_id;
        let global_id = port.global_id;

        let Some(node) = self.nodes.get_mut(&node_id) else {
            return false;
        };

        let existing_idx = node.ports.iter().position(|p| p.port_id == port_id);

        let changed = match existing_idx {
            Some(idx) => {
                if node.ports[idx] != port {
                    node.ports[idx] = port;
                    true
                } else {
                    false
                }
            }
            None => {
                node.ports.push(port);
                true
            }
        };

        if changed {
            self.port_index.insert(global_id, (node_id, port_id));
            self.bump_serial();
        }

        changed
    }

    /// Remove a port by its global ID.
    fn remove_port(&mut self, global_id: u32) -> bool {
        let Some((node_id, port_id)) = self.port_index.remove(&global_id) else {
            return false;
        };

        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.ports.retain(|p| p.port_id != port_id);
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
    /// Ensure metadata targets point at live nodes when possible.
    fn reconcile_with_nodes(&mut self, nodes: &HashMap<u32, NodeInfo>) {
        for target in [&mut self.audio_sink, &mut self.audio_source]
            .into_iter()
            .flatten()
        {
            if target.node_id.is_some_and(|id| !nodes.contains_key(&id)) {
                target.node_id = None;
            }

            if target.node_id.is_none()
                && let Some(name) = &target.name
                && let Some((&id, _)) = nodes.iter().find(|(_, n)| n.name.as_deref() == Some(name))
            {
                target.node_id = Some(id);
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
    commands: Arc<RwLock<Option<mpsc::Sender<RegistryCommand>>>>,
}

impl RegistryRuntime {
    fn new() -> Self {
        Self::default()
    }

    fn set_command_sender(&self, sender: mpsc::Sender<RegistryCommand>) {
        *self.commands.write() = Some(sender);
    }

    fn send_command(&self, command: RegistryCommand) -> bool {
        if let Some(sender) = self.commands.read().as_ref() {
            if sender.send(command).is_err() {
                warn!("[registry] failed to send command; channel closed");
                false
            } else {
                true
            }
        } else {
            warn!("[registry] command channel not initialised");
            false
        }
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

    // Set up command channel for link/routing operations
    let (command_tx, command_rx) = mpsc::channel::<RegistryCommand>();
    runtime.set_command_sender(command_tx);

    // State for managing links and routing metadata
    let mut link_state = LinkState::new(core.clone());
    let routing_metadata: Rc<RefCell<Option<Metadata>>> = Rc::new(RefCell::new(None));

    let metadata_bindings: Rc<RefCell<HashMap<u32, MetadataBinding>>> =
        Rc::new(RefCell::new(HashMap::new()));

    let registry_for_added = registry.clone();
    let metadata_for_added = Rc::clone(&metadata_bindings);
    let metadata_for_removed = Rc::clone(&metadata_bindings);
    let runtime_for_added = runtime.clone();
    let runtime_for_removed = runtime.clone();
    let routing_metadata_for_added = Rc::clone(&routing_metadata);

    let _registry_listener = registry
        .add_listener_local()
        .global(move |global| {
            handle_global_added(
                &registry_for_added,
                global,
                &runtime_for_added,
                &metadata_for_added,
                &routing_metadata_for_added,
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

    // Main loop with command processing
    let loop_ref = mainloop.loop_();
    let mut commands_disconnected = false;

    loop {
        // Process pending commands
        if !commands_disconnected {
            loop {
                match command_rx.try_recv() {
                    Ok(command) => {
                        handle_command(command, &mut link_state, &routing_metadata, &mainloop);
                    }
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        commands_disconnected = true;
                        break;
                    }
                }
            }
        }

        // Run the PipeWire main loop for a short iteration
        if loop_ref.iterate(Duration::from_millis(50)) < 0 {
            break;
        }
    }

    info!("[registry] PipeWire registry loop exited");

    // Drop resources tied to the loop before returning.
    drop(registry);
    drop(context);

    Ok(())
}

/// State for managing audio links within the registry thread.
struct LinkState {
    core: pw::core::CoreRc,
    active_links: FxHashMap<LinkSpec, pw::link::Link>,
}

impl LinkState {
    fn new(core: pw::core::CoreRc) -> Self {
        Self {
            core,
            active_links: FxHashMap::default(),
        }
    }

    fn apply_links(&mut self, desired: Vec<LinkSpec>) {
        let desired_set: FxHashSet<LinkSpec> = desired.iter().copied().collect();

        // Remove links that are no longer desired
        self.active_links.retain(|spec, _| {
            let keep = desired_set.contains(spec);
            if !keep {
                debug!(
                    "[registry] removed link {}:{} -> {}:{}",
                    spec.output_node, spec.output_port, spec.input_node, spec.input_port
                );
            }
            keep
        });

        // Create new links
        for spec in desired {
            if self.active_links.contains_key(&spec) {
                continue;
            }

            match create_passive_audio_link(
                &self.core,
                spec.output_node,
                spec.output_port,
                spec.input_node,
                spec.input_port,
            ) {
                Ok(link) => {
                    debug!(
                        "[registry] linked {}:{} -> {}:{}",
                        spec.output_node, spec.output_port, spec.input_node, spec.input_port
                    );
                    self.active_links.insert(spec, link);
                }
                Err(err) => {
                    error!(
                        "[registry] link failed {}:{} -> {}:{}: {err}",
                        spec.output_node, spec.output_port, spec.input_node, spec.input_port
                    );
                }
            }
        }
    }
}

fn handle_command(
    command: RegistryCommand,
    link_state: &mut LinkState,
    routing_metadata: &Rc<RefCell<Option<Metadata>>>,
    mainloop: &pw::main_loop::MainLoopRc,
) {
    match command {
        RegistryCommand::SetLinks(desired) => {
            link_state.apply_links(desired);
        }
        RegistryCommand::RouteNode {
            subject,
            target_object,
            target_node,
        } => {
            if let Some(metadata) = routing_metadata.borrow().as_ref() {
                let loop_ref = mainloop.loop_();

                metadata.set_property(
                    subject,
                    TARGET_OBJECT_KEY,
                    Some("Spa:Id"),
                    Some(&target_object),
                );
                loop_ref.iterate(Duration::from_millis(10));

                metadata.set_property(subject, TARGET_NODE_KEY, Some("Spa:Id"), Some(&target_node));
                loop_ref.iterate(Duration::from_millis(10));

                debug!(
                    "[registry] routed node {} -> object={}, node={}",
                    subject, target_object, target_node
                );
            } else {
                warn!(
                    "[registry] cannot route node {}; no metadata bound",
                    subject
                );
            }
        }
    }
}

fn handle_global_added(
    registry: &RegistryRc,
    global: &GlobalObject<&DictRef>,
    runtime: &RegistryRuntime,
    metadata_bindings: &Rc<RefCell<HashMap<u32, MetadataBinding>>>,
    routing_metadata: &Rc<RefCell<Option<Metadata>>>,
) {
    // Each global represents a PipeWire object entering the graph. We bind and
    // mirror only the objects we care about (nodes, devices, ports, metadata).
    match global.type_ {
        ObjectType::Node => {
            let info = NodeInfo::from_global(global);
            runtime.mutate(move |state| state.upsert_node(info));
        }
        ObjectType::Device => {
            runtime.mutate(|state| {
                state.add_device();
                true
            });
        }
        ObjectType::Port => {
            if let Some(port) = GraphPort::from_global(global) {
                runtime.mutate(move |state| state.upsert_port(port));
            }
        }
        ObjectType::Metadata => {
            process_metadata_added(
                registry,
                global,
                runtime,
                metadata_bindings,
                routing_metadata,
            );
        }
        _ => {}
    }
}

fn handle_global_removed(
    id: u32,
    runtime: &RegistryRuntime,
    metadata_bindings: &Rc<RefCell<HashMap<u32, MetadataBinding>>>,
) {
    // Removal events arrive for any global object. Try ports, then nodes,
    // devices, and finally metadata proxies that we previously bound.
    if runtime.mutate(|state| state.remove_port(id)) {
        return;
    }

    if runtime.mutate(|state| state.remove_node(id)) {
        return;
    }

    if metadata_bindings.borrow_mut().remove(&id).is_some() {
        runtime.mutate(|state| state.clear_metadata_defaults(id));
    }
}

/// Metadata objects are probed in this preference order when multiple exist.
const PREFERRED_METADATA_NAMES: &[&str] = &["settings", "default"];

fn process_metadata_added(
    registry: &RegistryRc,
    global: &GlobalObject<&DictRef>,
    runtime: &RegistryRuntime,
    metadata_bindings: &Rc<RefCell<HashMap<u32, MetadataBinding>>>,
    routing_metadata: &Rc<RefCell<Option<Metadata>>>,
) {
    let metadata_id = global.id;
    if metadata_bindings.borrow().contains_key(&metadata_id) {
        return;
    }

    let props = dict_to_map(global.props.as_ref().copied());
    let metadata_name = props.get("metadata.name").cloned();

    let metadata = match registry.bind::<Metadata, _>(global) {
        Ok(metadata) => metadata,
        Err(err) => {
            warn!("[registry] failed to bind metadata {metadata_id}: {err}");
            return;
        }
    };

    // Check if this is a preferred metadata for routing
    let is_preferred = metadata_name
        .as_deref()
        .map(|candidate| {
            PREFERRED_METADATA_NAMES
                .iter()
                .any(|preferred| preferred.eq_ignore_ascii_case(candidate))
        })
        .unwrap_or(false);

    // Store as routing metadata if preferred, or if we don't have one yet
    {
        let mut routing_ref = routing_metadata.borrow_mut();
        if (is_preferred || routing_ref.is_none())
            && let Ok(routing_copy) = registry.bind::<Metadata, _>(global)
        {
            *routing_ref = Some(routing_copy);
            info!(
                "[registry] using metadata '{}' for routing",
                metadata_name.as_deref().unwrap_or("unnamed")
            );
        }
    }

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
