use crate::audio::{VIRTUAL_SINK_NAME, pw_loopback, pw_registry, pw_router};
use crate::ui::RoutingCommand;
use crate::ui::app::config::{CaptureMode, DeviceSelection};
use crate::util::log;
use crate::util::pipewire::pair_ports_by_channel;
use async_channel::Sender;
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::mpsc;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

const ROUTER_RECOVERY_DELAY: Duration = Duration::from_millis(500);
const ROUTER_RETRY_DELAY: Duration = Duration::from_secs(5);
const POLL_INTERVAL: Duration = Duration::from_millis(100);
const ROUTE_RETRY_INTERVAL: Duration = Duration::from_millis(350);

pub fn init_registry_monitor(
    command_rx: mpsc::Receiver<RoutingCommand>,
    snapshot_tx: Sender<pw_registry::RegistrySnapshot>,
) -> Option<pw_registry::AudioRegistryHandle> {
    let handle = pw_registry::spawn_registry()
        .inspect_err(|err| error!("[registry] failed to start PipeWire registry: {err:?}"))
        .ok()?;

    let handle_for_thread = handle.clone();
    std::thread::Builder::new()
        .name("openmeters-registry-monitor".into())
        .spawn(move || run_monitor_loop(handle_for_thread, command_rx, snapshot_tx))
        .inspect_err(|err| error!("[registry] failed to spawn monitor thread: {err}"))
        .ok()?;

    Some(handle)
}

fn run_monitor_loop(
    handle: pw_registry::AudioRegistryHandle,
    command_rx: mpsc::Receiver<RoutingCommand>,
    snapshot_tx: Sender<pw_registry::RegistrySnapshot>,
) {
    let mut updates = handle.subscribe();
    let router = pw_router::Router::new()
        .inspect_err(|err| error!("[router] failed to initialise PipeWire router: {err:?}"))
        .ok();

    let mut routing = RoutingManager::new(router, command_rx);

    loop {
        if routing.apply_pending_commands() {
            continue;
        }

        match updates.recv_timeout(POLL_INTERVAL) {
            Ok(Some(snapshot)) => {
                log::registry_snapshot("update", &snapshot);
                routing.handle_snapshot(snapshot);

                if let Some(snapshot) = routing.snapshot.as_ref()
                    && snapshot_tx.send_blocking(snapshot.clone()).is_err()
                {
                    info!("[registry] UI channel closed; stopping");
                    break;
                }
            }
            Ok(None) | Err(mpsc::RecvTimeoutError::Timeout) => routing.handle_idle(),
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    info!("[registry] update stream ended");
}

struct RoutingManager {
    router: Option<pw_router::Router>,
    commands: mpsc::Receiver<RoutingCommand>,
    disabled_nodes: FxHashSet<u32>,
    routed_nodes: FxHashMap<u32, RouteState>,
    capture_mode: CaptureMode,
    device_target: DeviceSelection,
    snapshot: Option<pw_registry::RegistrySnapshot>,
    cached_target: Option<(u32, String)>,
    router_retry_after: Option<Instant>,
    router_epoch: u64,
    current_links: Vec<pw_loopback::LinkSpec>,
    sink_missing_warned: bool,
    device_missing_warned: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RouteTarget {
    Virtual(u32),
    Hardware(u32),
}

#[derive(Debug, Clone)]
struct RouteState {
    target: RouteTarget,
    last_attempt: Option<Instant>,
    failures: u32,
    epoch: u64,
}

impl RouteState {
    fn new(target: RouteTarget) -> Self {
        Self {
            target,
            last_attempt: None,
            failures: 0,
            epoch: 0,
        }
    }

    fn mark_success(&mut self, target: RouteTarget, now: Instant, epoch: u64) {
        self.target = target;
        self.last_attempt = Some(now);
        self.failures = 0;
        self.epoch = epoch;
    }

    fn mark_failure(&mut self, target: RouteTarget, now: Instant) {
        self.target = target;
        self.last_attempt = Some(now);
        self.failures = self.failures.saturating_add(1);
    }

    fn should_retry(&self, desired: RouteTarget, now: Instant, epoch: u64) -> bool {
        self.target != desired
            || self.epoch < epoch
            || (self.failures > 0
                && self
                    .last_attempt
                    .is_none_or(|last| now.duration_since(last) >= ROUTE_RETRY_INTERVAL))
    }
}

impl RoutingManager {
    fn new(router: Option<pw_router::Router>, commands: mpsc::Receiver<RoutingCommand>) -> Self {
        Self {
            router_epoch: router.is_some().into(),
            router,
            commands,
            disabled_nodes: FxHashSet::default(),
            routed_nodes: FxHashMap::default(),
            capture_mode: CaptureMode::Applications,
            device_target: DeviceSelection::Default,
            snapshot: None,
            cached_target: None,
            router_retry_after: None,
            current_links: Vec::new(),
            sink_missing_warned: false,
            device_missing_warned: false,
        }
    }

    fn handle_snapshot(&mut self, snapshot: pw_registry::RegistrySnapshot) {
        self.consume_commands();
        self.apply_snapshot(&snapshot, true);
        self.snapshot = Some(snapshot);
    }

    fn consume_commands(&mut self) -> bool {
        let mut changed = false;
        while let Ok(command) = self.commands.try_recv() {
            changed |= match command {
                RoutingCommand::SetApplicationEnabled { node_id, enabled } => {
                    if enabled {
                        self.disabled_nodes.remove(&node_id)
                    } else {
                        self.disabled_nodes.insert(node_id)
                    }
                }
                RoutingCommand::SetCaptureMode(mode) => {
                    if self.capture_mode != mode {
                        self.capture_mode = mode;
                        true
                    } else {
                        false
                    }
                }
                RoutingCommand::SelectCaptureDevice(selection) => {
                    if self.device_target != selection {
                        self.device_target = selection;
                        true
                    } else {
                        false
                    }
                }
            };
        }
        changed
    }

    fn apply_pending_commands(&mut self) -> bool {
        if !self.consume_commands() {
            return false;
        }

        if let Some(snapshot) = self.snapshot.take() {
            self.apply_snapshot(&snapshot, false);
            self.snapshot = Some(snapshot);
        }

        true
    }

    fn apply_snapshot(&mut self, snapshot: &pw_registry::RegistrySnapshot, log_sink_missing: bool) {
        self.cleanup_stale_nodes(snapshot);

        // Compute and apply desired links
        let desired_links = self.compute_desired_links(snapshot);
        self.ensure_links(desired_links);

        let Some(sink) = snapshot.find_node_by_label(VIRTUAL_SINK_NAME) else {
            if log_sink_missing && !self.sink_missing_warned {
                warn!("[router] virtual sink '{VIRTUAL_SINK_NAME}' not yet available");
                self.sink_missing_warned = true;
            }
            return;
        };

        self.sink_missing_warned = false;
        let hardware_sink = self.resolve_cached_node(snapshot);
        self.route_nodes(snapshot.route_candidates(sink), sink, hardware_sink);
    }

    fn cleanup_stale_nodes(&mut self, snapshot: &pw_registry::RegistrySnapshot) {
        let active_nodes: FxHashSet<_> = snapshot.nodes.iter().map(|n| n.id).collect();

        self.disabled_nodes.retain(|id| active_nodes.contains(id));
        self.routed_nodes.retain(|id, _| active_nodes.contains(id));

        if let Some((id, _)) = self.cached_target
            && !active_nodes.contains(&id)
        {
            self.cached_target = None;
        }
    }

    fn route_to_target(
        &mut self,
        node: &pw_registry::NodeInfo,
        target: &pw_registry::NodeInfo,
        desired: RouteTarget,
    ) {
        let now = Instant::now();
        let should_route = self
            .routed_nodes
            .get(&node.id)
            .is_none_or(|state| state.should_retry(desired, now, self.router_epoch));

        if !should_route {
            return;
        }

        let Some(router) = self.ensure_router() else {
            return;
        };

        let result = router.route_application_to_sink(node, target);
        let state = self
            .routed_nodes
            .entry(node.id)
            .or_insert_with(|| RouteState::new(desired));

        match result {
            Ok(()) => {
                state.mark_success(desired, now, self.router_epoch);
                let action = if matches!(desired, RouteTarget::Virtual(_)) {
                    "routed"
                } else {
                    "restored"
                };
                info!(
                    "[router] {action} '{}' #{} -> '{}' #{}",
                    node.display_name(),
                    node.id,
                    target.display_name(),
                    target.id
                );
            }
            Err(err) => {
                error!(
                    "[router] route failed '{}' #{}: {err:?}",
                    node.display_name(),
                    node.id
                );
                state.mark_failure(desired, now);
                self.router = None;
                self.router_retry_after = Some(now + ROUTER_RECOVERY_DELAY);
            }
        }
    }

    fn ensure_router(&mut self) -> Option<&pw_router::Router> {
        if self.router.as_ref().is_some_and(|r| r.is_healthy()) {
            return self.router.as_ref();
        }

        if self.router.is_some() {
            warn!("[router] detected stale metadata proxy; scheduling reconnection");
            self.router = None;
            self.router_retry_after = Some(Instant::now() + ROUTER_RECOVERY_DELAY);
        }

        let now = Instant::now();
        if self.router_retry_after.is_some_and(|t| now < t) {
            return None;
        }

        match pw_router::Router::new() {
            Ok(router) => {
                info!("[router] reinitialised PipeWire router connection");
                self.router = Some(router);
                self.router_retry_after = None;
                self.router_epoch = self.router_epoch.wrapping_add(1).max(1);
            }
            Err(err) => {
                error!("[router] failed to reinitialise router: {err:?}");
                self.router_retry_after = Some(now + ROUTER_RETRY_DELAY);
            }
        }

        self.router.as_ref()
    }

    fn handle_idle(&mut self) {
        if self.router.is_none() {
            self.ensure_router();
        }

        let now = Instant::now();
        if !self
            .routed_nodes
            .values()
            .any(|s| s.should_retry(s.target, now, self.router_epoch))
        {
            return;
        }

        if let Some(snapshot) = self.snapshot.take() {
            self.apply_snapshot(&snapshot, false);
            self.snapshot = Some(snapshot);
        }
    }

    fn resolve_cached_node<'a>(
        &mut self,
        snapshot: &'a pw_registry::RegistrySnapshot,
    ) -> Option<&'a pw_registry::NodeInfo> {
        if let Some(target) = snapshot.defaults.audio_sink.as_ref()
            && let Some(node) = snapshot.resolve_default_target(target)
        {
            self.cached_target = Some((node.id, node.display_name()));
            return Some(node);
        }

        if let Some((id, label)) = &self.cached_target
            && let Some(node) = snapshot
                .nodes
                .iter()
                .find(|n| n.id == *id || n.matches_label(label))
        {
            self.cached_target = Some((node.id, node.display_name()));
            return Some(node);
        }

        self.cached_target = None;
        None
    }

    fn compute_desired_links(
        &mut self,
        snapshot: &pw_registry::RegistrySnapshot,
    ) -> Vec<pw_loopback::LinkSpec> {
        let Some(openmeters_sink) = snapshot.find_node_by_label(VIRTUAL_SINK_NAME) else {
            return Vec::new();
        };

        // Determine source and target nodes based on capture mode
        let (source_node, target_node) = match (self.capture_mode, self.device_target) {
            (CaptureMode::Applications, _) => {
                // Forward: openmeters.sink -> default hardware sink
                self.device_missing_warned = false;
                let Some(hardware_sink) = self.resolve_default_sink_node(snapshot) else {
                    return Vec::new();
                };
                (openmeters_sink, hardware_sink)
            }
            (CaptureMode::Device, DeviceSelection::Default) => {
                // Capture from default: default hardware sink -> openmeters.sink
                self.device_missing_warned = false;
                let Some(hardware_sink) = self.resolve_default_sink_node(snapshot) else {
                    return Vec::new();
                };
                (hardware_sink, openmeters_sink)
            }
            (CaptureMode::Device, DeviceSelection::Node(id)) => {
                // Capture from specific node
                let source = snapshot.nodes.iter().find(|n| n.id == id);
                match source {
                    Some(node) => {
                        self.device_missing_warned = false;
                        (node, openmeters_sink)
                    }
                    None => {
                        if !self.device_missing_warned {
                            warn!("[router] capture node #{id} unavailable; using default");
                            self.device_missing_warned = true;
                        }
                        let Some(hardware_sink) = self.resolve_default_sink_node(snapshot) else {
                            return Vec::new();
                        };
                        (hardware_sink, openmeters_sink)
                    }
                }
            }
        };

        // Compute port pairs
        let source_ports = source_node.output_ports_for_loopback();
        let target_ports = target_node.input_ports_for_loopback();

        if source_ports.is_empty() {
            debug!(
                "[loopback] no output ports on source node #{} ({})",
                source_node.id,
                source_node.display_name()
            );
            return Vec::new();
        }

        if target_ports.is_empty() {
            debug!(
                "[loopback] no input ports on target node #{} ({})",
                target_node.id,
                target_node.display_name()
            );
            return Vec::new();
        }

        let pairs = pair_ports_by_channel(source_ports, target_ports);
        pairs
            .into_iter()
            .map(|(out_port, in_port)| pw_loopback::LinkSpec {
                output_node: source_node.id,
                output_port: out_port.port_id,
                input_node: target_node.id,
                input_port: in_port.port_id,
            })
            .collect()
    }

    fn resolve_default_sink_node<'a>(
        &mut self,
        snapshot: &'a pw_registry::RegistrySnapshot,
    ) -> Option<&'a pw_registry::NodeInfo> {
        // Try to resolve from metadata defaults
        if let Some(target) = snapshot.defaults.audio_sink.as_ref()
            && let Some(node) = snapshot.resolve_default_target(target)
        {
            return Some(node);
        }

        // Fall back to cached target
        if let Some((id, label)) = &self.cached_target
            && let Some(node) = snapshot
                .nodes
                .iter()
                .find(|n| n.id == *id || n.matches_label(label))
        {
            return Some(node);
        }

        None
    }

    fn ensure_links(&mut self, desired: Vec<pw_loopback::LinkSpec>) {
        if self.current_links != desired && pw_loopback::set_links(desired.clone()) {
            self.current_links = desired;
        }
    }

    fn route_nodes<'a>(
        &mut self,
        nodes: impl IntoIterator<Item = &'a pw_registry::NodeInfo>,
        sink: &pw_registry::NodeInfo,
        hardware_sink: Option<&pw_registry::NodeInfo>,
    ) {
        for node in nodes {
            let (target, desired) = match self.capture_mode {
                CaptureMode::Applications => {
                    if self.is_node_enabled(node.id) {
                        (Some(sink), RouteTarget::Virtual(sink.id))
                    } else {
                        match hardware_sink {
                            Some(hw) => (Some(hw), RouteTarget::Hardware(hw.id)),
                            None => (None, RouteTarget::Hardware(0)),
                        }
                    }
                }
                CaptureMode::Device => match hardware_sink {
                    Some(hw) => (Some(hw), RouteTarget::Hardware(hw.id)),
                    None => (None, RouteTarget::Hardware(0)),
                },
            };

            if let Some(target) = target {
                self.route_to_target(node, target, desired);
            } else if self.routed_nodes.remove(&node.id).is_some() {
                warn!(
                    "[router] unable to restore '{}' (id={}); no sink",
                    node.display_name(),
                    node.id
                );
            }
        }
    }

    fn is_node_enabled(&self, node_id: u32) -> bool {
        !self.disabled_nodes.contains(&node_id)
    }
}
