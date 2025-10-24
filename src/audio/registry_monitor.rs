use crate::audio::{VIRTUAL_SINK_NAME, pw_loopback, pw_registry, pw_router};
use crate::ui::RoutingCommand;
use crate::ui::app::config::{CaptureMode, DeviceSelection};
use crate::util::log;
use async_channel::Sender;
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};
use tracing::{error, info, warn};

const ROUTER_RECOVERY_DELAY: Duration = Duration::from_millis(500);
const ROUTER_RETRY_DELAY: Duration = Duration::from_secs(5);

pub fn init_registry_monitor(
    command_rx: mpsc::Receiver<RoutingCommand>,
    snapshot_tx: Sender<pw_registry::RegistrySnapshot>,
) -> Option<pw_registry::AudioRegistryHandle> {
    match pw_registry::spawn_registry() {
        Ok(handle) => {
            if let Err(err) = spawn_registry_monitor(handle.clone(), command_rx, snapshot_tx) {
                error!("[registry] failed to spawn monitor thread: {err}");
            }
            Some(handle)
        }
        Err(err) => {
            error!("[registry] failed to start PipeWire registry: {err:?}");
            None
        }
    }
}

fn spawn_registry_monitor(
    handle: pw_registry::AudioRegistryHandle,
    command_rx: mpsc::Receiver<RoutingCommand>,
    snapshot_tx: Sender<pw_registry::RegistrySnapshot>,
) -> std::io::Result<()> {
    std::thread::Builder::new()
        .name("openmeters-registry-monitor".into())
        .spawn(move || run_registry_monitor(handle, command_rx, snapshot_tx))
        .map(|_| ())
}

fn run_registry_monitor(
    handle: pw_registry::AudioRegistryHandle,
    command_rx: mpsc::Receiver<RoutingCommand>,
    snapshot_tx: Sender<pw_registry::RegistrySnapshot>,
) {
    let mut updates = handle.subscribe();
    let router = match pw_router::Router::new() {
        Ok(router) => Some(router),
        Err(err) => {
            error!("[router] failed to initialise PipeWire router: {err:?}");
            None
        }
    };

    let mut monitor = RegistryMonitor::new(router, command_rx);
    const POLL_INTERVAL: Duration = Duration::from_millis(100);

    loop {
        if monitor.process_pending_commands() {
            continue;
        }

        match updates.recv_timeout(POLL_INTERVAL) {
            Ok(Some(snapshot)) => {
                monitor.process_snapshot(&snapshot);
                if snapshot_tx.send_blocking(snapshot.clone()).is_err() {
                    info!("[registry] UI channel closed; stopping snapshot forwarding");
                    break;
                }
            }
            Ok(None) | Err(mpsc::RecvTimeoutError::Timeout) => {
                monitor.handle_idle();
                continue;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    info!("[registry] update stream ended");
}

struct RegistryMonitor {
    iteration: u64,
    routing: RoutingManager,
}

impl RegistryMonitor {
    fn new(router: Option<pw_router::Router>, command_rx: mpsc::Receiver<RoutingCommand>) -> Self {
        Self {
            iteration: 0,
            routing: RoutingManager::new(router, command_rx),
        }
    }

    fn process_snapshot(&mut self, snapshot: &pw_registry::RegistrySnapshot) {
        let label = if self.iteration == 0 {
            "initial snapshot"
        } else {
            "update"
        };

        log::registry_snapshot(label, snapshot);
        self.iteration += 1;

        self.routing.handle_snapshot(snapshot);
    }

    fn process_pending_commands(&mut self) -> bool {
        self.routing.apply_pending_commands()
    }

    fn handle_idle(&mut self) {
        self.routing.handle_idle();
    }
}

struct RoutingManager {
    router: Option<pw_router::Router>,
    commands: mpsc::Receiver<RoutingCommand>,
    preferences: FxHashMap<u32, bool>,
    routed_nodes: FxHashMap<u32, RouteState>,
    capture_mode: CaptureMode,
    device_target: DeviceSelection,
    last_sink_id: Option<u32>,
    sink_warning_logged: bool,
    last_snapshot: Option<Arc<pw_registry::RegistrySnapshot>>,
    last_hardware_sink_id: Option<u32>,
    last_hardware_sink_label: Option<String>,
    last_capture_node_id: Option<u32>,
    last_capture_label: Option<String>,
    capture_warning_logged: bool,
    router_retry_after: Option<Instant>,
    router_epoch: u64,
    loopback_mode: Option<pw_loopback::LoopbackMode>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RouteTarget {
    Virtual(u32),
    Hardware(u32),
}

#[derive(Debug, Clone)]
struct RouteState {
    target: RouteTarget,
    last_success: Option<Instant>,
    last_attempt: Option<Instant>,
    consecutive_failures: u32,
    last_success_epoch: u64,
}

impl RouteState {
    const RETRY_INTERVAL: Duration = Duration::from_millis(350);

    fn new(target: RouteTarget) -> Self {
        Self {
            target,
            last_success: None,
            last_attempt: None,
            consecutive_failures: 0,
            last_success_epoch: 0,
        }
    }

    fn mark_success(&mut self, target: RouteTarget, now: Instant, epoch: u64) {
        self.target = target;
        self.last_attempt = Some(now);
        self.last_success = Some(now);
        self.consecutive_failures = 0;
        self.last_success_epoch = epoch;
    }

    fn mark_failure(&mut self, target: RouteTarget, now: Instant) {
        self.target = target;
        self.last_attempt = Some(now);
        self.consecutive_failures = self.consecutive_failures.saturating_add(1);
    }

    fn should_retry(&self, desired: RouteTarget, now: Instant, epoch: u64) -> bool {
        self.target != desired || self.should_refresh(now, epoch)
    }

    fn should_refresh(&self, now: Instant, epoch: u64) -> bool {
        if self.last_success_epoch < epoch || self.last_success.is_none() {
            return true;
        }

        if self.consecutive_failures == 0 {
            return false;
        }

        match self.last_attempt {
            Some(last_attempt) => now.duration_since(last_attempt) >= Self::RETRY_INTERVAL,
            None => true,
        }
    }
}

impl RoutingManager {
    fn new(router: Option<pw_router::Router>, commands: mpsc::Receiver<RoutingCommand>) -> Self {
        let router_epoch = if router.is_some() { 1 } else { 0 };
        Self {
            router,
            commands,
            preferences: FxHashMap::default(),
            routed_nodes: FxHashMap::default(),
            capture_mode: CaptureMode::Applications,
            device_target: DeviceSelection::Default,
            last_sink_id: None,
            sink_warning_logged: false,
            last_snapshot: None,
            last_hardware_sink_id: None,
            last_hardware_sink_label: None,
            last_capture_node_id: None,
            last_capture_label: None,
            capture_warning_logged: false,
            router_retry_after: None,
            router_epoch,
            loopback_mode: None,
        }
    }

    fn handle_snapshot(&mut self, snapshot: &pw_registry::RegistrySnapshot) {
        self.last_snapshot = Some(Arc::new(snapshot.clone()));
        self.consume_commands();
        self.apply_snapshot(snapshot, true);
    }

    fn cleanup_removed_nodes(&mut self, observed: &FxHashSet<u32>) {
        self.preferences
            .retain(|node_id, _| observed.contains(node_id));
        self.routed_nodes
            .retain(|node_id, _| observed.contains(node_id));

        if let Some(id) = self.last_hardware_sink_id
            && !observed.contains(&id)
        {
            self.last_hardware_sink_id = None;
        }

        if let Some(id) = self.last_capture_node_id
            && !observed.contains(&id)
        {
            self.last_capture_node_id = None;
            self.last_capture_label = None;
        }
    }

    fn consume_commands(&mut self) -> bool {
        let mut changed = false;
        while let Ok(command) = self.commands.try_recv() {
            changed |= match command {
                RoutingCommand::SetApplicationEnabled { node_id, enabled } => {
                    let was_enabled = self.preferences.get(&node_id).copied().unwrap_or(true);
                    if was_enabled != enabled {
                        if enabled {
                            self.preferences.remove(&node_id);
                        } else {
                            self.preferences.insert(node_id, false);
                        }
                        true
                    } else {
                        false
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

        if let Some(snapshot) = self.last_snapshot.as_ref().cloned() {
            self.apply_snapshot(snapshot.as_ref(), false);
        }

        true
    }

    fn apply_snapshot(&mut self, snapshot: &pw_registry::RegistrySnapshot, log_sink_missing: bool) {
        let observed: FxHashSet<u32> = snapshot.nodes.iter().map(|node| node.id).collect();
        self.cleanup_removed_nodes(&observed);

        let desired_loopback = self.desired_loopback_mode(snapshot);
        self.ensure_loopback_mode(desired_loopback);

        let Some(sink) = snapshot.find_node_by_label(VIRTUAL_SINK_NAME) else {
            if log_sink_missing && !self.sink_warning_logged {
                warn!("[router] virtual sink '{VIRTUAL_SINK_NAME}' not yet available");
                self.sink_warning_logged = true;
            }
            return;
        };

        if self.last_sink_id != Some(sink.id) {
            self.routed_nodes.clear();
            self.last_sink_id = Some(sink.id);
        }

        self.sink_warning_logged = false;

        let candidates: Vec<_> = snapshot.route_candidates(sink).collect();
        let hardware_sink = self.resolve_hardware_sink(snapshot);

        match self.capture_mode {
            CaptureMode::Applications => {
                self.handle_applications_mode(&candidates, sink, hardware_sink)
            }
            CaptureMode::Device => self.handle_device_mode(&candidates, hardware_sink),
        }
    }

    fn route_to_target(
        &mut self,
        node: &pw_registry::NodeInfo,
        target: &pw_registry::NodeInfo,
        desired: RouteTarget,
    ) {
        let now = Instant::now();
        if !self
            .routed_nodes
            .get(&node.id)
            .is_none_or(|state| state.should_retry(desired, now, self.router_epoch))
        {
            return;
        }

        let route_result = match self.ensure_router() {
            Some(router) => router.route_application_to_sink(node, target),
            None => return,
        };

        let node_label = node.display_name();
        if route_result.is_ok() {
            let state = self
                .routed_nodes
                .entry(node.id)
                .or_insert_with(|| RouteState::new(desired));
            state.mark_success(desired, now, self.router_epoch);

            let action = if matches!(desired, RouteTarget::Virtual(_)) {
                "routed"
            } else {
                "restored"
            };
            info!(
                "[router] {action} '{}' #{} -> '{}' #{}",
                node_label,
                node.id,
                target.display_name(),
                target.id
            );
            return;
        }

        let err = route_result.err().unwrap();
        error!(
            "[router] route failed '{}' #{}: {err:?}",
            node_label, node.id
        );
        let state = self
            .routed_nodes
            .entry(node.id)
            .or_insert_with(|| RouteState::new(desired));
        state.mark_failure(desired, now);

        self.router = None;
        self.router_retry_after = Some(now + ROUTER_RECOVERY_DELAY);
    }

    fn ensure_router(&mut self) -> Option<&pw_router::Router> {
        if self
            .router
            .as_ref()
            .is_some_and(|router| router.is_healthy())
        {
            return self.router.as_ref();
        }

        if self
            .router
            .as_ref()
            .is_some_and(|router| !router.is_healthy())
        {
            warn!("[router] detected stale metadata proxy; scheduling router reconnection");
            self.router = None;
            self.router_retry_after = Some(Instant::now() + ROUTER_RECOVERY_DELAY);
        }

        let now = Instant::now();
        if let Some(retry_after) = self.router_retry_after
            && now < retry_after
        {
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
        let now = Instant::now();

        if self.router.is_none() {
            self.ensure_router();
        }

        let Some(snapshot) = self.last_snapshot.as_ref().cloned() else {
            return;
        };

        if !self
            .routed_nodes
            .values()
            .any(|state| state.should_refresh(now, self.router_epoch))
        {
            return;
        }

        self.apply_snapshot(snapshot.as_ref(), false);
    }

    fn resolve_hardware_sink<'a>(
        &mut self,
        snapshot: &'a pw_registry::RegistrySnapshot,
    ) -> Option<&'a pw_registry::NodeInfo> {
        if let Some(target) = snapshot.defaults.audio_sink.as_ref()
            && let Some(node) = snapshot.resolve_default_target(target)
        {
            self.last_hardware_sink_id = Some(node.id);
            self.last_hardware_sink_label = Some(node.display_name());
            return Some(node);
        }

        if let Some(id) = self.last_hardware_sink_id
            && let Some(node) = snapshot.nodes.iter().find(|node| node.id == id)
        {
            self.last_hardware_sink_id = Some(node.id);
            self.last_hardware_sink_label = Some(node.display_name());
            return Some(node);
        }

        if let Some(label) = self.last_hardware_sink_label.as_deref()
            && let Some(node) = snapshot.nodes.iter().find(|node| node.matches_label(label))
        {
            self.last_hardware_sink_id = Some(node.id);
            self.last_hardware_sink_label = Some(node.display_name());
            return Some(node);
        }

        self.last_hardware_sink_id = None;
        None
    }

    fn desired_loopback_mode(
        &mut self,
        snapshot: &pw_registry::RegistrySnapshot,
    ) -> pw_loopback::LoopbackMode {
        use pw_loopback::{DeviceCaptureTarget, LoopbackMode};

        let mode = match (self.capture_mode, self.device_target) {
            (CaptureMode::Applications, _) => LoopbackMode::ForwardToDefaultSink,
            (CaptureMode::Device, DeviceSelection::Default) => {
                LoopbackMode::CaptureFromDevice(DeviceCaptureTarget::Default)
            }
            (CaptureMode::Device, DeviceSelection::Node(id)) => {
                return self.capture_specific_node(snapshot, id);
            }
        };
        self.last_capture_node_id = None;
        self.last_capture_label = None;
        self.capture_warning_logged = false;
        mode
    }

    fn ensure_loopback_mode(&mut self, desired: pw_loopback::LoopbackMode) {
        if self.loopback_mode != Some(desired) {
            self.loopback_mode = pw_loopback::set_mode(desired).then_some(desired);
        }
    }

    fn is_node_enabled(&self, node_id: u32) -> bool {
        self.preferences.get(&node_id).copied().unwrap_or(true)
    }

    fn handle_applications_mode(
        &mut self,
        nodes: &[&pw_registry::NodeInfo],
        sink: &pw_registry::NodeInfo,
        hardware_sink: Option<&pw_registry::NodeInfo>,
    ) {
        for node in nodes {
            if self.is_node_enabled(node.id) {
                self.route_to_target(node, sink, RouteTarget::Virtual(sink.id));
            } else if let Some(hardware) = hardware_sink {
                self.route_to_target(node, hardware, RouteTarget::Hardware(hardware.id));
            } else if self.routed_nodes.remove(&node.id).is_some() {
                warn!(
                    "[router] no hardware sink for '{}' (id={})",
                    node.display_name(),
                    node.id
                );
            }
        }
    }

    fn handle_device_mode(
        &mut self,
        nodes: &[&pw_registry::NodeInfo],
        hardware_sink: Option<&pw_registry::NodeInfo>,
    ) {
        match hardware_sink {
            Some(hardware) => {
                for node in nodes {
                    self.route_to_target(node, hardware, RouteTarget::Hardware(hardware.id));
                }
            }
            None => {
                for node in nodes {
                    if self.routed_nodes.remove(&node.id).is_some() {
                        warn!(
                            "[router] unable to restore '{}' (id={}); no sink",
                            node.display_name(),
                            node.id
                        );
                    }
                }
            }
        }
    }

    fn capture_specific_node(
        &mut self,
        snapshot: &pw_registry::RegistrySnapshot,
        node_id: u32,
    ) -> pw_loopback::LoopbackMode {
        use pw_loopback::{DeviceCaptureTarget, LoopbackMode};

        if let Some(node) = snapshot.nodes.iter().find(|n| n.id == node_id) {
            self.last_capture_node_id = Some(node_id);
            self.last_capture_label = Some(node.display_name());
            self.capture_warning_logged = false;
            return LoopbackMode::CaptureFromDevice(DeviceCaptureTarget::Node(node_id));
        }

        if !self.capture_warning_logged {
            let label = self.last_capture_label.as_deref().unwrap_or("unknown");
            warn!(
                "[router] capture node '{}' (#{node_id}) unavailable; using default",
                label
            );
            self.capture_warning_logged = true;
        }

        self.last_capture_node_id = None;
        self.last_capture_label = None;
        LoopbackMode::CaptureFromDevice(DeviceCaptureTarget::Default)
    }
}
