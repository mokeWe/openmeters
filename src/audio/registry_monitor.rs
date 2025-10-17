use crate::audio::{VIRTUAL_SINK_NAME, pw_registry, pw_router};
use crate::ui::RoutingCommand;
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
            Ok(None) => continue,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
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
}

struct RoutingManager {
    router: Option<pw_router::Router>,
    commands: mpsc::Receiver<RoutingCommand>,
    preferences: FxHashMap<u32, bool>,
    routed_nodes: FxHashMap<u32, RouteState>,
    last_sink_id: Option<u32>,
    sink_warning_logged: bool,
    last_snapshot: Option<Arc<pw_registry::RegistrySnapshot>>,
    last_hardware_sink_id: Option<u32>,
    last_hardware_sink_label: Option<String>,
    router_retry_after: Option<Instant>,
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
}

impl RouteState {
    const REFRESH_INTERVAL: Duration = Duration::from_secs(2);
    const RETRY_INTERVAL: Duration = Duration::from_millis(350);

    fn new(target: RouteTarget) -> Self {
        Self {
            target,
            last_success: None,
            last_attempt: None,
            consecutive_failures: 0,
        }
    }

    fn mark_success(&mut self, target: RouteTarget, now: Instant) {
        self.target = target;
        self.last_attempt = Some(now);
        self.last_success = Some(now);
        self.consecutive_failures = 0;
    }

    fn mark_failure(&mut self, target: RouteTarget, now: Instant) {
        self.target = target;
        self.last_attempt = Some(now);
        self.consecutive_failures = self.consecutive_failures.saturating_add(1);
    }

    fn should_retry(&self, target: RouteTarget, now: Instant) -> bool {
        if self.target != target {
            return true;
        }

        if let Some(last_success) = self.last_success {
            if now.duration_since(last_success) >= Self::REFRESH_INTERVAL {
                return true;
            }
        } else {
            // No successful routing recorded yet; allow retry as soon as permitted.
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
        Self {
            router,
            commands,
            preferences: FxHashMap::default(),
            routed_nodes: FxHashMap::default(),
            last_sink_id: None,
            sink_warning_logged: false,
            last_snapshot: None,
            last_hardware_sink_id: None,
            last_hardware_sink_label: None,
            router_retry_after: None,
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
    }

    fn consume_commands(&mut self) -> bool {
        let mut changed = false;
        while let Ok(command) = self.commands.try_recv() {
            match command {
                RoutingCommand::SetApplicationEnabled { node_id, enabled } => {
                    self.preferences.insert(node_id, enabled);
                    changed = true;
                }
            }
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

        let Some(sink) = snapshot.find_node_by_label(VIRTUAL_SINK_NAME) else {
            if log_sink_missing && !self.sink_warning_logged {
                warn!(
                    "[router] virtual sink '{}' not yet available; will retry on future updates",
                    VIRTUAL_SINK_NAME
                );
                self.sink_warning_logged = true;
            }
            return;
        };

        if self.last_sink_id != Some(sink.id) {
            self.routed_nodes.clear();
            self.last_sink_id = Some(sink.id);
        }

        self.sink_warning_logged = false;

        let hardware_sink = self.resolve_hardware_sink(snapshot);

        for node in snapshot.route_candidates(sink) {
            if self.is_node_enabled(node.id) {
                self.route_to_target(node, sink, RouteTarget::Virtual(sink.id));
            } else if let Some(hardware) = hardware_sink {
                self.route_to_target(node, hardware, RouteTarget::Hardware(hardware.id));
            } else if self.routed_nodes.remove(&node.id).is_some() {
                warn!(
                    "[router] no hardware sink available to restore '{}' (id={})",
                    node.display_name(),
                    node.id
                );
            }
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
            .is_none_or(|state| state.should_retry(desired, now))
        {
            return;
        }

        let route_result = {
            let router = match self.ensure_router() {
                Some(router) => router,
                None => return,
            };

            router.route_application_to_sink(node, target)
        };

        if route_result.is_ok() {
            let state = self
                .routed_nodes
                .entry(node.id)
                .or_insert_with(|| RouteState::new(desired));
            state.mark_success(desired, now);

            let action = match desired {
                RouteTarget::Virtual(_) => "routed",
                RouteTarget::Hardware(_) => "restored",
            };
            info!(
                "[router] {action} '{}' (id={}) -> '{}' (id={})",
                node.display_name(),
                node.id,
                target.display_name(),
                target.id
            );
            return;
        }

        let err = route_result.err().unwrap();
        error!(
            "[router] failed to route node '{}' (id={}): {err:?}",
            node.display_name(),
            node.id
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
            }
            Err(err) => {
                error!("[router] failed to reinitialise router: {err:?}");
                self.router_retry_after = Some(now + ROUTER_RETRY_DELAY);
            }
        }

        self.router.as_ref()
    }

    fn resolve_hardware_sink<'a>(
        &mut self,
        snapshot: &'a pw_registry::RegistrySnapshot,
    ) -> Option<&'a pw_registry::NodeInfo> {
        if let Some(target) = snapshot.defaults.audio_sink.as_ref()
            && let Some(node) = snapshot.resolve_default_target(target)
        {
            self.cache_hardware_sink(node);
            return Some(node);
        }

        if let Some(id) = self.last_hardware_sink_id
            && let Some(node) = snapshot.nodes.iter().find(|node| node.id == id)
        {
            self.cache_hardware_sink(node);
            return Some(node);
        }

        if let Some(label) = self.last_hardware_sink_label.as_deref()
            && let Some(node) = snapshot.nodes.iter().find(|node| node.matches_label(label))
        {
            self.cache_hardware_sink(node);
            return Some(node);
        }

        self.last_hardware_sink_id = None;
        None
    }

    fn cache_hardware_sink(&mut self, node: &pw_registry::NodeInfo) {
        self.last_hardware_sink_id = Some(node.id);
        self.last_hardware_sink_label = Some(node.display_name());
    }

    fn is_node_enabled(&self, node_id: u32) -> bool {
        self.preferences.get(&node_id).copied().unwrap_or(true)
    }
}
