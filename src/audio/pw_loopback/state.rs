use super::{DeviceCaptureTarget, LoopbackMode, OPENMETERS_SINK_NAME};
use crate::util::pipewire::{
    DefaultTarget, GraphNode, GraphPort, PortDirection, create_passive_audio_link,
    pair_ports_by_channel, parse_metadata_name,
};
use pipewire as pw;
use rustc_hash::{FxHashMap, FxHashSet};
use tracing::{debug, error, info, warn};

type RoutePlan = (u32, u32, Vec<(GraphPort, GraphPort)>);

pub(super) struct LoopbackState {
    core: pw::core::CoreRc,
    nodes: FxHashMap<u32, TrackedNode>,
    port_index: FxHashMap<u32, (u32, u32)>,
    default_sink: DefaultTarget,
    openmeters_node_id: Option<u32>,
    active_links: FxHashMap<LinkKey, pw::link::Link>,
    mode: LoopbackMode,
}

impl LoopbackState {
    pub(super) fn new(core: pw::core::CoreRc) -> Self {
        Self {
            core,
            nodes: FxHashMap::default(),
            port_index: FxHashMap::default(),
            default_sink: DefaultTarget::default(),
            openmeters_node_id: None,
            active_links: FxHashMap::default(),
            mode: LoopbackMode::ForwardToDefaultSink,
        }
    }

    pub(super) fn upsert_node(&mut self, node: GraphNode) {
        let node_id = node.id();
        let entry = self.nodes.entry(node_id).or_default();
        entry.set_info(node);

        if entry.has_name(OPENMETERS_SINK_NAME) {
            self.set_openmeters_node(Some(node_id));
        } else if self.openmeters_node_id == Some(node_id) {
            self.set_openmeters_node(None);
        }

        self.resolve_default_sink_node();
        self.refresh_links();
    }

    pub(super) fn remove_node(&mut self, node_id: u32) -> bool {
        let existed = self.nodes.remove(&node_id).is_some();
        if !existed {
            return false;
        }

        self.port_index.retain(|_, (owner, _)| *owner != node_id);

        if self.openmeters_node_id == Some(node_id) {
            self.set_openmeters_node(None);
        }

        if self.default_sink.node_id == Some(node_id) {
            self.default_sink.node_id = None;
        }

        self.resolve_default_sink_node();
        self.refresh_links();
        true
    }

    pub(super) fn upsert_port(&mut self, port: GraphPort) {
        let node_id = port.node_id;
        let port_id = port.port_id;
        let node = self.nodes.entry(node_id).or_default();
        node.upsert_port(port.clone());
        self.port_index.insert(port.global_id, (node_id, port_id));

        if self.is_tracked_node(node_id) {
            debug!("[loopback] port discovered: node={node_id} port={port_id}");
            self.refresh_links();
        }
    }

    pub(super) fn remove_port_by_global(&mut self, global_id: u32) -> bool {
        let Some((node_id, port_id)) = self.port_index.remove(&global_id) else {
            return false;
        };

        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.remove_port(port_id);
        }

        if self.is_tracked_node(node_id) {
            debug!("[loopback] port removed: node={node_id} port={port_id}");
            self.refresh_links();
        }

        true
    }

    pub(super) fn update_default_sink(
        &mut self,
        metadata_id: u32,
        subject: u32,
        type_hint: Option<&str>,
        value: Option<&str>,
    ) {
        let parsed_name = parse_metadata_name(type_hint, value);
        let changed =
            self.default_sink
                .update(metadata_id, subject, type_hint, parsed_name.as_deref());

        self.resolve_default_sink_node();

        if changed {
            if let Some(node_id) = self.default_sink.node_id {
                info!("[loopback] default audio sink is node #{node_id}");
            } else if let Some(name) = self.default_sink.name.as_deref() {
                info!("[loopback] default audio sink set to '{name}' (node unresolved)");
            } else {
                info!("[loopback] default audio sink cleared");
            }
        }

        self.refresh_links();
    }

    pub(super) fn clear_metadata(&mut self, metadata_id: u32) {
        if self.default_sink.metadata_id == Some(metadata_id) {
            self.default_sink.clear();
            self.refresh_links();
        }
    }

    pub(super) fn set_mode(&mut self, mode: LoopbackMode) {
        if self.mode == mode {
            return;
        }

        match mode {
            LoopbackMode::ForwardToDefaultSink => {
                info!("[loopback] forwarding OpenMeters sink to default hardware sink");
            }
            LoopbackMode::CaptureFromDevice(DeviceCaptureTarget::Default) => {
                info!("[loopback] capturing audio from default hardware sink monitor");
            }
            LoopbackMode::CaptureFromDevice(DeviceCaptureTarget::Node(id)) => {
                if let Some(node) = self.nodes.get(&id).and_then(|node| node.info.as_ref()) {
                    let label = node
                        .description()
                        .or_else(|| node.name())
                        .unwrap_or("unnamed");
                    info!("[loopback] capturing audio from node #{} ({label})", id);
                } else {
                    info!("[loopback] capturing audio from node #{id} (details unresolved)");
                }
            }
        }

        self.mode = mode;
        self.refresh_links();
    }

    fn refresh_links(&mut self) {
        let plan = match self.mode {
            LoopbackMode::ForwardToDefaultSink => self.compute_forward_plan(),
            LoopbackMode::CaptureFromDevice(target) => self.compute_capture_plan(target),
        };

        if let Some((source_id, target_id, pairs)) = plan {
            self.apply_plan(source_id, target_id, pairs);
        }
    }

    fn compute_forward_plan(&mut self) -> Option<RoutePlan> {
        let source_id = self.openmeters_node_id?;
        let target_id = self.default_sink.node_id?;
        self.build_route_plan(source_id, target_id)
    }

    fn compute_capture_plan(&mut self, capture_target: DeviceCaptureTarget) -> Option<RoutePlan> {
        let target_id = self.openmeters_node_id?;
        let source_id = match capture_target {
            DeviceCaptureTarget::Default => {
                self.resolve_default_sink_node();
                self.default_sink.node_id.or_else(|| {
                    let name = self.default_sink.name.clone()?;
                    let id = self
                        .nodes
                        .iter()
                        .find_map(|(&id, node)| node.matches_name(&name).then_some(id))?;
                    self.default_sink.node_id = Some(id);
                    Some(id)
                })?
            }
            DeviceCaptureTarget::Node(id) => self.nodes.contains_key(&id).then_some(id)?,
        };
        if source_id == target_id {
            return None;
        }
        self.build_route_plan(source_id, target_id)
    }

    fn build_route_plan(&mut self, source_id: u32, target_id: u32) -> Option<RoutePlan> {
        let source_ports = self.select_ports(source_id, "source", "output", |node| {
            node.output_ports_for_loopback()
        })?;
        let target_ports = self.select_ports(target_id, "target", "input", |node| {
            node.input_ports_for_loopback()
        })?;
        Some((
            source_id,
            target_id,
            pair_ports_by_channel(source_ports, target_ports),
        ))
    }

    fn apply_plan(&mut self, source_id: u32, target_id: u32, pairs: Vec<(GraphPort, GraphPort)>) {
        let desired_keys: FxHashSet<LinkKey> = pairs
            .iter()
            .map(|(out_port, in_port)| LinkKey {
                output_node: source_id,
                output_port: out_port.port_id,
                input_node: target_id,
                input_port: in_port.port_id,
            })
            .collect();
        self.prune_stale_links(&desired_keys);

        for (output_port, input_port) in pairs {
            let key = LinkKey {
                output_node: source_id,
                output_port: output_port.port_id,
                input_node: target_id,
                input_port: input_port.port_id,
            };
            if self.active_links.contains_key(&key) {
                continue;
            }

            match create_passive_audio_link(
                &self.core,
                source_id,
                output_port.port_id,
                target_id,
                input_port.port_id,
            ) {
                Ok(link) => {
                    debug!(
                        "[loopback] linked {source_id}:{} -> {target_id}:{}",
                        output_port.port_id, input_port.port_id
                    );
                    self.active_links.insert(key, link);
                }
                Err(err) => {
                    error!(
                        "[loopback] link failed {source_id}:{} -> {target_id}:{}: {err}",
                        output_port.port_id, input_port.port_id
                    );
                }
            }
        }
    }

    fn prune_stale_links(&mut self, desired_keys: &FxHashSet<LinkKey>) {
        self.active_links.retain(|key, _| {
            let keep = desired_keys.contains(key);
            if !keep {
                debug!(
                    "[loopback] removed link {}:{} -> {}:{}",
                    key.output_node, key.output_port, key.input_node, key.input_port
                );
            }
            keep
        });
    }

    fn select_ports<F>(
        &mut self,
        node_id: u32,
        label: &str,
        port_role: &str,
        selector: F,
    ) -> Option<Vec<GraphPort>>
    where
        F: Fn(&TrackedNode) -> Vec<GraphPort>,
    {
        let node = match self.nodes.get(&node_id) {
            Some(node) => node,
            None => {
                self.clear_links();
                warn!(
                    "[loopback] {label} node {} missing; cleared active links",
                    node_id
                );
                return None;
            }
        };
        let ports = selector(node);
        if ports.is_empty() {
            self.clear_links();
            warn!("[loopback] no {port_role} ports on node {node_id}");
            return None;
        }

        Some(ports)
    }

    fn clear_links(&mut self) {
        if !self.active_links.is_empty() {
            self.active_links.clear();
            info!("[loopback] cleared all active links");
        }
    }

    fn resolve_default_sink_node(&mut self) {
        if let Some(node_id) = self.default_sink.node_id
            && self.nodes.contains_key(&node_id)
        {
            return;
        }

        if let Some(name) = self.default_sink.name.clone()
            && let Some(node_id) = self
                .nodes
                .iter()
                .find_map(|(&id, node)| node.matches_name(&name).then_some(id))
        {
            self.default_sink.node_id = Some(node_id);
        }
    }

    fn set_openmeters_node(&mut self, node_id: Option<u32>) {
        if self.openmeters_node_id != node_id {
            match node_id {
                Some(id) => info!("[loopback] detected OpenMeters sink node #{id}"),
                None if self.openmeters_node_id.is_some() => {
                    info!("[loopback] OpenMeters sink node removed")
                }
                None => {}
            }
            self.openmeters_node_id = node_id;
        }
    }

    fn is_tracked_node(&self, node_id: u32) -> bool {
        self.openmeters_node_id == Some(node_id) || self.default_sink.node_id == Some(node_id)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct LinkKey {
    output_node: u32,
    output_port: u32,
    input_node: u32,
    input_port: u32,
}

#[derive(Default)]
struct TrackedNode {
    info: Option<GraphNode>,
    ports: FxHashMap<u32, GraphPort>,
}

impl TrackedNode {
    fn set_info(&mut self, info: GraphNode) {
        self.info = Some(info);
    }

    fn upsert_port(&mut self, port: GraphPort) {
        self.ports.insert(port.port_id, port);
    }

    fn remove_port(&mut self, port_id: u32) {
        self.ports.remove(&port_id);
    }

    fn matches_name(&self, candidate: &str) -> bool {
        self.info
            .as_ref()
            .is_some_and(|info| info.matches_name(candidate))
    }

    fn has_name(&self, name: &str) -> bool {
        self.info.as_ref().is_some_and(|info| info.has_name(name))
    }

    fn output_ports_for_loopback(&self) -> Vec<GraphPort> {
        self.ports_for_loopback(
            |port| port.direction == PortDirection::Output && port.is_monitor,
            |port| port.direction == PortDirection::Output,
        )
    }

    fn input_ports_for_loopback(&self) -> Vec<GraphPort> {
        self.ports_for_loopback(
            |port| port.direction == PortDirection::Input && !port.is_monitor,
            |port| port.direction == PortDirection::Input,
        )
    }

    fn ports_for_loopback<F, G>(&self, primary: F, secondary: G) -> Vec<GraphPort>
    where
        F: Fn(&GraphPort) -> bool,
        G: Fn(&GraphPort) -> bool,
    {
        if let Some(ports) = self.collect_if(&primary) {
            return ports;
        }
        if let Some(ports) = self.collect_if(&secondary) {
            return ports;
        }
        self.ports.values().cloned().collect()
    }

    fn collect_if<F>(&self, predicate: &F) -> Option<Vec<GraphPort>>
    where
        F: Fn(&GraphPort) -> bool,
    {
        let ports: Vec<GraphPort> = self
            .ports
            .values()
            .filter(|port| predicate(port))
            .cloned()
            .collect();

        (!ports.is_empty()).then_some(ports)
    }
}
