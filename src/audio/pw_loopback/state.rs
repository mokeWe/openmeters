use super::OPENMETERS_SINK_NAME;
use crate::util::pipewire::{
    DefaultTarget, GraphNode, GraphPort, PortDirection, create_passive_audio_link,
    format_port_debug, pair_ports_by_channel, parse_metadata_name,
};
use pipewire as pw;
use std::collections::{HashMap, HashSet};

pub(super) struct LoopbackState {
    core: pw::core::CoreRc,
    nodes: HashMap<u32, TrackedNode>,
    port_index: HashMap<u32, (u32, u32)>,
    default_sink: DefaultTarget,
    openmeters_node_id: Option<u32>,
    active_links: HashMap<LinkKey, pw::link::Link>,
}

impl LoopbackState {
    pub(super) fn new(core: pw::core::CoreRc) -> Self {
        Self {
            core,
            nodes: HashMap::new(),
            port_index: HashMap::new(),
            default_sink: DefaultTarget::default(),
            openmeters_node_id: None,
            active_links: HashMap::new(),
        }
    }

    pub(super) fn upsert_node(&mut self, node: GraphNode) {
        let node_id = node.id();
        let entry = self
            .nodes
            .entry(node_id)
            .or_insert_with(TrackedNode::default);
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
        let node = self
            .nodes
            .entry(node_id)
            .or_insert_with(TrackedNode::default);
        node.upsert_port(port.clone());
        self.port_index.insert(port.global_id, (node_id, port_id));

        if self.is_tracked_node(node_id) {
            println!(
                "[loopback] tracked port discovered: node={} port={} dir={:?} monitor={} channel={} name={}",
                node_id,
                port_id,
                port.direction,
                port.is_monitor,
                port.channel.as_deref().unwrap_or("unknown"),
                port.name.as_deref().unwrap_or("unnamed")
            );
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
            println!(
                "[loopback] tracked port removed: node={} port={}",
                node_id, port_id
            );
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
                println!("[loopback] default audio sink is node #{node_id}");
            } else if let Some(name) = self.default_sink.name.as_deref() {
                println!("[loopback] default audio sink set to '{name}' (node unresolved)");
            } else {
                println!("[loopback] default audio sink cleared");
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

    fn refresh_links(&mut self) {
        let Some(plan) = self.compute_route_plan() else {
            return;
        };

        let desired_keys = plan.desired_keys();
        self.prune_stale_links(&desired_keys);

        let RoutePlan {
            source_id,
            target_id,
            pairs,
        } = plan;

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
                    let source_desc = format_port_debug(&output_port);
                    let target_desc = format_port_debug(&input_port);
                    println!(
                        "[loopback] linking node {} {} -> node {} {}",
                        source_id, source_desc, target_id, target_desc
                    );
                    self.active_links.insert(key, link);
                }
                Err(err) => {
                    eprintln!(
                        "[loopback] failed to create link {}:{} -> {}:{}: {err}",
                        source_id, output_port.port_id, target_id, input_port.port_id
                    );
                }
            }
        }
    }

    fn compute_route_plan(&mut self) -> Option<RoutePlan> {
        let Some(source_id) = self.openmeters_node_id else {
            self.clear_links();
            return None;
        };

        let Some(target_id) = self.default_sink.node_id else {
            self.clear_links();
            return None;
        };

        let source_ports = self.select_ports(source_id, "source", "output", |node| {
            node.output_ports_for_loopback()
        })?;

        let target_ports = self.select_ports(target_id, "target", "input", |node| {
            node.input_ports_for_loopback()
        })?;

        let pairs = pair_ports_by_channel(source_ports, target_ports);

        Some(RoutePlan {
            source_id,
            target_id,
            pairs,
        })
    }

    fn prune_stale_links(&mut self, desired_keys: &HashSet<LinkKey>) {
        let obsolete: Vec<LinkKey> = self
            .active_links
            .keys()
            .filter(|key| !desired_keys.contains(key))
            .copied()
            .collect();

        for key in obsolete {
            if let Some(link) = self.active_links.remove(&key) {
                drop(link);
            }
            println!(
                "[loopback] removed link {}:{} -> {}:{}",
                key.output_node, key.output_port, key.input_node, key.input_port
            );
        }
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
                println!(
                    "[loopback] {label} node {} missing; cleared active links",
                    node_id
                );
                return None;
            }
        };

        let ports = selector(node);
        if ports.is_empty() {
            let snapshot = node.clone_ports();
            self.clear_links();
            Self::log_ports_snapshot(label, node_id, &snapshot);
            println!(
                "[loopback] no {port_role} ports available on node {} for loopback",
                node_id
            );
            return None;
        }

        Some(ports)
    }

    fn clear_links(&mut self) {
        if self.active_links.is_empty() {
            return;
        }

        self.active_links.clear();
        println!("[loopback] cleared all active links");
    }

    fn log_ports_snapshot(label: &str, node_id: u32, ports: &[GraphPort]) {
        if ports.is_empty() {
            println!("[loopback] {label} node {} has no known ports", node_id);
            return;
        }

        println!(
            "[loopback] {label} node {} port inventory ({} ports):",
            node_id,
            ports.len()
        );
        for port in ports {
            println!("[loopback]   {}", format_port_debug(port));
        }
    }

    fn resolve_default_sink_node(&mut self) {
        if let Some(node_id) = self.default_sink.node_id {
            if self.nodes.contains_key(&node_id) {
                return;
            }
        }

        if let Some(name) = self.default_sink.name.clone() {
            if let Some(node_id) = self
                .nodes
                .iter()
                .find_map(|(&id, node)| node.matches_name(&name).then_some(id))
            {
                self.default_sink.node_id = Some(node_id);
            }
        }
    }

    fn set_openmeters_node(&mut self, node_id: Option<u32>) {
        if self.openmeters_node_id == node_id {
            return;
        }

        match (self.openmeters_node_id, node_id) {
            (_, Some(id)) => println!("[loopback] detected OpenMeters sink node #{id}"),
            (Some(_), None) => println!("[loopback] OpenMeters sink node removed"),
            (None, None) => {}
        }

        self.openmeters_node_id = node_id;
    }

    fn is_tracked_node(&self, node_id: u32) -> bool {
        self.openmeters_node_id == Some(node_id) || self.default_sink.node_id == Some(node_id)
    }
}

struct RoutePlan {
    source_id: u32,
    target_id: u32,
    pairs: Vec<(GraphPort, GraphPort)>,
}

impl RoutePlan {
    fn desired_keys(&self) -> HashSet<LinkKey> {
        self.pairs
            .iter()
            .map(|(out_port, in_port)| LinkKey {
                output_node: self.source_id,
                output_port: out_port.port_id,
                input_node: self.target_id,
                input_port: in_port.port_id,
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct LinkKey {
    output_node: u32,
    output_port: u32,
    input_node: u32,
    input_port: u32,
}

struct TrackedNode {
    info: Option<GraphNode>,
    ports: HashMap<u32, GraphPort>,
}

impl Default for TrackedNode {
    fn default() -> Self {
        Self {
            info: None,
            ports: HashMap::new(),
        }
    }
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
        let monitor_ports: Vec<_> = self
            .ports
            .values()
            .filter(|port| port.direction == PortDirection::Output && port.is_monitor)
            .cloned()
            .collect();

        if !monitor_ports.is_empty() {
            return monitor_ports;
        }

        let output_ports: Vec<_> = self
            .ports
            .values()
            .filter(|port| port.direction == PortDirection::Output)
            .cloned()
            .collect();
        if !output_ports.is_empty() {
            return output_ports;
        }

        self.ports.values().cloned().collect()
    }

    fn input_ports_for_loopback(&self) -> Vec<GraphPort> {
        let playback_ports: Vec<_> = self
            .ports
            .values()
            .filter(|port| port.direction == PortDirection::Input && !port.is_monitor)
            .cloned()
            .collect();

        if !playback_ports.is_empty() {
            return playback_ports;
        }

        let input_ports: Vec<_> = self
            .ports
            .values()
            .filter(|port| port.direction == PortDirection::Input)
            .cloned()
            .collect();
        if !input_ports.is_empty() {
            return input_ports;
        }

        self.ports.values().cloned().collect()
    }

    fn clone_ports(&self) -> Vec<GraphPort> {
        self.ports.values().cloned().collect()
    }
}
