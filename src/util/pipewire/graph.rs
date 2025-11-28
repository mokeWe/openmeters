use crate::util::pipewire::dict::dict_to_map;
use pipewire as pw;
use pw::registry::GlobalObject;
use pw::spa::utils::dict::DictRef;
use std::collections::{HashMap, VecDeque};
use tracing::warn;

/// Direction of a PipeWire port.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PortDirection {
    Input,
    Output,
    Unknown,
}

impl PortDirection {
    pub fn from_str(value: &str) -> Self {
        match value.to_ascii_lowercase().as_str() {
            "in" => PortDirection::Input,
            "out" => PortDirection::Output,
            _ => PortDirection::Unknown,
        }
    }
}

/// Representation of a port exposed by a PipeWire node.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphPort {
    pub global_id: u32,
    pub port_id: u32,
    pub node_id: u32,
    pub channel: Option<String>,
    pub direction: PortDirection,
    pub is_monitor: bool,
}

impl GraphPort {
    pub fn from_global(global: &GlobalObject<&DictRef>) -> Option<Self> {
        let props = dict_to_map(global.props.as_ref().copied());

        let port_id_str = props
            .get(*pw::keys::PORT_ID)
            .or_else(|| props.get("port.id"));

        let port_id = match port_id_str.and_then(|value| value.parse::<u32>().ok()) {
            Some(id) => id,
            None => {
                warn!(
                    "[loopback] skipping port global {}: missing or invalid port.id (props keys: {:?})",
                    global.id,
                    props.keys().collect::<Vec<_>>()
                );
                return None;
            }
        };

        let node_id_str = props
            .get(*pw::keys::NODE_ID)
            .or_else(|| props.get("node.id"));

        let node_id = match node_id_str.and_then(|value| value.parse::<u32>().ok()) {
            Some(id) => id,
            None => {
                warn!(
                    "[loopback] skipping port global {}: missing or invalid node.id (props keys: {:?})",
                    global.id,
                    props.keys().collect::<Vec<_>>()
                );
                return None;
            }
        };

        let direction = props
            .get(*pw::keys::PORT_DIRECTION)
            .or_else(|| props.get("port.direction"))
            .map(|dir| PortDirection::from_str(dir))
            .unwrap_or(PortDirection::Unknown);

        let channel = props
            .get(*pw::keys::AUDIO_CHANNEL)
            .or_else(|| props.get("audio.channel"))
            .cloned();

        let is_monitor = props
            .get(*pw::keys::PORT_MONITOR)
            .or_else(|| props.get("port.monitor"))
            .map(|value| matches!(value.to_ascii_lowercase().as_str(), "true" | "1"))
            .unwrap_or(false);

        Some(Self {
            global_id: global.id,
            port_id,
            node_id,
            channel,
            direction,
            is_monitor,
        })
    }

    #[inline]
    pub fn channel_key(&self) -> Option<&str> {
        self.channel.as_deref()
    }
}

/// Normalise an audio channel name for comparisons.
pub fn normalize_channel_name(value: &str) -> String {
    value.trim().to_ascii_uppercase()
}

/// Pair output and input ports by channel, falling back to positional matching.
pub fn pair_ports_by_channel(
    mut sources: Vec<GraphPort>,
    targets: Vec<GraphPort>,
) -> Vec<(GraphPort, GraphPort)> {
    let mut targets_by_channel: HashMap<String, VecDeque<GraphPort>> = HashMap::new();
    let mut fallback_targets: VecDeque<GraphPort> = VecDeque::new();

    for target in targets {
        if let Some(channel) = target.channel_key() {
            targets_by_channel
                .entry(normalize_channel_name(channel))
                .or_default()
                .push_back(target);
        } else {
            fallback_targets.push_back(target);
        }
    }

    sources.sort_by(|a, b| {
        let a_key = (a.channel.as_deref().unwrap_or(""), a.port_id);
        let b_key = (b.channel.as_deref().unwrap_or(""), b.port_id);
        a_key.cmp(&b_key)
    });

    let mut plans = Vec::new();

    for source in sources {
        let mut candidate = None;

        if let Some(channel) = source.channel_key()
            && let Some(queue) = targets_by_channel.get_mut(&normalize_channel_name(channel))
        {
            candidate = queue.pop_front();
        }

        if candidate.is_none() {
            candidate = fallback_targets.pop_front();
        }

        if candidate.is_none()
            && let Some(queue) = targets_by_channel
                .values_mut()
                .find(|queue| !queue.is_empty())
        {
            candidate = queue.pop_front();
        }

        if let Some(target) = candidate {
            plans.push((source.clone(), target));
        }
    }

    plans
}
