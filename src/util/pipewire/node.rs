use pipewire as pw;
use std::collections::HashMap;

/// General direction of a node (input/output/unknown) inferred from metadata.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NodeDirection {
    Input,
    Output,
    #[default]
    Unknown,
}

/// Determine the direction of a node by inspecting its media class and port hints.
pub fn derive_node_direction(
    media_class: Option<&str>,
    props: &HashMap<String, String>,
) -> NodeDirection {
    if let Some(class) = media_class {
        let lowered = class.to_ascii_lowercase();
        if lowered.contains("sink") || lowered.contains("output") {
            return NodeDirection::Output;
        }
        if lowered.contains("source") || lowered.contains("input") {
            return NodeDirection::Input;
        }
    }

    if let Some(direction) = props.get(*pw::keys::PORT_DIRECTION) {
        match direction.to_ascii_lowercase().as_str() {
            "in" => return NodeDirection::Input,
            "out" => return NodeDirection::Output,
            _ => {}
        }
    }

    NodeDirection::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_direction_prefers_media_class_keywords() {
        let props = HashMap::new();
        assert_eq!(
            derive_node_direction(Some("Audio/Sink"), &props),
            NodeDirection::Output
        );
        assert_eq!(
            derive_node_direction(Some("Stream/Input/Audio"), &props),
            NodeDirection::Input
        );
    }
}
