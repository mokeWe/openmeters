use serde_json::Value;
use tracing::warn;

/// Metadata key identifying the default audio sink in PipeWire.
pub const DEFAULT_AUDIO_SINK_KEY: &str = "default.audio.sink";
/// Metadata key identifying the default audio source in PipeWire.
pub const DEFAULT_AUDIO_SOURCE_KEY: &str = "default.audio.source";

/// Shared representation of a default PipeWire target as reported via metadata.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DefaultTarget {
    pub metadata_id: Option<u32>,
    pub node_id: Option<u32>,
    pub name: Option<String>,
    pub type_hint: Option<String>,
}

impl DefaultTarget {
    /// Apply a metadata update to this target, returning `true` when a field changed.
    pub fn update(
        &mut self,
        metadata_id: u32,
        subject: u32,
        type_hint: Option<&str>,
        name: Option<&str>,
    ) -> bool {
        let mut changed = false;

        if self.metadata_id != Some(metadata_id) {
            self.metadata_id = Some(metadata_id);
            changed = true;
        }

        let new_node_id = if subject != 0 { Some(subject) } else { None };
        if self.node_id != new_node_id {
            self.node_id = new_node_id;
            changed = true;
        }

        if self.type_hint.as_deref() != type_hint {
            self.type_hint = type_hint.map(|hint| hint.to_string());
            changed = true;
        }

        if self.name.as_deref() != name {
            self.name = name.map(|val| val.to_string());
            changed = true;
        }

        changed
    }

    /// Reset the stored metadata to its default, empty state.
    pub fn clear(&mut self) {
        *self = Default::default();
    }
}

/// Parse a PipeWire metadata value representing a default device target name.
pub fn parse_metadata_name(type_hint: Option<&str>, value: Option<&str>) -> Option<String> {
    let value = value?;
    let trimmed = value.trim();

    let expects_json = matches!(type_hint, Some(hint) if hint.eq_ignore_ascii_case("Spa:String:JSON"))
        || trimmed.starts_with('{');

    if expects_json {
        match serde_json::from_str::<Value>(trimmed) {
            Ok(Value::Object(map)) => map
                .get("name")
                .and_then(|name| name.as_str().map(|s| s.to_string())),
            Ok(Value::String(name)) => Some(name),
            Ok(_) => None,
            Err(err) => {
                warn!(
                    "[loopback] failed to parse default sink metadata JSON: {err} (value={value})"
                );
                None
            }
        }
    } else if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Encoded values used to route a node towards a specific target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetMetadataPayload {
    pub type_hint: &'static str,
    pub target_object: String,
    pub target_node: String,
    pub target_name: String,
}

impl TargetMetadataPayload {
    fn new(type_hint: &'static str, object: String, node: String, name: String) -> Self {
        Self {
            type_hint,
            target_object: object,
            target_node: node,
            target_name: name,
        }
    }
}

/// Encode metadata payload describing a PipeWire target node.
pub fn format_target_metadata(
    object_serial: Option<&str>,
    node_id: u32,
    node_name: &str,
) -> TargetMetadataPayload {
    let sanitized_name = node_name.trim().to_string();
    let object_value = object_serial
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.to_string())
        .unwrap_or_else(|| node_id.to_string());

    TargetMetadataPayload::new("Spa:Id", object_value, node_id.to_string(), sanitized_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_payload_prefers_object_serial_when_available() {
        let payload = format_target_metadata(Some("102"), 7, "speakers");
        assert_eq!(payload.type_hint, "Spa:Id");
        assert_eq!(payload.target_object, "102");
        assert_eq!(payload.target_node, "7");
        assert_eq!(payload.target_name, "speakers");
    }

    #[test]
    fn metadata_payload_falls_back_to_node_id() {
        let payload = format_target_metadata(None, 9, "  sink  ");
        assert_eq!(payload.target_object, "9");
        assert_eq!(payload.target_node, "9");
        assert_eq!(payload.target_name, "sink");
    }
}
