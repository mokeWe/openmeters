use pipewire as pw;
use pw::properties::properties;

/// PipeWire factory used to create link objects.
pub const LINK_FACTORY_NAME: &str = "link-factory";

/// Create a passive audio link between two ports.
///
/// The link keeps audio flowing without waking the target; this mirrors the
/// default behaviour of PipeWire when linking monitoring streams.
pub fn create_passive_audio_link(
    core: &pw::core::CoreRc,
    output_node: u32,
    output_port: u32,
    input_node: u32,
    input_port: u32,
) -> Result<pw::link::Link, pw::Error> {
    let props = properties! {
        *pw::keys::LINK_OUTPUT_NODE => output_node.to_string(),
        *pw::keys::LINK_OUTPUT_PORT => output_port.to_string(),
        *pw::keys::LINK_INPUT_NODE => input_node.to_string(),
        *pw::keys::LINK_INPUT_PORT => input_port.to_string(),
        *pw::keys::LINK_PASSIVE => "true",
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Playback",
        *pw::keys::MEDIA_ROLE => "Playback",
    };

    core.create_object::<pw::link::Link>(LINK_FACTORY_NAME, &props)
}
