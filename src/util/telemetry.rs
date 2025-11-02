use std::sync::OnceLock;
use tracing::Level;
use tracing_subscriber::{EnvFilter, fmt};

static TELEMETRY_INIT: OnceLock<()> = OnceLock::new();

pub fn init() {
    TELEMETRY_INIT.get_or_init(|| {
        let env_filter = EnvFilter::try_from_default_env()
            .or_else(|_| EnvFilter::try_new("openmeters=info"))
            .unwrap_or_else(|_| EnvFilter::default().add_directive(Level::INFO.into()));

        if let Err(err) = fmt()
            .with_env_filter(env_filter)
            .with_target(false)
            .compact()
            .try_init()
        {
            eprintln!("[telemetry] failed to initialise tracing subscriber: {err}");
        }
    });
}
