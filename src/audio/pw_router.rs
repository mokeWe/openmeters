//! PipeWire routing helper for directing applications to specific sinks.

use super::pw_registry::NodeInfo;
use crate::util::{dict_to_map, pipewire::metadata::format_target_metadata};
use anyhow::{Context, Result, bail};
use pipewire as pw;
use pw::metadata::Metadata;
use pw::proxy::{ProxyListener, ProxyT};
use pw::registry::RegistryRc;
use pw::types::ObjectType;
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::time::{Duration, Instant};
use tracing::{info, warn};

/// Metadata key instructing PipeWire where to route a node by object serial.
const TARGET_OBJECT_KEY: &str = "target.object";
/// Metadata key instructing PipeWire where to route a node by numeric node id.
const TARGET_NODE_KEY: &str = "target.node";
/// Metadata objects are probed in this preference order when multiple exist.
const PREFERRED_METADATA_NAMES: &[&str] = &["settings", "default"];
/// How long the router waits for metadata discovery before giving up.
const METADATA_DISCOVERY_TIMEOUT: Duration = Duration::from_secs(2);
/// Grace period after binding an acceptable metadata object before proceeding.
const FALLBACK_WAIT_GRACE: Duration = Duration::from_millis(250);

/// Helper that issues metadata updates to move applications between sinks.
pub struct Router {
    mainloop: pw::main_loop::MainLoopRc,
    _core: pw::core::CoreRc,
    metadata: Metadata,
    _metadata_listener: ProxyListener,
    metadata_failed: Rc<Cell<bool>>,
}

impl Router {
    /// Establish a dedicated PipeWire connection capable of issuing routing commands.
    ///
    /// A dedicated main loop is created so routing operations do not interfere with
    /// the application's primary PipeWire context. The router binds a metadata
    /// object once during initialisation and reuses it for all subsequent updates.
    pub fn new() -> Result<Self> {
        pw::init();

        let mainloop = pw::main_loop::MainLoopRc::new(None)
            .context("failed to create PipeWire main loop for router")?;
        let context = pw::context::ContextRc::new(&mainloop, None)
            .context("failed to create PipeWire context for router")?;
        let core = context
            .connect_rc(None)
            .context("failed to connect to PipeWire core for router")?;
        let registry = core
            .get_registry_rc()
            .context("failed to obtain PipeWire registry for router")?;

        core.sync(0)
            .context("failed to synchronise with PipeWire core while initialising router")?;

        let (metadata, metadata_name) =
            select_metadata(&mainloop, &registry).context("failed to bind PipeWire metadata")?;

        info!(
            "[router] using metadata '{}'",
            metadata_name.as_deref().unwrap_or("unnamed")
        );

        let metadata_failed = Rc::new(Cell::new(false));
        let listener = metadata
            .upcast_ref()
            .add_listener_local()
            .destroy({
                let flag = metadata_failed.clone();
                move || {
                    if !flag.replace(true) {
                        warn!("[router] metadata proxy destroyed; routing unavailable until reconnection");
                    }
                }
            })
            .removed({
                let flag = metadata_failed.clone();
                move || {
                    if !flag.replace(true) {
                        warn!("[router] metadata proxy removed; routing unavailable until reconnection");
                    }
                }
            })
            .error({
                let flag = metadata_failed.clone();
                move |seq, res, message| {
                    if !flag.replace(true) {
                        warn!(
                            "[router] metadata proxy error (seq={seq}, res={res}): {message}; routing unavailable"
                        );
                    }
                }
            })
            .register();

        Ok(Self {
            mainloop,
            _core: core,
            metadata,
            _metadata_listener: listener,
            metadata_failed,
        })
    }

    /// Route the provided application node to the supplied sink using descriptive metadata.
    pub fn route_application_to_sink(&self, application: &NodeInfo, sink: &NodeInfo) -> Result<()> {
        self.ensure_metadata_alive()?;

        let subject = application.id;

        let preferred_label = sink.display_name();

        let metadata =
            format_target_metadata(sink.object_serial(), sink.id, preferred_label.as_str());

        self.set_metadata_property(
            subject,
            TARGET_OBJECT_KEY,
            metadata.type_hint,
            &metadata.target_object,
        )?;
        self.pump_loop(1);
        self.ensure_metadata_alive()?;

        self.set_metadata_property(
            subject,
            TARGET_NODE_KEY,
            metadata.type_hint,
            &metadata.target_node,
        )?;
        self.pump_loop(2);

        self.ensure_metadata_alive()?;

        Ok(())
    }

    fn set_metadata_property(
        &self,
        subject: u32,
        key: &str,
        type_hint: &'static str,
        value: &str,
    ) -> Result<()> {
        self.metadata
            .set_property(subject, key, Some(type_hint), Some(value));
        Ok(())
    }

    /// Service the router's main loop for a short burst to flush pending events.
    fn pump_loop(&self, iterations: usize) {
        let loop_ref = self.mainloop.loop_();
        for _ in 0..iterations {
            loop_ref.iterate(Duration::from_millis(10));
        }
    }

    fn ensure_metadata_alive(&self) -> Result<()> {
        if self.metadata_failed.get() {
            bail!("PipeWire metadata proxy is unavailable");
        }
        Ok(())
    }

    pub fn is_healthy(&self) -> bool {
        !self.metadata_failed.get()
    }
}

/// Select a metadata object from the registry, favouring well-known names when possible.
fn select_metadata(
    mainloop: &pw::main_loop::MainLoopRc,
    registry: &RegistryRc,
) -> Result<(Metadata, Option<String>)> {
    let selected_metadata: Rc<RefCell<Option<Metadata>>> = Rc::new(RefCell::new(None));
    let selected_name: Rc<RefCell<Option<String>>> = Rc::new(RefCell::new(None));
    let done = Rc::new(Cell::new(false));

    let registry_for_cb = registry.clone();
    let result_for_cb = Rc::clone(&selected_metadata);
    let name_for_cb = Rc::clone(&selected_name);
    let done_for_cb = Rc::clone(&done);

    let listener = registry
        .add_listener_local()
        .global(move |global| {
            if global.type_ != ObjectType::Metadata {
                return;
            }

            let props = dict_to_map(global.props.as_ref().copied());
            let metadata_name = props.get("metadata.name").cloned();

            match registry_for_cb.bind::<Metadata, _>(global) {
                Ok(metadata) => {
                    let is_preferred = metadata_name
                        .as_deref()
                        .map(|candidate| {
                            PREFERRED_METADATA_NAMES
                                .iter()
                                .any(|preferred| preferred.eq_ignore_ascii_case(candidate))
                        })
                        .unwrap_or(false);

                    if is_preferred {
                        *result_for_cb.borrow_mut() = Some(metadata);
                        *name_for_cb.borrow_mut() = metadata_name;
                        done_for_cb.set(true);
                    } else if result_for_cb.borrow().is_none() {
                        *result_for_cb.borrow_mut() = Some(metadata);
                        *name_for_cb.borrow_mut() = metadata_name;
                    }
                }
                Err(err) => {
                    warn!("[router] failed to bind metadata {}: {err}", global.id);
                }
            }
        })
        .register();

    let loop_ref = mainloop.loop_();
    let overall_deadline = Instant::now() + METADATA_DISCOVERY_TIMEOUT;
    let mut fallback_deadline: Option<Instant> = None;

    while !done.get() && Instant::now() < overall_deadline {
        loop_ref.iterate(Duration::from_millis(50));

        if done.get() {
            break;
        }

        if selected_metadata.borrow().is_some() && fallback_deadline.is_none() {
            fallback_deadline = Some(Instant::now() + FALLBACK_WAIT_GRACE);
        }

        if let Some(limit) = fallback_deadline
            && Instant::now() >= limit
        {
            break;
        }
    }

    drop(listener);

    let Some(metadata) = selected_metadata.borrow_mut().take() else {
        bail!("no PipeWire metadata objects discovered while initialising router");
    };

    let metadata_name = selected_name.borrow().clone();

    Ok((metadata, metadata_name))
}
