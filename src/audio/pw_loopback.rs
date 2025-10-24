//! PipeWire loopback management for OpenMeters.

mod state;

use crate::util::pipewire::{DEFAULT_AUDIO_SINK_KEY, GraphNode, GraphPort};
use anyhow::{Context, Result};
use pipewire as pw;
use pw::metadata::{Metadata, MetadataListener};
use pw::registry::{GlobalObject, RegistryRc};
use pw::spa::utils::dict::DictRef;
use pw::types::ObjectType;
use state::LoopbackState;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{OnceLock, mpsc};
use std::thread;
use std::time::Duration;
use tracing::{error, info, warn};

const LOOPBACK_THREAD_NAME: &str = "openmeters-pw-loopback";
const OPENMETERS_SINK_NAME: &str = "openmeters.sink";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LoopbackMode {
    #[default]
    ForwardToDefaultSink,
    CaptureFromDevice(DeviceCaptureTarget),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeviceCaptureTarget {
    #[default]
    Default,
    Node(u32),
}

enum LoopbackCommand {
    SetMode(LoopbackMode),
}

static LOOPBACK_THREAD: OnceLock<thread::JoinHandle<()>> = OnceLock::new();
static LOOPBACK_COMMAND: OnceLock<mpsc::Sender<LoopbackCommand>> = OnceLock::new();

/// Start the loopback controller in a background thread if it is not already running.
pub fn run() {
    if LOOPBACK_THREAD.get().is_some() {
        return;
    }

    let (command_tx, command_rx) = mpsc::channel();

    match thread::Builder::new()
        .name(LOOPBACK_THREAD_NAME.into())
        .spawn(move || {
            if let Err(err) = run_loopback(command_rx) {
                error!("[loopback] stopped: {err:?}");
            }
        }) {
        Ok(handle) => {
            if LOOPBACK_COMMAND.set(command_tx).is_err() {
                warn!("[loopback] loopback command channel already initialised");
            }
            let _ = LOOPBACK_THREAD.set(handle);
        }
        Err(err) => error!("[loopback] failed to spawn thread: {err}"),
    }
}

pub fn set_mode(mode: LoopbackMode) -> bool {
    run();

    if let Some(sender) = LOOPBACK_COMMAND.get() {
        if sender.send(LoopbackCommand::SetMode(mode)).is_err() {
            warn!("[loopback] failed to dispatch mode change command");
            false
        } else {
            true
        }
    } else {
        warn!("[loopback] loopback thread not initialised; mode request dropped");
        false
    }
}

fn run_loopback(command_rx: mpsc::Receiver<LoopbackCommand>) -> Result<()> {
    pw::init();

    let mainloop =
        pw::main_loop::MainLoopRc::new(None).context("failed to create PipeWire main loop")?;
    let context = pw::context::ContextRc::new(&mainloop, None)
        .context("failed to create PipeWire context")?;
    let core = context
        .connect_rc(None)
        .context("failed to connect to PipeWire core")?;
    let registry = core
        .get_registry_rc()
        .context("failed to obtain PipeWire registry")?;

    let runtime = LoopbackRuntime::new(core.clone(), registry.clone());
    let runtime_for_added = runtime.clone();
    let runtime_for_removed = runtime.clone();

    let _registry_listener = registry
        .add_listener_local()
        .global(move |global| {
            runtime_for_added.handle_global_added(global);
        })
        .global_remove(move |id| {
            runtime_for_removed.handle_global_removed(id);
        })
        .register();

    info!("[loopback] PipeWire loopback thread running");
    let loop_ref = mainloop.loop_();
    let mut commands_disconnected = false;

    loop {
        if !commands_disconnected {
            loop {
                match command_rx.try_recv() {
                    Ok(command) => runtime.handle_command(command),
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        commands_disconnected = true;
                        break;
                    }
                }
            }
        }

        if loop_ref.iterate(Duration::from_millis(50)) < 0 {
            break;
        }
    }
    info!("[loopback] PipeWire loopback loop exited");

    Ok(())
}

#[derive(Clone)]
struct LoopbackRuntime {
    registry: RegistryRc,
    state: Rc<RefCell<LoopbackState>>,
    metadata_bindings: Rc<RefCell<HashMap<u32, (Metadata, MetadataListener)>>>,
}

impl LoopbackRuntime {
    fn new(core: pw::core::CoreRc, registry: RegistryRc) -> Self {
        Self {
            registry,
            state: Rc::new(RefCell::new(LoopbackState::new(core))),
            metadata_bindings: Rc::new(RefCell::new(HashMap::new())),
        }
    }

    fn handle_global_added(&self, global: &GlobalObject<&DictRef>) {
        match global.type_ {
            ObjectType::Node => {
                if let Some(node) = GraphNode::from_global(global) {
                    self.state.borrow_mut().upsert_node(node);
                }
            }
            ObjectType::Port => {
                if let Some(port) = GraphPort::from_global(global) {
                    self.state.borrow_mut().upsert_port(port);
                }
            }
            ObjectType::Metadata => {
                self.process_metadata_added(global);
            }
            _ => {}
        }
    }

    fn handle_global_removed(&self, id: u32) {
        let res = {
            let mut state = self.state.borrow_mut();
            state.remove_port_by_global(id) || state.remove_node(id)
        };
        if res {
            return;
        }

        if self.metadata_bindings.borrow_mut().remove(&id).is_some() {
            self.state.borrow_mut().clear_metadata(id);
        }
    }

    fn process_metadata_added(&self, global: &GlobalObject<&DictRef>) {
        let metadata_id = global.id;
        let mut bindings = self.metadata_bindings.borrow_mut();
        if bindings.contains_key(&metadata_id) {
            return;
        }

        let metadata = match self.registry.bind::<Metadata, _>(global) {
            Ok(metadata) => metadata,
            Err(err) => {
                error!("[loopback] failed to bind metadata {metadata_id}: {err}");
                return;
            }
        };

        let runtime = self.clone();
        let metadata_listener = metadata
            .add_listener_local()
            .property(move |subject, key, type_hint, value| {
                runtime.handle_metadata_property(metadata_id, subject, key, type_hint, value);
                0
            })
            .register();

        bindings.insert(metadata_id, (metadata, metadata_listener));
    }

    fn handle_metadata_property(
        &self,
        metadata_id: u32,
        subject: u32,
        key: Option<&str>,
        type_hint: Option<&str>,
        value: Option<&str>,
    ) {
        if key != Some(DEFAULT_AUDIO_SINK_KEY) {
            return;
        }

        self.state
            .borrow_mut()
            .update_default_sink(metadata_id, subject, type_hint, value);
    }

    fn handle_command(&self, command: LoopbackCommand) {
        match command {
            LoopbackCommand::SetMode(mode) => self.state.borrow_mut().set_mode(mode),
        }
    }
}
