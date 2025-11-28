//! PipeWire link management.
//!
//! This module manages audio links between PipeWire nodes. It receives explicit
//! link requests from the registry monitor and maintains the active links.

use crate::util::pipewire::create_passive_audio_link;
use anyhow::{Context, Result};
use pipewire as pw;
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::{OnceLock, mpsc};
use std::thread;
use std::time::Duration;
use tracing::{debug, error, info, warn};

const LOOPBACK_THREAD_NAME: &str = "openmeters-pw-loopback";

/// A single port-to-port link specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LinkSpec {
    pub output_node: u32,
    pub output_port: u32,
    pub input_node: u32,
    pub input_port: u32,
}

/// Command sent to the link manager thread.
#[derive(Debug, Clone)]
enum LinkCommand {
    /// Set the desired links. Any links not in this set will be removed.
    SetLinks(Vec<LinkSpec>),
}

static LOOPBACK_THREAD: OnceLock<thread::JoinHandle<()>> = OnceLock::new();
static LOOPBACK_COMMAND: OnceLock<mpsc::Sender<LinkCommand>> = OnceLock::new();

/// Start the link manager in a background thread if not already running.
pub fn run() {
    if LOOPBACK_THREAD.get().is_some() {
        return;
    }

    let (command_tx, command_rx) = mpsc::channel();

    match thread::Builder::new()
        .name(LOOPBACK_THREAD_NAME.into())
        .spawn(move || {
            if let Err(err) = run_link_manager(command_rx) {
                error!("[loopback] stopped: {err:?}");
            }
        }) {
        Ok(handle) => {
            if LOOPBACK_COMMAND.set(command_tx).is_err() {
                warn!("[loopback] link command channel already initialised");
            }
            let _ = LOOPBACK_THREAD.set(handle);
        }
        Err(err) => error!("[loopback] failed to spawn thread: {err}"),
    }
}

/// Send a link command to the manager thread.
fn send_command(command: LinkCommand) -> bool {
    run();

    if let Some(sender) = LOOPBACK_COMMAND.get() {
        if sender.send(command).is_err() {
            warn!("[loopback] failed to dispatch link command");
            false
        } else {
            true
        }
    } else {
        warn!("[loopback] link manager thread not initialised");
        false
    }
}

/// Convenience: set the desired links.
pub fn set_links(links: Vec<LinkSpec>) -> bool {
    send_command(LinkCommand::SetLinks(links))
}

fn run_link_manager(command_rx: mpsc::Receiver<LinkCommand>) -> Result<()> {
    pw::init();

    let mainloop =
        pw::main_loop::MainLoopRc::new(None).context("failed to create PipeWire main loop")?;
    let context = pw::context::ContextRc::new(&mainloop, None)
        .context("failed to create PipeWire context")?;
    let core = context
        .connect_rc(None)
        .context("failed to connect to PipeWire core")?;

    let mut state = LinkManagerState::new(core);

    info!("[loopback] PipeWire link manager running");
    let loop_ref = mainloop.loop_();
    let mut commands_disconnected = false;

    loop {
        if !commands_disconnected {
            loop {
                match command_rx.try_recv() {
                    Ok(command) => state.handle_command(command),
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

    info!("[loopback] PipeWire link manager exited");
    Ok(())
}

struct LinkManagerState {
    core: pw::core::CoreRc,
    active_links: FxHashMap<LinkSpec, pw::link::Link>,
}

impl LinkManagerState {
    fn new(core: pw::core::CoreRc) -> Self {
        Self {
            core,
            active_links: FxHashMap::default(),
        }
    }

    fn handle_command(&mut self, command: LinkCommand) {
        match command {
            LinkCommand::SetLinks(desired) => self.apply_links(desired),
        }
    }

    fn apply_links(&mut self, desired: Vec<LinkSpec>) {
        let desired_set: FxHashSet<LinkSpec> = desired.iter().copied().collect();

        // Remove links that are no longer desired
        self.active_links.retain(|spec, _| {
            let keep = desired_set.contains(spec);
            if !keep {
                debug!(
                    "[loopback] removed link {}:{} -> {}:{}",
                    spec.output_node, spec.output_port, spec.input_node, spec.input_port
                );
            }
            keep
        });

        // Create new links
        for spec in desired {
            if self.active_links.contains_key(&spec) {
                continue;
            }

            match create_passive_audio_link(
                &self.core,
                spec.output_node,
                spec.output_port,
                spec.input_node,
                spec.input_port,
            ) {
                Ok(link) => {
                    debug!(
                        "[loopback] linked {}:{} -> {}:{}",
                        spec.output_node, spec.output_port, spec.input_node, spec.input_port
                    );
                    self.active_links.insert(spec, link);
                }
                Err(err) => {
                    error!(
                        "[loopback] link failed {}:{} -> {}:{}: {err}",
                        spec.output_node, spec.output_port, spec.input_node, spec.input_port
                    );
                }
            }
        }
    }
}
