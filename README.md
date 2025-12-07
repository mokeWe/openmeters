# OpenMeters

![preview](screenshots/preview.gif)

<small>Above is an early preview and thus may not represent the current state of the application. Build it yourself to see where it currently stands.</small>

OpenMeters is a fast and simple audio metering application for Linux, built with Rust and PipeWire.

## roadmap

### features

Checked features are implemented, unchecked features are planned. If you have ideas for more features, please feel free to open an issue/pull request!

#### general

- [x] Per-application capture
- [x] Per-device capture

#### visuals

- [x] **loudness**
  - [x] LUFS (ITU-R BS.1770-5)
    - [x] Short-term
    - [x] Momentary
  - [x] RMS
    - [x] Fast
    - [x] Slow
  - [x] True Peak
- [x] **oscilloscope**
  - [x] Stable mode - Follows X cycles of the fundamental.
  - [x] Free-run mode - Scrolls continuously through time, not triggered.
- [x] **spectrogram**
  - [x] Reassignment and synchrosqueezing (sharper frequency resolution)
  - [x] Note & frequency tooltips
  - [x] mel, log, and linear scales
  - [x] Adjustable colormap
- [x] **spectrum analyzer**
  - [x] peak frequency label
  - [x] averaging modes
    - [x] exponential
    - [x] peak hold
    - [x] none
  - [x] mel, log, and linear scales
  - [x] Adjustable colormap
- [x] **stereometer**
  - [ ] Correlation meter
  - [x] Vectorscope/XY mode
  - [x] 'Dot cloud' (Mid/Side goniometer) mode
- [x] **waveform**
  - [x] Adjustable colormap

## build and run

1. Ensure you have a working Rust toolchain. The recommended way is via [rustup](https://rustup.rs/).
2. Clone the repository:

   ```bash
   git clone https://github.com/httpsworldview/openmeters/
   cd openmeters
   ```

3. Build and run the application:

   ```bash
   cargo build -r
   ./target/release/openmeters
   ```

   or run it directly with Cargo:

   ```bash
   cargo run
   ```

## usage

Upon launch you'll see a configuration page. Here you can select which applications or devices to monitor, the modules you want to display, and some global settings (window decorations, global background color, etc.). By default, we monitor applications.
  
### keybinds

- `ctrl+shift+h`: Show/hide top bar
- `p`: Pause/resume audio capture
- `q` twice: Quit application

### configuration

Configurations are saved to `~/.config/openmeters/settings.json`. If you want to use settings/values not listed in the GUI, you can edit this file directly, however absurd values often lead to crashes. Invalid values will be replaced with defaults on load, or when saved.
If you encounter a bug that causes OpenMeters to misbehave, you can delete `settings.json` to reset everything, or change the problematic setting manually.

## credits

Thank *you* for checking out my shitty passion project. If you think OpenMeters is useful, please consider starring the repository and sharing it with others. I appreciate any and all criticism and feedback, so feel free to open issues or reach out to me.

### inspiration

- **EasyEffects** (<https://github.com/wwmm/easyeffects>) for being a great source of inspiration and for their excellent work in audio processing. Reading through their codebase taught me a lot about PipeWire.
- **MiniMeters** (<https://minimeters.app/>) for inspiring this entire project and for doing it better than I ever could. If you can, please support their work!
- **Ardura's Scrolloscope** (<https://github.com/ardura/Scrollscope>)
- **Tim Strasser's Oszilloskop** (<https://github.com/timstr/oszilloskop>)
- **Audacity** (<https://www.audacityteam.org/>)

### libraries used

- **Iced** (<https://github.com/iced-rs/iced>)
- **RustFFT** (<https://github.com/ejmahler/RustFFT>)
- **RealFFT** (<https://github.com/HEnquist/realfft>)
- **wgpu** (<https://github.com/gfx-rs/wgpu>)
