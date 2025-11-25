# OpenMeters

![preview](screenshots/preview.gif)

OpenMeters is an audio metering program for Linux, written in Rust for PipeWire systems.

## roadmap

### features

Checked features are implemented, unchecked features are planned. If you have ideas for more features, please feel free to open an issue/pull request!

- [x] Per-application capture
- [x] Per-device capture
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
  - [ ] Dot cloud
- [x] **waveform**
  - [x] Adjustable colormap

## build and run

1. Ensure you have a working Rust toolchain. The recommended way is via [rustup](https://rustup.rs/).
2. Install the required system dependencies. On Arch-based systems, for example, you can run:

   ```bash
   sudo pacman -S pipewire libpipewire 
   ```

3. Clone the repository:

   ```bash
   git clone https://github.com/mokeWe/openmeters/
   cd openmeters
   ```

4. Build and run the application:

   ```bash
   cargo build -r
   ./target/release/openmeters
   ```

   or run it directly with Cargo:

   ```bash
   cargo run
   ```

5. If you encounter any issues or want to contribute in any way, please feel free to open an issue or a pull request.

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

### inspiration

- **EasyEffects** (<https://github.com/wwmm/easyeffects>) for being a great source of inspiration and for their excellent work in audio processing. Reading through their codebase taught me a lot about PipeWire.
- **MiniMeters** (<https://minimeters.app/>) for inspiring this entire project and for doing it better than I ever could. If you can, please support their work!
- **Ardura's Scrolloscope** (<https://github.com/ardura/Scrollscope>) for making a simple oscilloscope complex and open-source.
- **Tim Strasser's Oszilloskop** (<https://github.com/timstr/oszilloskop>)

### libraries used

- **Iced** (<https://github.com/iced-rs/iced>) for being an excellent GUI toolkit.
- **RustFFT** (<https://github.com/ejmahler/RustFFT>) for the FFT implementation.
- **RealFFT** (<https://github.com/HEnquist/realfft>) for the real FFT implementation.
- **wgpu** (<https://github.com/gfx-rs/wgpu>) for the GPU rendering backend.
