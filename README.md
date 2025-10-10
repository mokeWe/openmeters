# OpenMeters

OpenMeters is an audio metering program for Linux, written in Rust for PipeWire systems.

## roadmap

### features

Checked features are implemented, unchecked features are planned.

- [x] Per-application capture
- [ ] Per-device capture
- [x] LUFS and peak metering
- [x] Oscilloscope
- [x] spectrogram
  - [x] Reassignment for better low-frequency resolution
- [x] spectrum analyzer
- [ ] stereometer
- [ ] waveform visualization
- [x] configurability
  - [x] persisted to disk
  - [ ] themes/color schemes

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

4. Build and run the application using Cargo:

   ```bash
   cargo run
   ```

5. If you encounter any issues or want to contribute in any way, please feel free to open an issue or a pull request on the GitHub repository.

## compatibility

OpenMeters is designed to work on Wayland systems running PipeWire. I cannot guarantee it will work on X11 at all. Due to the use of virtual sinks and how we route audio, OpenMeters may conflict with other software. This includes but is not limited to:

- EasyEffects
- Helvum
- Carla

I have chosen to use a virtual sink for audio capture because I personally wanted per-application monitoring, but capturing directly from a hardware sink's monitor _is_ planned, and shouldn't be too difficult to implement. If you desperately need this feature, please open an issue and I will prioritize it.

## credits

### inspiration

- **EasyEffects** (<https://github.com/wwmm/easyeffects>) for being a great source of inspiration and for their excellent work in audio processing. Reading through their codebase taught me a lot about PipeWire.
- **MiniMeters** (<https://minimeters.app/>) for inspiring this entire project and for doing it better than I ever could. If you can, please support their work!
- **Andura's Scrolloscope** (<https://github.com/ardura/Scrollscope>) for making a simple oscilloscope complex and open-source.
- **Tim Strasser's Oszilloskop** (<https://github.com/timstr/oszilloskop>)

### libraries used

- **Iced** (<https://github.com/iced-rs/iced>) for being an excellent GUI toolkit.
- **RustFFT** (<https://github.com/ejmahler/RustFFT>) for the FFT implementation.
- **RealFFT** (<https://github.com/HEnquist/realfft>) for the real FFT implementation.
- **wgpu** (<https://github.com/gfx-rs/wgpu>) for the GPU rendering backend.
