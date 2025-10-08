# OpenMeters

OpenMeters is an audio metering program for Linux, written in Rust for PipeWire systems.

## roadmap

### features

- [x] Per-application capture
- [ ] Per-device capture
- [x] LUFS and peak metering
- [x] Oscilloscope
- [x] spectrogram
- [ ] spectrum analyzer
- [ ] stereometer
- [ ] waveform visualization

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
