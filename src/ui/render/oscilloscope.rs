use iced::Rectangle;
use iced::advanced::graphics::Viewport;
use iced_wgpu::primitive::{Primitive, Storage};
use iced_wgpu::wgpu;

use crate::ui::render::common::{ClipTransform, InstanceBuffer, SdfVertex, create_sdf_pipeline};
use crate::ui::render::geometry::{self, DEFAULT_FEATHER, append_strip};

#[derive(Debug, Clone)]
pub struct OscilloscopeParams {
    pub bounds: Rectangle,
    pub channels: usize,
    pub samples_per_channel: usize,
    pub samples: Vec<f32>,
    pub colors: Vec<[f32; 4]>,
    pub fill_alpha: f32,
}

#[derive(Debug)]
pub struct OscilloscopePrimitive {
    params: OscilloscopeParams,
}

impl OscilloscopePrimitive {
    pub fn new(params: OscilloscopeParams) -> Self {
        Self { params }
    }

    fn build_vertices(&self, viewport: &Viewport) -> Vec<SdfVertex> {
        let samples_per_channel = self.params.samples_per_channel;
        let channels = self.params.channels.max(1);

        if samples_per_channel < 2 || self.params.samples.len() < channels * samples_per_channel {
            return Vec::new();
        }

        let bounds = self.params.bounds;
        let clip = ClipTransform::from_viewport(viewport);

        const VERTICAL_PADDING: f32 = 8.0;
        const CHANNEL_GAP: f32 = 12.0;
        const AMPLITUDE_SCALE: f32 = 0.9;
        const STROKE_WIDTH: f32 = 1.0;
        const LINE_ALPHA: f32 = 0.92;

        let usable_height = (bounds.height
            - VERTICAL_PADDING * 2.0
            - CHANNEL_GAP * (channels.saturating_sub(1) as f32))
            .max(1.0);
        let channel_height = usable_height / channels as f32;
        let amplitude_scale = channel_height * 0.5 * AMPLITUDE_SCALE;
        let step = bounds.width.max(1.0) / (samples_per_channel.saturating_sub(1) as f32).max(1.0);

        let mut vertices = Vec::new();

        for (channel_idx, channel_samples) in self
            .params
            .samples
            .chunks_exact(samples_per_channel)
            .take(channels)
            .enumerate()
        {
            let color = self
                .params
                .colors
                .get(channel_idx)
                .copied()
                .unwrap_or([0.6, 0.8, 0.9, 1.0]);
            let center = bounds.y
                + VERTICAL_PADDING
                + channel_idx as f32 * (channel_height + CHANNEL_GAP)
                + channel_height * 0.5;

            let positions: Vec<_> = channel_samples
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    (
                        bounds.x + i as f32 * step,
                        center - s.clamp(-1.0, 1.0) * amplitude_scale,
                    )
                })
                .collect();

            // Build fill vertices (solid, no AA needed)
            let mut fill_vertices = Vec::with_capacity(positions.len() * 2);
            let fill_color = [color[0], color[1], color[2], self.params.fill_alpha];

            for &(x, y) in &positions {
                let above_zero = y < center;
                let fill_to_y = if above_zero { y } else { center };
                let fill_from_y = if above_zero { center } else { y };

                fill_vertices.push(SdfVertex::solid(clip.to_clip(x, fill_to_y), fill_color));
                fill_vertices.push(SdfVertex::solid(clip.to_clip(x, fill_from_y), fill_color));
            }

            append_strip(&mut vertices, fill_vertices);

            // Build antialiased line strip using geometry helper
            let line_color = [color[0], color[1], color[2], LINE_ALPHA];
            let line_strip = geometry::build_aa_line_strip(
                &positions,
                STROKE_WIDTH,
                DEFAULT_FEATHER,
                line_color,
                &clip,
            );
            append_strip(&mut vertices, line_strip);
        }

        vertices
    }
}

impl Primitive for OscilloscopePrimitive {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        storage: &mut Storage,
        _bounds: &Rectangle,
        viewport: &Viewport,
    ) {
        if !storage.has::<Pipeline>() {
            storage.store(Pipeline::new(device, format));
        }

        let pipeline = storage
            .get_mut::<Pipeline>()
            .expect("pipeline must exist after storage check");

        let vertices = self.build_vertices(viewport);
        pipeline.prepare(device, queue, &vertices);
    }

    fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        storage: &Storage,
        target: &wgpu::TextureView,
        clip_bounds: &Rectangle<u32>,
    ) {
        let Some(pipeline) = storage.get::<Pipeline>() else {
            return;
        };

        if pipeline.buffer.vertex_count == 0 {
            return;
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Oscilloscope pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_scissor_rect(
            clip_bounds.x,
            clip_bounds.y,
            clip_bounds.width.max(1),
            clip_bounds.height.max(1),
        );
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_vertex_buffer(
            0,
            pipeline
                .buffer
                .vertex_buffer
                .slice(0..pipeline.buffer.used_bytes()),
        );
        pass.draw(0..pipeline.buffer.vertex_count, 0..1);
    }
}

#[derive(Debug)]
struct Pipeline {
    pipeline: wgpu::RenderPipeline,
    buffer: InstanceBuffer<SdfVertex>,
}

impl Pipeline {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        Self {
            pipeline: create_sdf_pipeline(
                device,
                format,
                "Oscilloscope",
                wgpu::PrimitiveTopology::TriangleStrip,
            ),
            buffer: InstanceBuffer::new(device, "Oscilloscope vertex buffer", 1024),
        }
    }

    fn prepare(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, vertices: &[SdfVertex]) {
        if vertices.is_empty() {
            self.buffer.vertex_count = 0;
            return;
        }

        let required_size = std::mem::size_of_val(vertices) as wgpu::BufferAddress;
        self.buffer
            .ensure_capacity(device, "Oscilloscope vertex buffer", required_size);
        self.buffer.write(queue, vertices);
    }
}
