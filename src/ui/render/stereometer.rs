use bytemuck::{Pod, Zeroable};
use iced::Rectangle;
use iced::advanced::graphics::Viewport;
use iced_wgpu::primitive::{Primitive, Storage};
use iced_wgpu::wgpu;
use std::collections::HashMap;
use std::mem;

use crate::ui::render::common::{
    CacheTracker, ClipTransform, InstanceBuffer, create_shader_module,
};
use crate::ui::render::geometry::compute_normals;

#[derive(Debug, Clone)]
pub struct StereometerParams {
    pub instance_id: u64,
    pub bounds: Rectangle,
    pub xy_points: Vec<(f32, f32)>,
    pub color: [f32; 4],
    pub line_alpha: f32,
    pub fade_alpha: f32,
    pub amplitude_scale: f32,
    pub stroke_width: f32,
}

#[derive(Debug)]
pub struct StereometerPrimitive {
    params: StereometerParams,
}

impl StereometerPrimitive {
    pub fn new(params: StereometerParams) -> Self {
        Self { params }
    }

    fn key(&self) -> u64 {
        self.params.instance_id
    }

    fn build_vertices(&self, viewport: &Viewport) -> Vec<Vertex> {
        if self.params.xy_points.len() < 2 {
            return Vec::new();
        }

        let bounds = self.params.bounds;
        let clip = ClipTransform::from_viewport(viewport);

        let center_x = bounds.x + bounds.width * 0.5;
        let center_y = bounds.y + bounds.height * 0.5;
        let scale = 0.9 * self.params.amplitude_scale.max(0.01);
        let scale_x = bounds.width * 0.5 * scale;
        let scale_y = bounds.height * 0.5 * scale;

        let color = [
            self.params.color[0],
            self.params.color[1],
            self.params.color[2],
            self.params.line_alpha,
        ];

        let positions: Vec<_> = self
            .params
            .xy_points
            .iter()
            .map(|(x, y)| {
                (
                    center_x + x.clamp(-1.0, 1.0) * scale_x,
                    center_y - y.clamp(-1.0, 1.0) * scale_y,
                )
            })
            .collect();

        let normals = compute_normals(&positions);
        let half = self.params.stroke_width.max(0.1) * 0.5;
        let feather = 1.0f32;
        let outer = half + feather;

        build_line_strip(&positions, &normals, color, outer, half, feather, &clip)
    }
}

fn build_line_strip(
    positions: &[(f32, f32)],
    normals: &[(f32, f32)],
    color: [f32; 4],
    outer: f32,
    half: f32,
    feather: f32,
    clip: &ClipTransform,
) -> Vec<Vertex> {
    let mut vertices = Vec::with_capacity(positions.len() * 2);
    for (pos, normal) in positions.iter().zip(normals.iter()) {
        let offset_x = normal.0 * outer;
        let offset_y = normal.1 * outer;
        vertices.push(Vertex {
            position: clip.to_clip(pos.0 - offset_x, pos.1 - offset_y),
            color,
            params: [-outer, half, feather, 0.0],
        });
        vertices.push(Vertex {
            position: clip.to_clip(pos.0 + offset_x, pos.1 + offset_y),
            color,
            params: [outer, half, feather, 0.0],
        });
    }
    vertices
}

impl Primitive for StereometerPrimitive {
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
        pipeline.prepare_instance(device, queue, self.key(), &vertices);
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

        let Some(instance) = pipeline.instance(self.key()) else {
            return;
        };

        if instance.buffer.vertex_count == 0 {
            return;
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Stereometer pass"),
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
            instance
                .buffer
                .vertex_buffer
                .slice(0..instance.buffer.used_bytes()),
        );
        pass.draw(0..instance.buffer.vertex_count, 0..1);
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 4],
    params: [f32; 4],
}

impl Vertex {
    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

const VERTEX_LABEL: &str = "Stereometer vertex buffer";

#[derive(Debug)]
struct Pipeline {
    pipeline: wgpu::RenderPipeline,
    instances: HashMap<u64, InstanceRecord>,
    cache: CacheTracker,
}

#[derive(Debug)]
struct InstanceRecord {
    buffer: InstanceBuffer<Vertex>,
    last_used: u64,
}

impl Pipeline {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = create_shader_module(
            device,
            "Stereometer shader",
            include_str!("shaders/stereometer.wgsl"),
        );

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Stereometer pipeline layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Stereometer pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        Self {
            pipeline,
            instances: HashMap::new(),
            cache: CacheTracker::default(),
        }
    }

    fn prepare_instance(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        key: u64,
        vertices: &[Vertex],
    ) {
        let required_size = std::mem::size_of_val(vertices) as wgpu::BufferAddress;
        let (frame, threshold) = self.cache.advance();

        let entry = self.instances.entry(key).or_insert_with(|| InstanceRecord {
            buffer: InstanceBuffer::new(device, VERTEX_LABEL, required_size.max(1)),
            last_used: frame,
        });

        entry.last_used = frame;

        if vertices.is_empty() {
            entry.buffer.vertex_count = 0;
            self.prune(threshold);
            return;
        }

        entry
            .buffer
            .ensure_capacity(device, VERTEX_LABEL, required_size);
        entry.buffer.write(queue, vertices);
        self.prune(threshold);
    }

    fn instance(&self, key: u64) -> Option<&InstanceRecord> {
        self.instances.get(&key)
    }

    fn prune(&mut self, threshold: Option<u64>) {
        if let Some(threshold) = threshold {
            self.instances
                .retain(|_, record| record.last_used >= threshold);
        }
    }
}
