use bytemuck::{Pod, Zeroable};
use iced::Rectangle;
use iced::advanced::graphics::Viewport;
use iced_wgpu::primitive::{Primitive, Storage};
use iced_wgpu::wgpu;
use std::collections::HashMap;
use std::mem;
use std::sync::Arc;

use crate::ui::render::geometry::compute_normals;

#[derive(Debug, Clone)]
pub struct SpectrumParams {
    pub bounds: Rectangle,
    pub normalized_points: Arc<Vec<[f32; 2]>>,
    pub secondary_points: Arc<Vec<[f32; 2]>>,
    pub line_color: [f32; 4],
    pub line_width: f32,
    pub secondary_line_color: [f32; 4],
    pub secondary_line_width: f32,
    pub highlight_threshold: f32,
    pub highlight_color: [f32; 4],
}

#[derive(Debug)]
pub struct SpectrumPrimitive {
    params: SpectrumParams,
}

impl SpectrumPrimitive {
    pub fn new(params: SpectrumParams) -> Self {
        Self { params }
    }

    fn key(&self) -> usize {
        self as *const Self as usize
    }

    fn build_vertices(&self, viewport: &Viewport) -> Vec<Vertex> {
        let bounds = self.params.bounds;
        let clip = ClipTransform::new(
            viewport.logical_size().width.max(1.0),
            viewport.logical_size().height.max(1.0),
        );

        let mut positions = Vec::with_capacity(self.params.normalized_points.len());
        for point in self.params.normalized_points.iter() {
            let amp = point[1].clamp(0.0, 1.0);
            let x = bounds.x + bounds.width * point[0].clamp(0.0, 1.0);
            let y = bounds.y + bounds.height * (1.0 - amp);
            positions.push((x, y));
        }

        if positions.len() < 2 {
            return Vec::new();
        }

        let mut vertices = Vec::new();
        let baseline = bounds.y + bounds.height;

        // Highlight energetic columns before drawing the fill so they sit behind the curve.
        let highlight_color = self.params.highlight_color;
        let highlight_threshold = self.params.highlight_threshold.clamp(0.0, 1.0);
        if highlight_color[3] > 0.0 && highlight_threshold < 1.0 {
            let denom = (1.0 - highlight_threshold).max(1.0e-6);
            for (segment, points) in positions
                .windows(2)
                .zip(self.params.normalized_points.windows(2))
            {
                let amp_max = points[0][1]
                    .clamp(0.0, 1.0)
                    .max(points[1][1].clamp(0.0, 1.0));
                if amp_max < highlight_threshold {
                    continue;
                }

                let intensity = ((amp_max - highlight_threshold) / denom).clamp(0.0, 1.0);
                let alpha = (highlight_color[3] * intensity).clamp(0.0, 1.0);
                if alpha <= 0.0 {
                    continue;
                }

                let (x0, y0) = segment[0];
                let (x1, y1) = segment[1];

                let top_left = clip.to_clip(x0, y0);
                let top_right = clip.to_clip(x1, y1);
                let bottom_left = clip.to_clip(x0, baseline);
                let bottom_right = clip.to_clip(x1, baseline);
                let column_color = [
                    highlight_color[0],
                    highlight_color[1],
                    highlight_color[2],
                    alpha,
                ];

                vertices.extend_from_slice(&triangle_vertices(
                    bottom_left,
                    top_left,
                    top_right,
                    column_color,
                ));
                vertices.extend_from_slice(&triangle_vertices(
                    bottom_left,
                    top_right,
                    bottom_right,
                    column_color,
                ));
            }
        }

        // Build polyline with thickness.
        let normals = compute_normals(&positions);
        let half = self.params.line_width.max(0.1) * 0.5;
        let line_color = self.params.line_color;
        let mut prev_pair: Option<([f32; 2], [f32; 2])> = None;
        for ((x, y), normal) in positions.iter().zip(normals.iter()) {
            let offset_x = normal.0 * half;
            let offset_y = normal.1 * half;
            let current = (
                clip.to_clip(x - offset_x, y - offset_y),
                clip.to_clip(x + offset_x, y + offset_y),
            );
            if let Some((left0, right0)) = prev_pair {
                let (left1, right1) = current;
                vertices.extend_from_slice(&triangle_vertices(left0, right0, right1, line_color));
                vertices.extend_from_slice(&triangle_vertices(left0, right1, left1, line_color));
            }
            prev_pair = Some(current);
        }

        if !self.params.secondary_points.is_empty() {
            let mut overlay_positions = Vec::with_capacity(self.params.secondary_points.len());
            for point in self.params.secondary_points.iter() {
                let x = bounds.x + bounds.width * point[0].clamp(0.0, 1.0);
                let y = bounds.y + bounds.height * (1.0 - point[1].clamp(0.0, 1.0));
                overlay_positions.push((x, y));
            }

            if overlay_positions.len() >= 2 {
                let overlay_normals = compute_normals(&overlay_positions);
                let half_overlay = self.params.secondary_line_width.max(0.1) * 0.5;
                let overlay_color = self.params.secondary_line_color;
                let mut prev_overlay: Option<([f32; 2], [f32; 2])> = None;
                for ((x, y), normal) in overlay_positions.iter().zip(overlay_normals.iter()) {
                    let offset_x = normal.0 * half_overlay;
                    let offset_y = normal.1 * half_overlay;
                    let current = (
                        clip.to_clip(x - offset_x, y - offset_y),
                        clip.to_clip(x + offset_x, y + offset_y),
                    );
                    if let Some((left0, right0)) = prev_overlay {
                        let (left1, right1) = current;
                        vertices.extend_from_slice(&triangle_vertices(
                            left0,
                            right0,
                            right1,
                            overlay_color,
                        ));
                        vertices.extend_from_slice(&triangle_vertices(
                            left0,
                            right1,
                            left1,
                            overlay_color,
                        ));
                    }
                    prev_overlay = Some(current);
                }
            }
        }

        vertices
    }
}

impl Primitive for SpectrumPrimitive {
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
            .expect("spectrum pipeline must exist after storage check");

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
        if instance.vertex_count == 0 {
            return;
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Spectrum pass"),
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
        pass.set_vertex_buffer(0, instance.vertex_buffer.slice(0..instance.used_bytes()));
        pass.draw(0..instance.vertex_count, 0..1);
    }
}

fn triangle_vertices(a: [f32; 2], b: [f32; 2], c: [f32; 2], color: [f32; 4]) -> [Vertex; 3] {
    [
        Vertex { position: a, color },
        Vertex { position: b, color },
        Vertex { position: c, color },
    ]
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 4],
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
            ],
        }
    }
}

#[derive(Clone, Copy)]
struct ClipTransform {
    scale_x: f32,
    scale_y: f32,
}

impl ClipTransform {
    fn new(width: f32, height: f32) -> Self {
        Self {
            scale_x: 2.0 / width,
            scale_y: 2.0 / height,
        }
    }

    fn to_clip(self, x: f32, y: f32) -> [f32; 2] {
        [x * self.scale_x - 1.0, 1.0 - y * self.scale_y]
    }
}

#[derive(Debug)]
struct Pipeline {
    pipeline: wgpu::RenderPipeline,
    instances: HashMap<usize, InstanceBuffer>,
}

impl Pipeline {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Spectrum shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/spectrum.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Spectrum pipeline layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Spectrum pipeline"),
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
                topology: wgpu::PrimitiveTopology::TriangleList,
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
        }
    }

    fn prepare_instance(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        key: usize,
        vertices: &[Vertex],
    ) {
        let required_size = std::mem::size_of_val(vertices) as wgpu::BufferAddress;
        let entry = self
            .instances
            .entry(key)
            .or_insert_with(|| InstanceBuffer::new(device, required_size.max(1)));

        if vertices.is_empty() {
            entry.vertex_count = 0;
            return;
        }

        entry.ensure_capacity(device, required_size);
        queue.write_buffer(&entry.vertex_buffer, 0, bytemuck::cast_slice(vertices));
        entry.vertex_count = vertices.len() as u32;
    }

    fn instance(&self, key: usize) -> Option<&InstanceBuffer> {
        self.instances.get(&key)
    }
}

#[derive(Debug)]
struct InstanceBuffer {
    vertex_buffer: wgpu::Buffer,
    capacity: wgpu::BufferAddress,
    vertex_count: u32,
}

impl InstanceBuffer {
    fn new(device: &wgpu::Device, size: wgpu::BufferAddress) -> Self {
        let buffer = create_vertex_buffer(device, size.max(1));
        Self {
            vertex_buffer: buffer,
            capacity: size.max(1),
            vertex_count: 0,
        }
    }

    fn ensure_capacity(&mut self, device: &wgpu::Device, size: wgpu::BufferAddress) {
        if size <= self.capacity {
            return;
        }

        let new_capacity = size.next_power_of_two().max(1);
        self.vertex_buffer = create_vertex_buffer(device, new_capacity);
        self.capacity = new_capacity;
    }

    fn used_bytes(&self) -> wgpu::BufferAddress {
        self.vertex_count as wgpu::BufferAddress * mem::size_of::<Vertex>() as wgpu::BufferAddress
    }
}

fn create_vertex_buffer(device: &wgpu::Device, size: wgpu::BufferAddress) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Spectrum vertex buffer"),
        size,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
