use bytemuck::{Pod, Zeroable};
use iced::Rectangle;
use iced::advanced::graphics::Viewport;
use iced_wgpu::primitive::{Primitive, Storage};
use std::collections::HashMap;
use std::mem;

#[derive(Debug, Clone)]
pub struct OscilloscopeParams {
    pub bounds: Rectangle,
    pub channels: usize,
    pub samples_per_channel: usize,
    pub samples: Vec<f32>,
    pub colors: Vec<[f32; 4]>,
    pub line_alpha: f32,
    pub fade_alpha: f32,
    pub vertical_padding: f32,
    pub channel_gap: f32,
    pub amplitude_scale: f32,
    pub stroke_width: f32,
}

#[derive(Debug)]
pub struct OscilloscopePrimitive {
    params: OscilloscopeParams,
}

impl OscilloscopePrimitive {
    pub fn new(params: OscilloscopeParams) -> Self {
        Self { params }
    }

    fn key(&self) -> usize {
        self as *const Self as usize
    }

    fn build_vertices(&self, viewport: &Viewport) -> Vec<Vertex> {
        let samples_per_channel = self.params.samples_per_channel;
        let channels = self.params.channels.max(1);
        let total_samples = channels * samples_per_channel;
        if samples_per_channel < 2 || self.params.samples.len() < total_samples {
            return Vec::new();
        }

        let bounds = self.params.bounds;
        let clip = ClipTransform::new(
            viewport.logical_size().width.max(1.0),
            viewport.logical_size().height.max(1.0),
        );

        let width = bounds.width.max(1.0);
        let vertical_padding = self.params.vertical_padding.max(0.0);
        let channel_gap = self.params.channel_gap.max(0.0);
        let usable_height = (bounds.height
            - vertical_padding * 2.0
            - channel_gap * (channels.saturating_sub(1) as f32))
            .max(1.0);
        let channel_height = usable_height / channels as f32;
        let amplitude_scale = channel_height * 0.5 * self.params.amplitude_scale.max(0.01);
        let step = width / (samples_per_channel.saturating_sub(1) as f32).max(1.0);

        let mut vertices = Vec::with_capacity(samples_per_channel * 2 * channels);
        let half_thickness = (self.params.stroke_width.max(0.1)) * 0.5;
        let mut previous_last: Option<Vertex> = None;
        for channel in 0..channels {
            let color = self
                .params
                .colors
                .get(channel)
                .cloned()
                .unwrap_or([0.6, 0.8, 0.9, 1.0]);
            let color = [color[0], color[1], color[2], self.params.line_alpha];

            let top = bounds.y + vertical_padding + channel as f32 * (channel_height + channel_gap);
            let center = top + channel_height * 0.5;

            let mut channel_vertices = Vec::with_capacity(samples_per_channel * 2);
            let mut positions = Vec::with_capacity(samples_per_channel);

            for index in 0..samples_per_channel {
                let sample_index = channel * samples_per_channel + index;
                if let Some(sample) = self.params.samples.get(sample_index) {
                    let x = bounds.x + step * index as f32;
                    let y = center - sample * amplitude_scale;
                    positions.push((x, y));
                }
            }

            if positions.len() < 2 {
                continue;
            }

            let normals = compute_normals(&positions);

            for (pos, normal) in positions.iter().zip(normals.iter()) {
                let offset_x = normal.0 * half_thickness;
                let offset_y = normal.1 * half_thickness;
                let left = clip.to_clip(pos.0 - offset_x, pos.1 - offset_y);
                let right = clip.to_clip(pos.0 + offset_x, pos.1 + offset_y);

                channel_vertices.push(Vertex {
                    position: left,
                    color,
                });
                channel_vertices.push(Vertex {
                    position: right,
                    color,
                });
            }

            if let (Some(last), Some(first)) = (previous_last, channel_vertices.first().cloned()) {
                vertices.push(last);
                vertices.push(first);
            }

            previous_last = channel_vertices.last().cloned();
            vertices.extend(channel_vertices);
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
        pass.set_vertex_buffer(0, instance.vertex_buffer.slice(0..instance.used_bytes()));
        pass.draw(0..instance.vertex_count, 0..1);
    }
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

    #[inline]
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
            label: Some("Oscilloscope shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/oscilloscope.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Oscilloscope pipeline layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Oscilloscope pipeline"),
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
        label: Some("Oscilloscope vertex buffer"),
        size,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn compute_normals(positions: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let count = positions.len();
    if count == 0 {
        return Vec::new();
    }

    if count == 1 {
        return vec![(0.0, 1.0)];
    }

    let mut segment_normals = Vec::with_capacity(count.saturating_sub(1));
    for i in 0..(count - 1) {
        let (x0, y0) = positions[i];
        let (x1, y1) = positions[i + 1];
        let dx = x1 - x0;
        let dy = y1 - y0;
        let length = (dx * dx + dy * dy).sqrt();
        if length <= f32::EPSILON {
            segment_normals.push((0.0, 1.0));
        } else {
            segment_normals.push((-dy / length, dx / length));
        }
    }

    let mut normals = Vec::with_capacity(count);
    for i in 0..count {
        let normal = if i == 0 {
            segment_normals.first().copied().unwrap_or((0.0, 1.0))
        } else if i == count - 1 {
            segment_normals
                .get(count - 2)
                .copied()
                .unwrap_or((0.0, 1.0))
        } else {
            let prev = segment_normals.get(i - 1).copied().unwrap_or((0.0, 1.0));
            let next = segment_normals.get(i).copied().unwrap_or((0.0, 1.0));
            let nx = prev.0 + next.0;
            let ny = prev.1 + next.1;
            let len = (nx * nx + ny * ny).sqrt();
            if len <= f32::EPSILON {
                next
            } else {
                (nx / len, ny / len)
            }
        };
        normals.push(normal);
    }

    normals
}
