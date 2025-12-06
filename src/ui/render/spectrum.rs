use bytemuck::{Pod, Zeroable};
use iced::Rectangle;
use iced::advanced::graphics::Viewport;
use iced_wgpu::primitive::{Primitive, Storage};
use iced_wgpu::wgpu;
use std::collections::HashMap;
use std::mem;
use std::sync::Arc;

use crate::ui::render::common::{ClipTransform, InstanceBuffer};
use crate::ui::render::geometry::compute_normals;

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

#[derive(Debug, Clone)]
pub struct SpectrumParams {
    pub bounds: Rectangle,
    pub normalized_points: Arc<Vec<[f32; 2]>>,
    pub secondary_points: Arc<Vec<[f32; 2]>>,
    pub instance_key: usize,
    pub line_color: [f32; 4],
    pub line_width: f32,
    pub secondary_line_color: [f32; 4],
    pub secondary_line_width: f32,
    pub highlight_threshold: f32,
    pub spectrum_palette: Vec<[f32; 4]>,
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
        self.params.instance_key
    }

    fn build_vertices(&self, viewport: &Viewport) -> Vec<Vertex> {
        let bounds = self.params.bounds;
        let clip = ClipTransform::from_viewport(viewport);

        let positions = to_cartesian_positions(bounds, self.params.normalized_points.as_ref());
        if positions.len() < 2 {
            return Vec::new();
        }

        let mut vertices = Vec::new();
        let baseline = bounds.y + bounds.height;

        push_highlight_columns(
            &mut vertices,
            &clip,
            baseline,
            &positions,
            self.params.normalized_points.as_ref(),
            &self.params.spectrum_palette,
            self.params.highlight_threshold,
        );

        push_polyline(
            &mut vertices,
            &positions,
            self.params.line_width,
            self.params.line_color,
            &clip,
        );

        if self.params.secondary_points.len() >= 2 {
            let overlay_positions =
                to_cartesian_positions(bounds, self.params.secondary_points.as_ref());
            push_polyline(
                &mut vertices,
                &overlay_positions,
                self.params.secondary_line_width,
                self.params.secondary_line_color,
                &clip,
            );
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

fn to_cartesian_positions(bounds: Rectangle, points: &[[f32; 2]]) -> Vec<(f32, f32)> {
    let mut positions = Vec::with_capacity(points.len());
    for point in points.iter() {
        let x = bounds.x + bounds.width * point[0].clamp(0.0, 1.0);
        let y = bounds.y + bounds.height * (1.0 - point[1].clamp(0.0, 1.0));
        positions.push((x, y));
    }
    positions
}

fn push_highlight_columns(
    vertices: &mut Vec<Vertex>,
    clip: &ClipTransform,
    baseline: f32,
    positions: &[(f32, f32)],
    normalized_points: &[[f32; 2]],
    spectrum_palette: &[[f32; 4]],
    highlight_threshold: f32,
) {
    let threshold = highlight_threshold.clamp(0.0, 1.0);
    if spectrum_palette.is_empty() || threshold >= 1.0 {
        return;
    }

    let denom = (1.0 - threshold).max(1.0e-6);
    for (segment, points) in positions.windows(2).zip(normalized_points.windows(2)) {
        let amp_max = points[0][1]
            .clamp(0.0, 1.0)
            .max(points[1][1].clamp(0.0, 1.0));
        if amp_max < threshold {
            continue;
        }

        let intensity = ((amp_max - threshold) / denom).clamp(0.0, 1.0);
        let color = interpolate_palette_color(spectrum_palette, intensity);
        if color[3] <= 0.0 {
            continue;
        }

        let (x0, y0) = segment[0];
        let (x1, y1) = segment[1];
        let tl = clip.to_clip(x0, y0);
        let tr = clip.to_clip(x1, y1);
        let bl = clip.to_clip(x0, baseline);
        let br = clip.to_clip(x1, baseline);
        let params = [0.0; 4];

        vertices.extend_from_slice(&[
            Vertex {
                position: bl,
                color,
                params,
            },
            Vertex {
                position: tl,
                color,
                params,
            },
            Vertex {
                position: tr,
                color,
                params,
            },
            Vertex {
                position: bl,
                color,
                params,
            },
            Vertex {
                position: tr,
                color,
                params,
            },
            Vertex {
                position: br,
                color,
                params,
            },
        ]);
    }
}

/// Interpolates a color from the palette based on a normalized value [0.0, 1.0].
/// Returns an RGBA color array.
fn interpolate_palette_color(palette: &[[f32; 4]], t: f32) -> [f32; 4] {
    if palette.is_empty() {
        return [0.0, 0.0, 0.0, 0.0];
    }

    if palette.len() == 1 {
        return palette[0];
    }

    let t = t.clamp(0.0, 1.0);
    let max_index = (palette.len() - 1) as f32;
    let position = t * max_index;
    let index = position.floor() as usize;

    if index >= palette.len() - 1 {
        return palette[palette.len() - 1];
    }

    let fraction = position - index as f32;
    let color_a = palette[index];
    let color_b = palette[index + 1];

    [
        color_a[0] + (color_b[0] - color_a[0]) * fraction,
        color_a[1] + (color_b[1] - color_a[1]) * fraction,
        color_a[2] + (color_b[2] - color_a[2]) * fraction,
        color_a[3] + (color_b[3] - color_a[3]) * fraction,
    ]
}

fn push_polyline(
    vertices: &mut Vec<Vertex>,
    positions: &[(f32, f32)],
    width: f32,
    color: [f32; 4],
    clip: &ClipTransform,
) {
    if positions.len() < 2 || width < 0.0 {
        return;
    }

    let normals = compute_normals(positions);
    let half = width.max(0.1) * 0.5;
    let feather = 0.5;
    let outer = half + feather;

    let mut prev = None;
    for ((x, y), (nx, ny)) in positions.iter().zip(normals.iter()) {
        let current = (
            Vertex {
                position: clip.to_clip(x - nx * outer, y - ny * outer),
                color,
                params: [-outer, half, feather, 0.0],
            },
            Vertex {
                position: clip.to_clip(x + nx * outer, y + ny * outer),
                color,
                params: [outer, half, feather, 0.0],
            },
        );

        if let Some((l0, r0)) = prev {
            let (l1, r1) = &current;
            vertices.extend_from_slice(&[l0, r0, *r1, l0, *r1, *l1]);
        }
        prev = Some(current);
    }
}

#[derive(Debug)]
struct Pipeline {
    pipeline: wgpu::RenderPipeline,
    instances: HashMap<usize, InstanceBuffer<Vertex>>,
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
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
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
        let required_size = mem::size_of_val(vertices) as wgpu::BufferAddress;
        let buffer = self.instances.entry(key).or_insert_with(|| {
            InstanceBuffer::new(device, "Spectrum vertex buffer", required_size.max(1))
        });

        if vertices.is_empty() {
            buffer.vertex_count = 0;
        } else {
            buffer.ensure_capacity(device, "Spectrum vertex buffer", required_size);
            buffer.write(queue, vertices);
        }
    }

    fn instance(&self, key: usize) -> Option<&InstanceBuffer<Vertex>> {
        self.instances.get(&key)
    }
}
