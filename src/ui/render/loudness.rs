//! Rendering code for the loudness meters.

use iced::Rectangle;
use iced::advanced::graphics::Viewport;
use iced_wgpu::primitive::{Primitive, Storage};
use iced_wgpu::wgpu;
use std::collections::HashMap;

use crate::ui::render::common::{ClipTransform, SimplePipeline, SimpleVertex, quad_vertices};

const GAP_FRACTION: f32 = 0.1;
const BAR_WIDTH_SCALE: f32 = 0.6;
const INNER_GAP_RATIO: f32 = 0.09;
const GUIDE_LENGTH: f32 = 4.0;
const GUIDE_THICKNESS: f32 = 1.0;
const GUIDE_PADDING: f32 = 3.0;
const THRESHOLD_THICKNESS: f32 = 1.5;

/// A single meter bar with background and fill segments.
#[derive(Debug, Clone)]
pub struct MeterBar {
    pub bg_color: [f32; 4],
    pub fills: Vec<(f32, [f32; 4])>,
}

/// Parameters for rendering the loudness meter.
#[derive(Debug, Clone)]
pub struct RenderParams {
    pub min_db: f32,
    pub max_db: f32,
    pub bars: Vec<MeterBar>,
    pub guides: Vec<f32>,
    pub guide_color: [f32; 4],
    pub threshold_db: Option<f32>,
    pub left_padding: f32,
    pub right_padding: f32,
}

impl RenderParams {
    /// Convert dB value to 0..1 ratio with visual scaling.
    pub fn db_to_ratio(&self, db: f32) -> f32 {
        let range = self.max_db - self.min_db;
        if range <= f32::EPSILON {
            return 0.0;
        }
        let raw = ((db - self.min_db) / range).clamp(0.0, 1.0);
        raw.powf(0.9)
    }

    /// Get horizontal bounds of the meter area.
    pub fn meter_bounds(&self, bounds: &Rectangle) -> Option<(f32, f32, f32)> {
        let bar_count = self.bars.len();
        if bar_count == 0 {
            return None;
        }

        let meter_width = (bounds.width - self.left_padding - self.right_padding).max(0.0);
        if meter_width <= 0.0 {
            return None;
        }

        let gap = meter_width * GAP_FRACTION;
        let total_gap = gap * (bar_count - 1) as f32;
        let bar_slot = (meter_width - total_gap) / bar_count as f32;
        let bar_width = bar_slot * BAR_WIDTH_SCALE;
        let bar_offset = (bar_slot - bar_width) * 0.5;
        let stride = bar_width + gap;
        let meter_x = bounds.x + self.left_padding + bar_offset;

        Some((meter_x, bar_width, stride))
    }
}

/// Custom primitive that draws a loudness meter.
#[derive(Debug)]
pub struct LoudnessMeterPrimitive {
    pub params: RenderParams,
}

impl LoudnessMeterPrimitive {
    pub fn new(params: RenderParams) -> Self {
        Self { params }
    }

    fn key(&self) -> usize {
        self as *const Self as usize
    }

    fn build_vertices(&self, bounds: &Rectangle, viewport: &Viewport) -> Vec<SimpleVertex> {
        let clip = ClipTransform::from_viewport(viewport);
        let Some((meter_x, bar_width, stride)) = self.params.meter_bounds(bounds) else {
            return Vec::new();
        };

        let y0 = bounds.y;
        let y1 = bounds.y + bounds.height;
        let height = y1 - y0;
        let bar_count = self.params.bars.len();

        let mut vertices = Vec::with_capacity(bar_count * 18 + self.params.guides.len() * 6 + 12);

        for (i, bar) in self.params.bars.iter().enumerate() {
            let x0 = meter_x + i as f32 * stride;
            let x1 = x0 + bar_width;

            vertices.extend(quad_vertices(x0, y0, x1, y1, clip, bar.bg_color));
            let fill_count = bar.fills.len();
            if fill_count > 0 {
                let inner_gap = if fill_count > 1 && bar_width > 2.0 {
                    (bar_width * INNER_GAP_RATIO).min(bar_width * 0.4).max(0.5)
                } else {
                    0.0
                };
                let total_inner = inner_gap * (fill_count - 1) as f32;
                let seg_width = (bar_width - total_inner) / fill_count as f32;

                for (j, &(db, color)) in bar.fills.iter().enumerate() {
                    let ratio = self.params.db_to_ratio(db);
                    let fill_y = y1 - height * ratio;
                    let sx0 = x0 + j as f32 * (seg_width + inner_gap);
                    let sx1 = if j + 1 == fill_count {
                        x1
                    } else {
                        sx0 + seg_width
                    };
                    vertices.extend(quad_vertices(sx0, fill_y, sx1, y1, clip, color));
                }
            }
        }

        let guide_anchor = meter_x - GUIDE_PADDING;
        for &db in &self.params.guides {
            let ratio = self.params.db_to_ratio(db);
            let cy = y1 - height * ratio;
            let half = GUIDE_THICKNESS * 0.5;
            vertices.extend(quad_vertices(
                guide_anchor - GUIDE_LENGTH,
                (cy - half).max(y0),
                guide_anchor,
                (cy + half).min(y1),
                clip,
                self.params.guide_color,
            ));
        }

        if let Some(db) = self.params.threshold_db {
            let ratio = self.params.db_to_ratio(db);
            let cy = y1 - height * ratio;
            let half = THRESHOLD_THICKNESS * 0.5;
            for i in 0..bar_count {
                let x0 = meter_x + i as f32 * stride;
                let x1 = x0 + bar_width;
                vertices.extend(quad_vertices(
                    x0,
                    (cy - half).max(y0),
                    x1,
                    (cy + half).min(y1),
                    clip,
                    self.params.guide_color,
                ));
            }
        }

        vertices
    }
}

impl Primitive for LoudnessMeterPrimitive {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        storage: &mut Storage,
        bounds: &Rectangle,
        viewport: &Viewport,
    ) {
        if !storage.has::<Pipeline>() {
            storage.store(Pipeline::new(device, format));
        }

        let pipeline = storage.get_mut::<Pipeline>().expect("pipeline exists");
        let vertices = self.build_vertices(bounds, viewport);
        pipeline.prepare_instance(device, queue, "Loudness", self.key(), &vertices);
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
            label: Some("Loudness"),
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

type Pipeline = SimplePipeline<usize>;

impl Pipeline {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Loudness shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/loudness.wgsl").into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Loudness layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Loudness pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[SimpleVertex::layout()],
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
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self {
            pipeline,
            instances: HashMap::new(),
        }
    }
}
