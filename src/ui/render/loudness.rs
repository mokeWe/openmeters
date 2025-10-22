//! Rendering code for the loudness meters.

use iced::Rectangle;
use iced::advanced::graphics::Viewport;
use iced_wgpu::primitive::{Primitive, Storage};
use iced_wgpu::wgpu;
use std::collections::HashMap;

use crate::ui::render::common::{ClipTransform, SimplePipeline, SimpleVertex, quad_vertices};

const VALUE_SCALE_BASE: &[(f32, f32)] = &[
    (0.0, 0.0),
    (0.12, 0.10),
    (0.25, 0.24),
    (0.40, 0.42),
    (0.55, 0.58),
    (0.70, 0.73),
    (0.82, 0.86),
    (0.92, 0.95),
    (1.0, 1.0),
];

const VALUE_SCALE_EXPONENT: f32 = 1.1;

const INNER_FILL_GAP_RATIO: f32 = 0.09;

/// Describes how a loudness meter should be rendered for a single frame.
#[derive(Debug, Clone)]
pub struct VisualParams {
    pub min_loudness: f32,
    pub max_loudness: f32,
    pub channels: Vec<ChannelVisual>,
    pub channel_gap_fraction: f32,
    pub channel_width_scale: f32,
    pub short_term_value: f32,
    pub value_unit: String,
    pub guides: Vec<GuideLine>,
    pub left_padding: f32,
    pub right_padding: f32,
    pub guide_padding: f32,
    pub value_scale_bias: f32,
    pub vertical_padding: f32,
}

impl VisualParams {
    pub fn clamp_ratio(&self, value: f32) -> f32 {
        if (self.max_loudness - self.min_loudness).abs() <= f32::EPSILON {
            return 0.0;
        }

        ((value - self.min_loudness) / (self.max_loudness - self.min_loudness)).clamp(0.0, 1.0)
    }

    fn gap_fraction(&self) -> f32 {
        self.channel_gap_fraction.clamp(0.0, 0.5)
    }

    pub fn effective_left_padding(&self, total_width: f32) -> f32 {
        if total_width <= 0.0 {
            return 0.0;
        }

        let max_ratio = 0.45;
        self.left_padding.min(total_width * max_ratio)
    }

    pub fn effective_right_padding(&self, total_width: f32, effective_left_padding: f32) -> f32 {
        if total_width <= 0.0 {
            return 0.0;
        }

        let max_ratio = 0.55;
        let desired = self.right_padding.min(total_width * max_ratio);
        let remaining_width = (total_width - effective_left_padding).max(0.0);
        let min_meter_width = 12.0f32;
        if remaining_width <= min_meter_width {
            return 0.0;
        }

        desired.min((remaining_width - min_meter_width).max(0.0))
    }

    pub fn meter_horizontal_bounds(&self, bounds: &Rectangle) -> Option<(f32, f32)> {
        self.meter_geometry(bounds)
            .map(|geometry| (geometry.meter_left, geometry.meter_right))
    }

    pub fn meter_ratio(&self, value: f32) -> f32 {
        let raw = self.clamp_ratio(value);
        if raw <= 0.0 {
            return 0.0;
        }
        if raw >= 1.0 {
            return 1.0;
        }

        let shaped = raw.powf(VALUE_SCALE_EXPONENT);
        let bias = self.value_scale_bias.clamp(0.0, 1.0);
        let upper_idx = VALUE_SCALE_BASE
            .partition_point(|&(r, _)| r < shaped)
            .min(VALUE_SCALE_BASE.len() - 1);
        let lower_idx = upper_idx.saturating_sub(1);
        let (r0, v0) = VALUE_SCALE_BASE[lower_idx];
        let (r1, v1) = VALUE_SCALE_BASE[upper_idx];
        let span = (r1 - r0).max(f32::EPSILON);
        let t = ((shaped - r0) / span).clamp(0.0, 1.0);
        let smooth_t = t * t * (3.0 - 2.0 * t);
        let target0 = lerp(r0, v0, bias);
        let target1 = lerp(r1, v1, bias);
        lerp(target0, target1, smooth_t)
    }

    pub fn vertical_bounds(&self, bounds: &Rectangle) -> Option<(f32, f32)> {
        if bounds.height <= 0.0 {
            return None;
        }

        let padding = self.vertical_padding.max(0.0).min(bounds.height * 0.45);
        let y0 = bounds.y + padding;
        let y1 = bounds.y + bounds.height - padding;

        if y1 - y0 <= f32::EPSILON {
            None
        } else {
            Some((y0, y1))
        }
    }

    fn meter_geometry(&self, bounds: &Rectangle) -> Option<MeterGeometry> {
        let channel_count = self.channels.len();
        if channel_count == 0 {
            return None;
        }

        let total_width = bounds.width.max(0.0);
        let effective_left = self.effective_left_padding(total_width);
        let effective_right = self.effective_right_padding(total_width, effective_left);

        let available_width = (total_width - effective_left - effective_right).max(0.0);
        if available_width <= f32::EPSILON {
            return None;
        }

        let gap = (available_width * self.gap_fraction()).min(available_width);
        let total_gap = gap * (channel_count.saturating_sub(1) as f32);
        let bar_area_width = (available_width - total_gap).max(0.0);
        if bar_area_width <= f32::EPSILON {
            return None;
        }

        let channel_count_f = channel_count as f32;
        let bar_slot_width = (bar_area_width / channel_count_f).max(0.0);
        let width_scale = self.channel_width_scale.clamp(0.05, 1.0);
        let bar_width = (bar_slot_width * width_scale).max(0.0);
        if bar_width <= f32::EPSILON {
            return None;
        }

        let stride = bar_width + gap;
        let bar_offset = ((bar_slot_width - bar_width) * 0.5).max(0.0);
        let meter_left = bounds.x + effective_left + bar_offset;
        let span = channel_count.saturating_sub(1) as f32;
        let meter_right = meter_left + span * stride + bar_width;

        Some(MeterGeometry {
            meter_left,
            meter_right,
            bar_width,
            stride,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct MeterGeometry {
    meter_left: f32,
    meter_right: f32,
    bar_width: f32,
    stride: f32,
}

#[derive(Debug, Clone)]
pub struct ChannelVisual {
    pub background_color: [f32; 4],
    pub fills: Vec<FillVisual>,
}

#[derive(Debug, Clone)]
pub struct FillVisual {
    pub value_loudness: f32,
    pub color: [f32; 4],
}

#[derive(Debug, Clone)]
pub struct GuideLine {
    pub value_loudness: f32,
    pub color: [f32; 4],
    pub length: f32,
    pub thickness: f32,
    pub label: Option<String>,
    pub label_width: f32,
}

/// Custom primitive that draws a loudness meter using the iced_wgpu backend.
#[derive(Debug)]
pub struct LoudnessMeterPrimitive {
    pub params: VisualParams,
}

impl LoudnessMeterPrimitive {
    pub fn new(params: VisualParams) -> Self {
        Self { params }
    }

    fn key(&self) -> usize {
        self as *const Self as usize
    }

    fn build_vertices(&self, bounds: &Rectangle, viewport: &Viewport) -> Vec<SimpleVertex> {
        let Some(layout) = LayoutContext::new(bounds, viewport, &self.params) else {
            return Vec::new();
        };

        let mut vertices = Vec::with_capacity(layout.estimate_vertex_count(&self.params));
        layout.push_channel_vertices(&mut vertices, &self.params);
        layout.push_guide_vertices(&mut vertices, &self.params);
        vertices
    }
}

struct LayoutContext {
    clip: ClipTransform,
    x0: f32,
    x1: f32,
    y0: f32,
    y1: f32,
    height: f32,
    bar_width: f32,
    stride: f32,
}

impl LayoutContext {
    fn new(bounds: &Rectangle, viewport: &Viewport, params: &VisualParams) -> Option<Self> {
        let (y0, y1) = params.vertical_bounds(bounds)?;
        let geometry = params.meter_geometry(bounds)?;
        let clip = ClipTransform::from_viewport(viewport);
        let height = y1 - y0;
        if height <= f32::EPSILON {
            return None;
        }

        Some(Self {
            clip,
            x0: geometry.meter_left,
            x1: geometry.meter_right,
            y0,
            y1,
            height,
            bar_width: geometry.bar_width,
            stride: geometry.stride,
        })
    }

    fn estimate_vertex_count(&self, params: &VisualParams) -> usize {
        params
            .channels
            .iter()
            .map(|ch| 6 + ch.fills.len() * 6)
            .sum::<usize>()
            + params.guides.len() * 6
    }

    fn bar_span(&self, index: usize) -> (f32, f32) {
        let offset = index as f32 * self.stride;
        let bar_x0 = self.x0 + offset;
        let bar_x1 = (bar_x0 + self.bar_width).min(self.x1);
        (bar_x0, bar_x1)
    }

    fn push_channel_vertices(&self, vertices: &mut Vec<SimpleVertex>, params: &VisualParams) {
        for (index, channel) in params.channels.iter().enumerate() {
            let (bar_x0, bar_x1) = self.bar_span(index);
            vertices.extend(quad_vertices(
                bar_x0,
                self.y0,
                bar_x1,
                self.y1,
                self.clip,
                channel.background_color,
            ));

            let fill_count = channel.fills.len();
            if fill_count == 0 {
                continue;
            }

            let bar_width = bar_x1 - bar_x0;
            if bar_width <= f32::EPSILON {
                continue;
            }

            let inner_gap = Self::inner_gap(bar_width, fill_count);
            let total_gap = inner_gap * fill_count.saturating_sub(1) as f32;
            let available_width = (bar_width - total_gap).max(0.0);
            if available_width <= f32::EPSILON {
                continue;
            }

            let segment_width = available_width / fill_count as f32;

            for (segment_index, fill) in channel.fills.iter().enumerate() {
                let fill_ratio = params.meter_ratio(fill.value_loudness);
                let fill_top = self.y1 - self.height * fill_ratio;
                let seg_x0 = bar_x0 + segment_index as f32 * (segment_width + inner_gap);
                let seg_x1 = if segment_index + 1 == fill_count {
                    bar_x1
                } else {
                    seg_x0 + segment_width
                };

                vertices.extend(quad_vertices(
                    seg_x0, fill_top, seg_x1, self.y1, self.clip, fill.color,
                ));
            }
        }
    }

    fn inner_gap(bar_width: f32, fill_count: usize) -> f32 {
        if fill_count <= 1 || bar_width <= f32::EPSILON {
            return 0.0;
        }

        let desired_gap = bar_width * INNER_FILL_GAP_RATIO;
        let max_gap = bar_width * 0.4;
        let min_gap = 0.5f32.min(max_gap);
        desired_gap.clamp(min_gap, max_gap)
    }

    fn push_guide_vertices(&self, vertices: &mut Vec<SimpleVertex>, params: &VisualParams) {
        let anchor_x = self.x0 - params.guide_padding;
        for guide in &params.guides {
            let ratio = params.clamp_ratio(guide.value_loudness);
            let center_y = self.y1 - self.height * ratio;
            let half_thick = guide.thickness * 0.5;
            let top = (center_y - half_thick).clamp(self.y0, self.y1);
            let bottom = (center_y + half_thick).clamp(self.y0, self.y1);
            let line_x0 = anchor_x - guide.length;

            vertices.extend(quad_vertices(
                line_x0,
                top,
                anchor_x,
                bottom,
                self.clip,
                guide.color,
            ));
        }
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
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

        let pipeline = storage
            .get_mut::<Pipeline>()
            .expect("pipeline must exist after storage check");

        let vertices = self.build_vertices(bounds, viewport);
        pipeline.prepare_instance(
            device,
            queue,
            "Loudness vertex buffer",
            self.key(),
            &vertices,
        );
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
            label: Some("Loudness pass"),
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Loudness pipeline layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Loudness meter pipeline"),
            layout: Some(&pipeline_layout),
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
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
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
}
