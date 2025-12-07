use iced::Rectangle;
use iced::advanced::graphics::Viewport;
use iced_wgpu::primitive::{Primitive, Storage};
use iced_wgpu::wgpu;

use crate::ui::render::common::{ClipTransform, InstanceBuffer, SdfPipeline, SdfVertex};
use crate::ui::settings::StereometerMode;

const PI: f32 = std::f32::consts::PI;
const SQRT2_INV: f32 = std::f32::consts::FRAC_1_SQRT_2;

#[derive(Debug, Clone)]
pub struct StereometerParams {
    pub instance_id: u64,
    pub bounds: Rectangle,
    pub points: Vec<(f32, f32)>,
    pub trace_color: [f32; 4],
    pub grid_color: [f32; 4],
    pub mode: StereometerMode,
}

#[derive(Debug)]
pub struct StereometerPrimitive {
    params: StereometerParams,
}

impl StereometerPrimitive {
    pub fn new(params: StereometerParams) -> Self {
        Self { params }
    }

    fn build_vertices(&self, viewport: &Viewport) -> Vec<SdfVertex> {
        let bounds = self.params.bounds;
        let clip = ClipTransform::from_viewport(viewport);
        let cx = bounds.x + bounds.width * 0.5;
        let cy = bounds.y + bounds.height * 0.5;
        let radius = (bounds.width.min(bounds.height) * 0.5) - 2.0;

        let mut verts = Vec::new();

        match self.params.mode {
            StereometerMode::DotCloud => {
                self.build_grid(&mut verts, cx, cy, radius, &clip);
                self.build_dots(&mut verts, cx, cy, radius, &clip);
            }
            StereometerMode::Lissajous => {
                self.build_trace(&mut verts, cx, cy, radius * 0.9, &clip);
            }
        }

        verts
    }

    fn build_grid(
        &self,
        verts: &mut Vec<SdfVertex>,
        cx: f32,
        cy: f32,
        radius: f32,
        clip: &ClipTransform,
    ) {
        let color = self.params.grid_color;

        for &r in &[0.125, 0.25, 0.5, 1.0] {
            self.build_circle(verts, cx, cy, radius * r, color, clip);
        }

        for &(dx, dy) in &[
            (1.0, 0.0),
            (0.0, 1.0),
            (SQRT2_INV, SQRT2_INV),
            (SQRT2_INV, -SQRT2_INV),
        ] {
            let p0 = (cx - dx * radius, cy - dy * radius);
            let p1 = (cx + dx * radius, cy + dy * radius);
            self.build_line_segment(verts, p0, p1, color, color, clip);
        }
    }

    fn build_dots(
        &self,
        verts: &mut Vec<SdfVertex>,
        cx: f32,
        cy: f32,
        radius: f32,
        clip: &ClipTransform,
    ) {
        let n = self.params.points.len();
        if n == 0 {
            return;
        }

        let scale = radius * SQRT2_INV;
        let [cr, cg, cb, ca] = self.params.trace_color;

        for (i, &(l, r)) in self.params.points.iter().enumerate() {
            let alpha = ca * (i + 1) as f32 / n as f32;
            let px = cx + (r - l) * SQRT2_INV * scale;
            let py = cy - (l + r) * SQRT2_INV * scale;
            self.build_dot(verts, px, py, [cr, cg, cb, alpha], clip);
        }
    }

    fn build_trace(
        &self,
        verts: &mut Vec<SdfVertex>,
        cx: f32,
        cy: f32,
        radius: f32,
        clip: &ClipTransform,
    ) {
        let n = self.params.points.len();
        if n < 2 {
            return;
        }

        let [cr, cg, cb, ca] = self.params.trace_color;

        for i in 0..n - 1 {
            let (x0, y0) = self.params.points[i];
            let (x1, y1) = self.params.points[i + 1];

            let p0 = (
                cx + x0.clamp(-1.0, 1.0) * radius,
                cy - y0.clamp(-1.0, 1.0) * radius,
            );
            let p1 = (
                cx + x1.clamp(-1.0, 1.0) * radius,
                cy - y1.clamp(-1.0, 1.0) * radius,
            );

            let t0 = i as f32 / (n - 1) as f32;
            let t1 = (i + 1) as f32 / (n - 1) as f32;

            self.build_line_segment(
                verts,
                p0,
                p1,
                [cr, cg, cb, ca * t0],
                [cr, cg, cb, ca * t1],
                clip,
            );
        }
    }

    fn build_circle(
        &self,
        verts: &mut Vec<SdfVertex>,
        cx: f32,
        cy: f32,
        r: f32,
        color: [f32; 4],
        clip: &ClipTransform,
    ) {
        const SEGMENTS: usize = 64;
        for i in 0..SEGMENTS {
            let a0 = (i as f32 / SEGMENTS as f32) * 2.0 * PI;
            let a1 = ((i + 1) as f32 / SEGMENTS as f32) * 2.0 * PI;
            let p0 = (cx + a0.cos() * r, cy + a0.sin() * r);
            let p1 = (cx + a1.cos() * r, cy + a1.sin() * r);
            self.build_line_segment(verts, p0, p1, color, color, clip);
        }
    }

    fn build_line_segment(
        &self,
        verts: &mut Vec<SdfVertex>,
        p0: (f32, f32),
        p1: (f32, f32),
        c0: [f32; 4],
        c1: [f32; 4],
        clip: &ClipTransform,
    ) {
        let dx = p1.0 - p0.0;
        let dy = p1.1 - p0.1;
        let len = (dx * dx + dy * dy).sqrt().max(1e-6);
        let (nx, ny) = (-dy / len, dx / len);

        const OUTER: f32 = 1.5;
        const HALF: f32 = 0.5;
        const FEATHER: f32 = 1.0;

        let (ox, oy) = (nx * OUTER, ny * OUTER);

        let v0 = SdfVertex {
            position: clip.to_clip(p0.0 - ox, p0.1 - oy),
            color: c0,
            params: [-OUTER, 0.0, HALF, FEATHER],
        };
        let v1 = SdfVertex {
            position: clip.to_clip(p0.0 + ox, p0.1 + oy),
            color: c0,
            params: [OUTER, 0.0, HALF, FEATHER],
        };
        let v2 = SdfVertex {
            position: clip.to_clip(p1.0 - ox, p1.1 - oy),
            color: c1,
            params: [-OUTER, 0.0, HALF, FEATHER],
        };
        let v3 = SdfVertex {
            position: clip.to_clip(p1.0 + ox, p1.1 + oy),
            color: c1,
            params: [OUTER, 0.0, HALF, FEATHER],
        };

        verts.extend([v0, v1, v2, v1, v3, v2]);
    }

    fn build_dot(
        &self,
        verts: &mut Vec<SdfVertex>,
        cx: f32,
        cy: f32,
        color: [f32; 4],
        clip: &ClipTransform,
    ) {
        const R: f32 = 1.5;
        const FEATHER: f32 = 0.75;
        let o = R + FEATHER;

        let v = |px, py, ox, oy| SdfVertex {
            position: clip.to_clip(px, py),
            color,
            params: [ox, oy, R, FEATHER],
        };

        verts.extend([
            v(cx - o, cy - o, -o, -o),
            v(cx - o, cy + o, -o, o),
            v(cx + o, cy - o, o, -o),
            v(cx + o, cy - o, o, -o),
            v(cx - o, cy + o, -o, o),
            v(cx + o, cy + o, o, o),
        ]);
    }
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

        let pipeline = storage.get_mut::<Pipeline>().unwrap();
        let vertices = self.build_vertices(viewport);
        pipeline.prepare(device, queue, self.params.instance_id, &vertices);
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
        let Some(instance) = pipeline.instance(self.params.instance_id) else {
            return;
        };
        if instance.vertex_count == 0 {
            return;
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Stereometer"),
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
        pass.set_pipeline(&pipeline.inner.pipeline);
        pass.set_vertex_buffer(
            0,
            instance.vertex_buffer.slice(0..instance.used_bytes()),
        );
        pass.draw(0..instance.vertex_count, 0..1);
    }
}

#[derive(Debug)]
struct Pipeline {
    inner: SdfPipeline<u64>,
}

impl Pipeline {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        Self {
            inner: SdfPipeline::new(device, format, "Stereometer", wgpu::PrimitiveTopology::TriangleList),
        }
    }

    fn prepare(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, key: u64, vertices: &[SdfVertex]) {
        self.inner.prepare_instance(device, queue, "Stereometer", key, vertices);
    }

    fn instance(&self, key: u64) -> Option<&InstanceBuffer<SdfVertex>> {
        self.inner.instance(key)
    }
}
