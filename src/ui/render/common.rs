//! Common rendering utilities shared across visualization primitives.

use bytemuck::{Pod, Zeroable};
use iced::advanced::graphics::Viewport;
use iced_wgpu::wgpu;
use std::collections::HashMap;
use std::mem;

/// Transforms logical screen coordinates to clip space coordinates.
#[derive(Clone, Copy)]
pub struct ClipTransform {
    scale_x: f32,
    scale_y: f32,
}

impl ClipTransform {
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            scale_x: 2.0 / width,
            scale_y: 2.0 / height,
        }
    }

    pub fn from_viewport(viewport: &Viewport) -> Self {
        let logical_size = viewport.logical_size();
        Self::new(logical_size.width.max(1.0), logical_size.height.max(1.0))
    }

    #[inline]
    pub fn to_clip(self, x: f32, y: f32) -> [f32; 2] {
        [x * self.scale_x - 1.0, 1.0 - y * self.scale_y]
    }
}

/// Vertex with SDF params for antialiased rendering.
///
/// `params`: `[dist_x, dist_y, radius, feather]`
/// - Solid: `[0, 0, 1000, 1]`
/// - Line: `[±outer, 0, half_width, feather]`
/// - Dot: `[ox, oy, radius, feather]`
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SdfVertex {
    pub position: [f32; 2],
    pub color: [f32; 4],
    pub params: [f32; 4],
}

impl SdfVertex {
    /// Params for solid fills (always full coverage).
    pub const SOLID_PARAMS: [f32; 4] = [0.0, 0.0, 1000.0, 1.0];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<SdfVertex>() as wgpu::BufferAddress,
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

    #[inline]
    pub fn solid(position: [f32; 2], color: [f32; 4]) -> Self {
        Self {
            position,
            color,
            params: Self::SOLID_PARAMS,
        }
    }

    #[inline]
    pub fn antialiased(
        position: [f32; 2],
        color: [f32; 4],
        signed_distance: f32,
        half_width: f32,
        feather: f32,
    ) -> Self {
        Self {
            position,
            color,
            params: [signed_distance, 0.0, half_width, feather],
        }
    }
}

/// Manages a growable GPU vertex buffer for a single primitive instance.
#[derive(Debug)]
pub struct InstanceBuffer<V: Pod> {
    pub vertex_buffer: wgpu::Buffer,
    pub capacity: wgpu::BufferAddress,
    pub vertex_count: u32,
    _marker: std::marker::PhantomData<V>,
}

impl<V: Pod> InstanceBuffer<V> {
    pub fn new(device: &wgpu::Device, label: &'static str, size: wgpu::BufferAddress) -> Self {
        let buffer = create_vertex_buffer(device, label, size.max(1));
        Self {
            vertex_buffer: buffer,
            capacity: size.max(1),
            vertex_count: 0,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn ensure_capacity(
        &mut self,
        device: &wgpu::Device,
        label: &'static str,
        size: wgpu::BufferAddress,
    ) {
        if size <= self.capacity {
            return;
        }

        let new_capacity = size.next_power_of_two().max(1);
        self.vertex_buffer = create_vertex_buffer(device, label, new_capacity);
        self.capacity = new_capacity;
    }

    pub fn write(&mut self, queue: &wgpu::Queue, vertices: &[V]) {
        if vertices.is_empty() {
            self.vertex_count = 0;
            return;
        }

        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(vertices));
        self.vertex_count = vertices.len() as u32;
    }

    pub fn used_bytes(&self) -> wgpu::BufferAddress {
        self.vertex_count as wgpu::BufferAddress * mem::size_of::<V>() as wgpu::BufferAddress
    }
}

pub fn create_vertex_buffer(
    device: &wgpu::Device,
    label: &'static str,
    size: wgpu::BufferAddress,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Produces eviction thresholds for pruning stale cache entries.
#[derive(Debug, Clone)]
pub struct CacheTracker {
    frame: u64,
    counter: u64,
    retain: u64,
    interval: u64,
}

impl CacheTracker {
    pub const fn new(retain: u64, interval: u64) -> Self {
        Self {
            frame: 0,
            counter: 0,
            retain,
            interval,
        }
    }

    /// Returns `(frame, Some(eviction_threshold))` every `interval` frames.
    pub fn advance(&mut self) -> (u64, Option<u64>) {
        self.frame = self.frame.wrapping_add(1).max(1);
        if self.interval == 0 {
            return (self.frame, None);
        }

        self.counter = self.counter.wrapping_add(1);
        if self.counter.is_multiple_of(self.interval) {
            let threshold = self.frame.saturating_sub(self.retain);
            (self.frame, Some(threshold))
        } else {
            (self.frame, None)
        }
    }
}

impl Default for CacheTracker {
    fn default() -> Self {
        Self::new(1024, 256)
    }
}

#[inline]
pub fn create_shader_module(
    device: &wgpu::Device,
    label: &'static str,
    source: &'static str,
) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}

/// Creates a render pipeline using `sdf.wgsl` with the given topology.
pub fn create_sdf_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    label: &'static str,
    topology: wgpu::PrimitiveTopology,
) -> wgpu::RenderPipeline {
    let shader = create_shader_module(device, label, include_str!("shaders/sdf.wgsl"));

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[SdfVertex::layout()],
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
            topology,
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
    })
}

/// Writes a tightly packed texture region with a consistent `bytes_per_row` layout.
#[inline]
pub fn write_texture_region(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    origin: wgpu::Origin3d,
    extent: wgpu::Extent3d,
    bytes_per_row: u32,
    data: &[u8],
) {
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture,
            mip_level: 0,
            origin,
            aspect: wgpu::TextureAspect::All,
        },
        data,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: None,
        },
        extent,
    );
}

/// Helper to generate six SDF vertices forming a solid quad (two triangles).
#[inline]
pub fn quad_vertices(
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    clip: ClipTransform,
    color: [f32; 4],
) -> [SdfVertex; 6] {
    let tl = clip.to_clip(x0, y0);
    let tr = clip.to_clip(x1, y0);
    let bl = clip.to_clip(x0, y1);
    let br = clip.to_clip(x1, y1);

    [
        SdfVertex::solid(tl, color),
        SdfVertex::solid(bl, color),
        SdfVertex::solid(br, color),
        SdfVertex::solid(tl, color),
        SdfVertex::solid(br, color),
        SdfVertex::solid(tr, color),
    ]
}

#[derive(Debug)]
pub struct CachedInstance {
    pub buffer: InstanceBuffer<SdfVertex>,
    pub last_used: u64,
}

/// Pipeline + instance cache for SDF-based primitives.
#[derive(Debug)]
pub struct SdfPipeline<K> {
    pub pipeline: wgpu::RenderPipeline,
    pub instances: HashMap<K, CachedInstance>,
    pub cache: CacheTracker,
}

impl<K: std::hash::Hash + Eq + Copy> SdfPipeline<K> {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        label: &'static str,
        topology: wgpu::PrimitiveTopology,
    ) -> Self {
        Self {
            pipeline: create_sdf_pipeline(device, format, label, topology),
            instances: HashMap::new(),
            cache: CacheTracker::default(),
        }
    }

    pub fn prepare_instance(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: &'static str,
        key: K,
        vertices: &[SdfVertex],
    ) {
        let (frame, threshold) = self.cache.advance();
        let required_size = mem::size_of_val(vertices) as wgpu::BufferAddress;

        let entry = self.instances.entry(key).or_insert_with(|| CachedInstance {
            buffer: InstanceBuffer::new(device, label, required_size.max(1)),
            last_used: frame,
        });
        entry.last_used = frame;

        if vertices.is_empty() {
            entry.buffer.vertex_count = 0;
        } else {
            entry.buffer.ensure_capacity(device, label, required_size);
            entry.buffer.write(queue, vertices);
        }

        if let Some(t) = threshold {
            self.instances.retain(|_, e| e.last_used >= t);
        }
    }

    pub fn instance(&self, key: K) -> Option<&InstanceBuffer<SdfVertex>> {
        self.instances.get(&key).map(|e| &e.buffer)
    }
}
