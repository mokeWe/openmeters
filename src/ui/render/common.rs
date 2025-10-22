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

/// Simple 2D vertex with position and color.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SimpleVertex {
    pub position: [f32; 2],
    pub color: [f32; 4],
}

impl SimpleVertex {
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<SimpleVertex>() as wgpu::BufferAddress,
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

/// Creates a GPU buffer for vertex data with dynamic sizing.
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

/// Tracks frame advancement and produces thresholds for pruning cached instances.
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

    /// Advances the internal frame counter and returns the current frame along
    /// with an optional eviction threshold.
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

/// Convenience wrapper for WGSL shader compilation.
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

/// Helper to generate six vertices forming a quad (two triangles).
#[inline]
pub fn quad_vertices(
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    clip: ClipTransform,
    color: [f32; 4],
) -> [SimpleVertex; 6] {
    let tl = clip.to_clip(x0, y0);
    let tr = clip.to_clip(x1, y0);
    let bl = clip.to_clip(x0, y1);
    let br = clip.to_clip(x1, y1);

    [
        SimpleVertex {
            position: tl,
            color,
        },
        SimpleVertex {
            position: bl,
            color,
        },
        SimpleVertex {
            position: br,
            color,
        },
        SimpleVertex {
            position: tl,
            color,
        },
        SimpleVertex {
            position: br,
            color,
        },
        SimpleVertex {
            position: tr,
            color,
        },
    ]
}

/// Helper to generate three vertices forming a triangle.
#[inline]
pub fn triangle_vertices(
    a: [f32; 2],
    b: [f32; 2],
    c: [f32; 2],
    color: [f32; 4],
) -> [SimpleVertex; 3] {
    [
        SimpleVertex { position: a, color },
        SimpleVertex { position: b, color },
        SimpleVertex { position: c, color },
    ]
}

/// Common pipeline management for simple vertex-only rendering.
#[derive(Debug)]
pub struct SimplePipeline<K> {
    pub pipeline: wgpu::RenderPipeline,
    pub instances: HashMap<K, InstanceBuffer<SimpleVertex>>,
}

impl<K: std::hash::Hash + Eq + Copy> SimplePipeline<K> {
    pub fn prepare_instance(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: &'static str,
        key: K,
        vertices: &[SimpleVertex],
    ) {
        let required_size = mem::size_of_val(vertices) as wgpu::BufferAddress;
        let entry = self
            .instances
            .entry(key)
            .or_insert_with(|| InstanceBuffer::new(device, label, required_size.max(1)));

        if vertices.is_empty() {
            entry.vertex_count = 0;
            return;
        }

        entry.ensure_capacity(device, label, required_size);
        entry.write(queue, vertices);
    }

    pub fn instance(&self, key: K) -> Option<&InstanceBuffer<SimpleVertex>> {
        self.instances.get(&key)
    }
}
