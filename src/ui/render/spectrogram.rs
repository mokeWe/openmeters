use bytemuck::{Pod, Zeroable};
use iced::Rectangle;
use iced::advanced::graphics::Viewport;
use iced_wgpu::primitive::{self, Primitive};
use iced_wgpu::wgpu;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::ui::render::common::{
    CacheTracker, ClipTransform, InstanceBuffer, create_shader_module, write_texture_region,
};

pub const SPECTROGRAM_PALETTE_SIZE: usize = 5;
pub const PALETTE_LUT_SIZE: u32 = 256;
const FLAG_CAPACITY_POW2: u32 = 0b1;

#[derive(Debug, Clone)]
pub struct SpectrogramParams {
    pub instance_id: u64,
    pub bounds: Rectangle,
    pub texture_width: u32,
    pub texture_height: u32,
    pub column_count: u32,
    pub latest_column: u32,
    pub base_data: Option<Arc<Vec<f32>>>,
    pub column_updates: Vec<SpectrogramColumnUpdate>,
    pub palette: [[f32; 4]; SPECTROGRAM_PALETTE_SIZE],
    pub background: [f32; 4],
    pub contrast: f32,
}

#[derive(Debug, Clone)]
pub struct SpectrogramColumnUpdate {
    pub column_index: u32,
    pub values: Arc<ColumnBuffer>,
}

#[derive(Clone, Debug)]
pub struct ColumnBufferPool {
    buffers: Arc<Mutex<Vec<Vec<f32>>>>,
}

impl ColumnBufferPool {
    pub fn new() -> Self {
        Self {
            buffers: Arc::new(Mutex::new(Vec::with_capacity(32))),
        }
    }

    pub fn acquire(&self, len: usize) -> Vec<f32> {
        let mut buffers = self.buffers.lock().unwrap();

        let pos = buffers.iter().rposition(|b| b.capacity() >= len);

        if let Some(idx) = pos {
            let mut buffer = buffers.swap_remove(idx);
            buffer.clear();
            buffer.resize(len, 0.0);
            buffer
        } else {
            // Pre-allocate with 25% headroom to reduce future reallocations
            let capacity = (len * 5) / 4;
            let mut buffer = Vec::with_capacity(capacity);
            buffer.resize(len, 0.0);
            buffer
        }
    }

    pub fn release(&self, mut buffer: Vec<f32>) {
        const MAX_POOL_SIZE: usize = 64;
        const MAX_BUFFER_CAPACITY: usize = 16_384;

        if buffer.capacity() > MAX_BUFFER_CAPACITY {
            return;
        }

        buffer.clear();
        let mut buffers = self.buffers.lock().unwrap();

        if buffers.len() < MAX_POOL_SIZE {
            buffers.push(buffer);
        }
    }
}

impl Default for ColumnBufferPool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct ColumnBuffer {
    pool: ColumnBufferPool,
    data: Option<Vec<f32>>, // owned column data reused via pool
}

impl ColumnBuffer {
    pub fn new(data: Vec<f32>, pool: ColumnBufferPool) -> Self {
        Self {
            pool,
            data: Some(data),
        }
    }

    pub fn as_slice(&self) -> &[f32] {
        self.data.as_deref().unwrap_or(&[])
    }
}

impl Drop for ColumnBuffer {
    fn drop(&mut self) {
        if let Some(data) = self.data.take() {
            self.pool.release(data);
        }
    }
}

#[derive(Debug)]
pub struct SpectrogramPrimitive {
    params: SpectrogramParams,
}

impl SpectrogramPrimitive {
    pub fn new(params: SpectrogramParams) -> Self {
        Self { params }
    }

    fn key(&self) -> u64 {
        self.params.instance_id
    }

    fn build_vertices(&self, viewport: &Viewport) -> [Vertex; 6] {
        let clip = ClipTransform::from_viewport(viewport);
        let bounds = self.params.bounds;
        let left = bounds.x;
        let right = bounds.x + bounds.width.max(1.0);
        let top = bounds.y;
        let bottom = bounds.y + bounds.height.max(1.0);

        [
            Vertex {
                position: clip.to_clip(left, top),
                tex_coords: [0.0, 0.0],
            },
            Vertex {
                position: clip.to_clip(right, top),
                tex_coords: [1.0, 0.0],
            },
            Vertex {
                position: clip.to_clip(right, bottom),
                tex_coords: [1.0, 1.0],
            },
            Vertex {
                position: clip.to_clip(left, top),
                tex_coords: [0.0, 0.0],
            },
            Vertex {
                position: clip.to_clip(right, bottom),
                tex_coords: [1.0, 1.0],
            },
            Vertex {
                position: clip.to_clip(left, bottom),
                tex_coords: [0.0, 1.0],
            },
        ]
    }
}

impl Primitive for SpectrogramPrimitive {
    type Pipeline = Pipeline;

    fn prepare(
        &self,
        pipeline: &mut Self::Pipeline,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _bounds: &Rectangle,
        viewport: &Viewport,
    ) {
        let params = &self.params;

        if params.texture_width == 0 || params.texture_height == 0 {
            pipeline.prepare_instance(device, queue, self.key(), None, params);
            return;
        }

        let vertices = self.build_vertices(viewport);
        pipeline.prepare_instance(device, queue, self.key(), Some(&vertices), params);
    }

    fn render(
        &self,
        pipeline: &Self::Pipeline,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clip_bounds: &Rectangle<u32>,
    ) {
        let Some(instance) = pipeline.instance(self.key()) else {
            return;
        };

        let Some(resources) = instance.resources() else {
            return;
        };

        if instance.vertex_count() == 0 {
            return;
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Spectrogram pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
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

        pass.set_pipeline(pipeline.render_pipeline());
        pass.set_bind_group(0, resources.bind_group(), &[]);
        pass.set_vertex_buffer(0, instance.vertex_buffer_slice());
        pass.draw(0..instance.vertex_count(), 0..1);
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
struct Vertex {
    position: [f32; 2],
    tex_coords: [f32; 2],
}

const VERTEX_SIZE: wgpu::BufferAddress = std::mem::size_of::<Vertex>() as wgpu::BufferAddress;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
struct SpectrogramUniforms {
    dims_wrap_flags: [f32; 4],
    latest_and_count: [u32; 4],
    style: [f32; 4],
    background: [f32; 4],
}

impl SpectrogramUniforms {
    fn new(params: &SpectrogramParams) -> Self {
        let capacity = params.texture_width;
        let is_pow2 = capacity > 0 && capacity.is_power_of_two();
        let (wrap_mask, flags) = if is_pow2 {
            (capacity - 1, FLAG_CAPACITY_POW2)
        } else {
            (0, 0)
        };

        Self {
            dims_wrap_flags: [
                params.texture_width as f32,
                params.texture_height as f32,
                f32::from_bits(wrap_mask),
                f32::from_bits(flags),
            ],
            latest_and_count: [params.latest_column, params.column_count, 0, 0],
            style: [params.contrast.max(0.01), 0.0, 0.0, 0.0],
            background: params.background,
        }
    }
}

const VERTEX_LABEL: &str = "Spectrogram quad buffer";

pub struct Pipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    instances: HashMap<u64, Instance>,
    cache: CacheTracker,
}

impl primitive::Pipeline for Pipeline {
    fn new(device: &wgpu::Device, _queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let shader = create_shader_module(
            device,
            "Spectrogram shader",
            include_str!("shaders/spectrogram.wgsl"),
        );

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Spectrogram bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D1,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Spectrogram pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Spectrogram pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
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
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
            instances: HashMap::new(),
            cache: CacheTracker::default(),
        }
    }
}

impl Pipeline {
    fn prepare_instance(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        key: u64,
        vertices: Option<&[Vertex; 6]>,
        params: &SpectrogramParams,
    ) {
        let (frame, threshold) = self.cache.advance();

        let entry = self
            .instances
            .entry(key)
            .or_insert_with(|| Instance::new(device));

        entry.last_used = frame;

        entry.update_vertices(device, queue, vertices);
        entry.update_resources(device, queue, &self.bind_group_layout, params);
        self.prune(threshold);
    }

    fn instance(&self, key: u64) -> Option<&Instance> {
        self.instances.get(&key)
    }

    fn render_pipeline(&self) -> &wgpu::RenderPipeline {
        &self.pipeline
    }

    fn prune(&mut self, threshold: Option<u64>) {
        if let Some(threshold) = threshold {
            self.instances
                .retain(|_, instance| instance.last_used >= threshold);
        }
    }
}

impl Vertex {
    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: VERTEX_SIZE,
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
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

struct Instance {
    vertices: InstanceBuffer<Vertex>,
    resources: Option<GpuResources>,
    last_used: u64,
}

impl Instance {
    fn new(device: &wgpu::Device) -> Self {
        Self {
            vertices: InstanceBuffer::new(device, VERTEX_LABEL, VERTEX_SIZE.max(1)),
            resources: None,
            last_used: 0,
        }
    }

    fn update_vertices(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vertices: Option<&[Vertex; 6]>,
    ) {
        let Some(vertices) = vertices else {
            self.vertices.vertex_count = 0;
            return;
        };

        let required = VERTEX_SIZE * vertices.len() as wgpu::BufferAddress;
        self.vertices
            .ensure_capacity(device, VERTEX_LABEL, required);
        self.vertices.write(queue, vertices);
    }

    fn update_resources(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        params: &SpectrogramParams,
    ) {
        if params.texture_width == 0 || params.texture_height == 0 {
            self.resources = None;
            return;
        }

        if self.resources.is_none() {
            self.resources = Some(GpuResources::new(
                device,
                layout,
                params.texture_width,
                params.texture_height,
            ));
        }

        if let Some(resources) = self.resources.as_mut() {
            resources.ensure_capacity(
                device,
                queue,
                layout,
                params.texture_width,
                params.texture_height,
            );
            resources.write(queue, params);
        }
    }

    fn vertex_buffer_slice(&self) -> wgpu::BufferSlice<'_> {
        self.vertices
            .vertex_buffer
            .slice(0..self.vertices.used_bytes())
    }

    fn vertex_count(&self) -> u32 {
        self.vertices.vertex_count
    }

    fn resources(&self) -> Option<&GpuResources> {
        self.resources.as_ref()
    }
}
struct GpuResources {
    uniforms: UniformBuffer,
    magnitude: MagnitudeResources,
    palette: PaletteResources,
    bind_group: wgpu::BindGroup,
    uniform_cache: SpectrogramUniforms,
    palette_cache: [[f32; 4]; SPECTROGRAM_PALETTE_SIZE],
}

impl GpuResources {
    fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, width: u32, height: u32) -> Self {
        let uniforms = UniformBuffer::new(device);
        let magnitude = MagnitudeResources::new(device, width, height);
        let palette = PaletteResources::new(device);

        let bind_group = create_bind_group(
            device,
            layout,
            uniforms.buffer(),
            magnitude.view(),
            palette.view(),
            palette.sampler(),
        );

        Self {
            uniforms,
            magnitude,
            palette,
            bind_group,
            uniform_cache: SpectrogramUniforms {
                dims_wrap_flags: [
                    width as f32,
                    height as f32,
                    f32::from_bits(0),
                    f32::from_bits(0),
                ],
                latest_and_count: [0, 0, 0, 0],
                style: [1.0, 0.0, 0.0, 0.0],
                background: [0.0; 4],
            },
            palette_cache: [[0.0; 4]; SPECTROGRAM_PALETTE_SIZE],
        }
    }

    fn ensure_capacity(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        width: u32,
        height: u32,
    ) {
        if let Some((old_texture, old_capacity)) =
            self.magnitude.ensure_capacity(device, width, height)
        {
            self.rebuild_bind_group(device, layout);

            let old_extent = magnitude_extent(old_capacity.0, old_capacity.1);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Spectrogram magnitude grow copy"),
            });

            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &old_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: self.magnitude.texture(),
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                old_extent,
            );

            queue.submit(Some(encoder.finish()));
        }
    }

    fn write(&mut self, queue: &wgpu::Queue, params: &SpectrogramParams) {
        let width = params.texture_width.min(self.magnitude.capacity().0);
        let height = params.texture_height.min(self.magnitude.capacity().1);
        if width == 0 || height == 0 {
            return;
        }

        if let Some(base) = &params.base_data {
            let column_stride = height as usize;
            debug_assert_eq!(
                base.len(),
                (params.texture_width * params.texture_height) as usize
            );
            for (column, values) in base.chunks(column_stride).enumerate().take(width as usize) {
                write_column(
                    queue,
                    self.magnitude.texture(),
                    column as u32,
                    height,
                    values,
                );
            }
        }

        for update in &params.column_updates {
            if update.values.as_slice().is_empty() {
                continue;
            }

            let column = update.column_index.min(width.saturating_sub(1));
            write_column(
                queue,
                self.magnitude.texture(),
                column,
                height,
                update.values.as_slice(),
            );
        }

        let uniforms = SpectrogramUniforms::new(params);
        if uniforms != self.uniform_cache {
            queue.write_buffer(self.uniforms.buffer(), 0, bytemuck::bytes_of(&uniforms));
            self.uniform_cache = uniforms;
        }

        if params.palette != self.palette_cache {
            let lut = populate_palette_lut(&params.palette);
            write_texture_region(
                queue,
                self.palette.texture(),
                wgpu::Origin3d::ZERO,
                wgpu::Extent3d {
                    width: PALETTE_LUT_SIZE,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                PALETTE_LUT_SIZE * 4,
                &lut,
            );
            self.palette_cache = params.palette;
        }
    }

    fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    fn rebuild_bind_group(&mut self, device: &wgpu::Device, layout: &wgpu::BindGroupLayout) {
        self.bind_group = create_bind_group(
            device,
            layout,
            self.uniforms.buffer(),
            self.magnitude.view(),
            self.palette.view(),
            self.palette.sampler(),
        );
    }
}

struct UniformBuffer {
    buffer: wgpu::Buffer,
}

impl UniformBuffer {
    fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spectrogram uniform buffer"),
            size: std::mem::size_of::<SpectrogramUniforms>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { buffer }
    }

    fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

struct PaletteResources {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
}

impl PaletteResources {
    fn new(device: &wgpu::Device) -> Self {
        let texture = create_palette_texture(device);
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Spectrogram palette view"),
            format: Some(wgpu::TextureFormat::Rgba8Unorm),
            dimension: Some(wgpu::TextureViewDimension::D1),
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Spectrogram palette sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        Self {
            texture,
            view,
            sampler,
        }
    }

    fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }
}

struct MagnitudeResources {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    capacity: (u32, u32),
}

impl MagnitudeResources {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let texture = create_magnitude_texture(device, width.max(1), height.max(1));
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Spectrogram magnitude view"),
            format: Some(wgpu::TextureFormat::R32Float),
            dimension: Some(wgpu::TextureViewDimension::D2),
            ..Default::default()
        });
        Self {
            texture,
            view,
            capacity: (width.max(1), height.max(1)),
        }
    }

    fn ensure_capacity(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> Option<(wgpu::Texture, (u32, u32))> {
        let target = (width.max(1), height.max(1));
        if target.0 <= self.capacity.0 && target.1 <= self.capacity.1 {
            return None;
        }

        let new_capacity = (target.0.max(self.capacity.0), target.1.max(self.capacity.1));
        let new_texture = create_magnitude_texture(device, new_capacity.0, new_capacity.1);
        let new_view = new_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Spectrogram magnitude view"),
            format: Some(wgpu::TextureFormat::R32Float),
            dimension: Some(wgpu::TextureViewDimension::D2),
            ..Default::default()
        });

        let old_capacity = std::mem::replace(&mut self.capacity, new_capacity);
        let old_texture = std::mem::replace(&mut self.texture, new_texture);
        self.view = new_view;

        Some((old_texture, old_capacity))
    }

    fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    fn capacity(&self) -> (u32, u32) {
        self.capacity
    }
}

fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
    magnitude_view: &wgpu::TextureView,
    palette_view: &wgpu::TextureView,
    palette_sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Spectrogram bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: uniform_buffer,
                    offset: 0,
                    size: None,
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(magnitude_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(palette_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(palette_sampler),
            },
        ],
    })
}

fn create_magnitude_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Spectrogram magnitude texture"),
        size: magnitude_extent(width, height),
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[wgpu::TextureFormat::R32Float],
    })
}

fn create_palette_texture(device: &wgpu::Device) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Spectrogram palette texture"),
        size: wgpu::Extent3d {
            width: PALETTE_LUT_SIZE,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D1,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
    })
}

fn write_column(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    column: u32,
    height: u32,
    values: &[f32],
) {
    if height == 0 {
        return;
    }

    debug_assert_eq!(values.len(), height as usize);

    write_texture_region(
        queue,
        texture,
        wgpu::Origin3d {
            x: 0,
            y: column,
            z: 0,
        },
        column_extent(height),
        height * std::mem::size_of::<f32>() as u32,
        bytemuck::cast_slice(values),
    );
}

fn populate_palette_lut(palette: &[[f32; 4]; SPECTROGRAM_PALETTE_SIZE]) -> Vec<u8> {
    let size = PALETTE_LUT_SIZE as usize;
    let mut data = vec![0u8; size * 4];

    if PALETTE_LUT_SIZE == 0 {
        return data;
    }

    let max_index = (PALETTE_LUT_SIZE - 1) as f32;
    let segments = (SPECTROGRAM_PALETTE_SIZE - 1) as f32;

    for (i, chunk) in data.chunks_exact_mut(4).enumerate() {
        let t = if PALETTE_LUT_SIZE == 1 {
            0.0
        } else {
            i as f32 / max_index
        };
        let scaled = t * segments;
        let index = scaled.floor() as usize;
        let next = (index + 1).min(SPECTROGRAM_PALETTE_SIZE - 1);
        let frac = scaled - index as f32;

        for (channel, byte) in chunk.iter_mut().enumerate() {
            let value =
                palette[index][channel] + (palette[next][channel] - palette[index][channel]) * frac;
            *byte = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }

    data
}

fn magnitude_extent(width: u32, height: u32) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: height.max(1),
        height: width.max(1),
        depth_or_array_layers: 1,
    }
}

fn column_extent(height: u32) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: height.max(1),
        height: 1,
        depth_or_array_layers: 1,
    }
}
