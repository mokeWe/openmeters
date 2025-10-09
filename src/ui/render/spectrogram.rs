use bytemuck::{Pod, Zeroable};
use iced::Rectangle;
use iced::advanced::graphics::Viewport;
use iced_wgpu::primitive::{Primitive, Storage};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

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
    pub base_data: Option<Arc<[f32]>>,
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
            buffers: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn acquire(&self, len: usize) -> Vec<f32> {
        let mut buffers = self.buffers.lock().unwrap();
        let mut buffer = buffers.pop().unwrap_or_else(|| Vec::with_capacity(len));
        if buffer.capacity() < len {
            buffer.reserve(len - buffer.capacity());
        }
        buffer.resize(len, 0.0);
        buffer
    }

    pub fn release(&self, mut buffer: Vec<f32>) {
        buffer.clear();
        self.buffers.lock().unwrap().push(buffer);
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
        let logical_size = viewport.logical_size();
        let width = logical_size.width.max(1.0);
        let height = logical_size.height.max(1.0);
        let scale_x = 2.0 / width;
        let scale_y = 2.0 / height;
        let to_clip = |x: f32, y: f32| [x * scale_x - 1.0, 1.0 - y * scale_y];
        let bounds = self.params.bounds;

        let left = bounds.x;
        let right = bounds.x + bounds.width.max(1.0);
        let top = bounds.y;
        let bottom = bounds.y + bounds.height.max(1.0);

        [
            Vertex {
                position: to_clip(left, top),
                tex_coords: [0.0, 0.0],
            },
            Vertex {
                position: to_clip(right, top),
                tex_coords: [1.0, 0.0],
            },
            Vertex {
                position: to_clip(right, bottom),
                tex_coords: [1.0, 1.0],
            },
            Vertex {
                position: to_clip(left, top),
                tex_coords: [0.0, 0.0],
            },
            Vertex {
                position: to_clip(right, bottom),
                tex_coords: [1.0, 1.0],
            },
            Vertex {
                position: to_clip(left, bottom),
                tex_coords: [0.0, 1.0],
            },
        ]
    }
}

impl Primitive for SpectrogramPrimitive {
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

        let params = &self.params;
        let Some(pipeline) = storage.get_mut::<Pipeline>() else {
            return;
        };

        if params.texture_width == 0 || params.texture_height == 0 {
            pipeline.prepare_instance(device, queue, self.key(), None, params);
            return;
        }

        let vertices = self.build_vertices(viewport);
        pipeline.prepare_instance(device, queue, self.key(), Some(&vertices), params);
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

struct Pipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    instances: HashMap<u64, Instance>,
}

impl Pipeline {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Spectrogram shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/spectrogram.wgsl").into()),
        });

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
            bind_group_layout,
            instances: HashMap::new(),
        }
    }

    fn prepare_instance(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        key: u64,
        vertices: Option<&[Vertex; 6]>,
        params: &SpectrogramParams,
    ) {
        let entry = self
            .instances
            .entry(key)
            .or_insert_with(|| Instance::new(device));

        entry.update_vertices(device, queue, vertices);
        entry.update_resources(device, queue, &self.bind_group_layout, params);
    }

    fn instance(&self, key: u64) -> Option<&Instance> {
        self.instances.get(&key)
    }

    fn render_pipeline(&self) -> &wgpu::RenderPipeline {
        &self.pipeline
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
    vertex_buffer: wgpu::Buffer,
    capacity: wgpu::BufferAddress,
    vertex_count: u32,
    resources: Option<GpuResources>,
    cached_vertices: Option<[Vertex; 6]>,
}

impl Instance {
    fn new(device: &wgpu::Device) -> Self {
        let buffer = create_vertex_buffer(device, 1);
        Self {
            vertex_buffer: buffer,
            capacity: 1,
            vertex_count: 0,
            resources: None,
            cached_vertices: None,
        }
    }

    fn update_vertices(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vertices: Option<&[Vertex; 6]>,
    ) {
        let Some(vertices) = vertices else {
            self.vertex_count = 0;
            self.cached_vertices = None;
            return;
        };

        if self.cached_vertices == Some(*vertices) {
            return;
        }

        let required = VERTEX_SIZE * vertices.len() as wgpu::BufferAddress;
        if required > self.capacity {
            let new_capacity = required.next_power_of_two().max(1);
            self.vertex_buffer = create_vertex_buffer(device, new_capacity);
            self.capacity = new_capacity;
        }

        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(vertices));
        self.vertex_count = vertices.len() as u32;
        self.cached_vertices = Some(*vertices);
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
        let used = VERTEX_SIZE * self.vertex_count as wgpu::BufferAddress;
        self.vertex_buffer.slice(0..used)
    }

    fn vertex_count(&self) -> u32 {
        self.vertex_count
    }

    fn resources(&self) -> Option<&GpuResources> {
        self.resources.as_ref()
    }
}
struct GpuResources {
    uniform_buffer: wgpu::Buffer,
    magnitude_texture: wgpu::Texture,
    magnitude_view: wgpu::TextureView,
    palette_texture: wgpu::Texture,
    palette_view: wgpu::TextureView,
    palette_sampler: wgpu::Sampler,
    bind_group: wgpu::BindGroup,
    capacity: (u32, u32),
    uniform_cache: SpectrogramUniforms,
    palette_cache: [[f32; 4]; SPECTROGRAM_PALETTE_SIZE],
}

impl GpuResources {
    fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, width: u32, height: u32) -> Self {
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spectrogram uniform buffer"),
            size: std::mem::size_of::<SpectrogramUniforms>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let magnitude_texture = create_magnitude_texture(device, width.max(1), height.max(1));
        let magnitude_view = magnitude_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Spectrogram magnitude view"),
            format: Some(wgpu::TextureFormat::R32Float),
            dimension: Some(wgpu::TextureViewDimension::D2),
            ..Default::default()
        });

        let palette_texture = create_palette_texture(device);
        let palette_view = palette_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Spectrogram palette view"),
            format: Some(wgpu::TextureFormat::Rgba8Unorm),
            dimension: Some(wgpu::TextureViewDimension::D1),
            ..Default::default()
        });

        let palette_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Spectrogram palette sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let bind_group = create_bind_group(
            device,
            layout,
            &uniform_buffer,
            &magnitude_view,
            &palette_view,
            &palette_sampler,
        );

        Self {
            uniform_buffer,
            magnitude_texture,
            magnitude_view,
            palette_texture,
            palette_view,
            palette_sampler,
            bind_group,
            capacity: (width.max(1), height.max(1)),
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
        let target = (width.max(1), height.max(1));
        if target.0 <= self.capacity.0 && target.1 <= self.capacity.1 {
            return;
        }

        let new_capacity = (target.0.max(self.capacity.0), target.1.max(self.capacity.1));
        let new_texture = create_magnitude_texture(device, new_capacity.0, new_capacity.1);
        let new_view = new_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Spectrogram magnitude view"),
            format: Some(wgpu::TextureFormat::R32Float),
            dimension: Some(wgpu::TextureViewDimension::D2),
            ..Default::default()
        });

        let old_capacity = self.capacity;
        let old_texture = std::mem::replace(&mut self.magnitude_texture, new_texture);
        self.magnitude_view = new_view;

        self.bind_group = create_bind_group(
            device,
            layout,
            &self.uniform_buffer,
            &self.magnitude_view,
            &self.palette_view,
            &self.palette_sampler,
        );

        let old_extent = magnitude_extent(old_capacity.0, old_capacity.1);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Spectrogram magnitude grow copy"),
        });

        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: &old_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyTexture {
                texture: &self.magnitude_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            old_extent,
        );

        queue.submit(Some(encoder.finish()));

        self.capacity = new_capacity;
    }

    fn write(&mut self, queue: &wgpu::Queue, params: &SpectrogramParams) {
        let width = params.texture_width.min(self.capacity.0);
        let height = params.texture_height.min(self.capacity.1);
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
                    &self.magnitude_texture,
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
                &self.magnitude_texture,
                column,
                height,
                update.values.as_slice(),
            );
        }

        let uniforms = SpectrogramUniforms::new(params);
        if uniforms != self.uniform_cache {
            queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
            self.uniform_cache = uniforms;
        }

        if params.palette != self.palette_cache {
            let lut = populate_palette_lut(&params.palette);
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.palette_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &lut,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(PALETTE_LUT_SIZE * 4),
                    rows_per_image: None,
                },
                wgpu::Extent3d {
                    width: PALETTE_LUT_SIZE,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            self.palette_cache = params.palette;
        }
    }

    fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
}

fn create_vertex_buffer(device: &wgpu::Device, size: wgpu::BufferAddress) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Spectrogram vertex buffer"),
        size,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
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
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
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

    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: 0,
                y: column,
                z: 0,
            },
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(values),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(height * std::mem::size_of::<f32>() as u32),
            rows_per_image: None,
        },
        column_extent(height),
    );
}

fn populate_palette_lut(palette: &[[f32; 4]; SPECTROGRAM_PALETTE_SIZE]) -> Vec<u8> {
    let mut data = vec![0u8; (PALETTE_LUT_SIZE as usize) * 4];
    if PALETTE_LUT_SIZE == 0 {
        return data;
    }

    let max_index = (PALETTE_LUT_SIZE.saturating_sub(1)) as f32;
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
            let start = palette[index][channel];
            let stop = palette[next][channel];
            let value = start + (stop - start) * frac;
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
