struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coords: vec2<f32>,
}

;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

;

struct SpectrogramUniforms {
    dims_wrap_flags: vec4<f32>,
    latest_and_count: vec4<u32>,
    style: vec4<f32>,
    background: vec4<f32>,
}

;

struct MagnitudeParams {
    capacity: u32,
    wrap_mask: u32,
    oldest: u32,
    is_pow2: bool,
    is_full: bool,
}

;

const FLAG_CAPACITY_POW2: u32 = 0x1u;
const SHARPEN_STRENGTH: f32 = 0.55;
const SIGMA_HORIZONTAL: f32 = 0.14;
const SIGMA_VERTICAL: f32 = 0.10;
const SIGMA_DIAGONAL: f32 = 0.18;
const VARIANCE_CLAMP: f32 = 0.02;
const DIAGONAL_SPATIAL_WEIGHT: f32 = 0.70710677;
// 1 / sqrt(2)

const INV_SIGMA_HORIZONTAL: f32 = 1.0 / SIGMA_HORIZONTAL;
const INV_SIGMA_VERTICAL: f32 = 1.0 / SIGMA_VERTICAL;
const INV_SIGMA_DIAGONAL: f32 = 1.0 / SIGMA_DIAGONAL;
const INV_VARIANCE_CLAMP: f32 = 1.0 / VARIANCE_CLAMP;

@group(0) @binding(0)
var<uniform> uniforms: SpectrogramUniforms;
@group(0) @binding(1)
var magnitudes: texture_2d<f32>;
@group(0) @binding(2)
var palette_tex: texture_1d<f32>;
@group(0) @binding(3)
var palette_sampler: sampler;

fn sample_palette(value: f32) -> vec4<f32> {
    let clamped = clamp(value, 0.0, 1.0);
    let contrast = max(uniforms.style.x, 0.01);
    let adjusted = pow(clamped, contrast);
    return textureSampleLevel(palette_tex, palette_sampler, adjusted, 0.0);
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4<f32>(input.position, 0.0, 1.0);
    output.tex_coords = input.tex_coords;
    return output;
}

fn logical_to_physical(logical: u32, params: MagnitudeParams) -> u32 {
    if params.is_full {
        if params.is_pow2 {
            return (params.oldest + logical) & params.wrap_mask;
        }
        return (params.oldest + logical) % params.capacity;
    }
    return logical;
}

fn sample_magnitude(logical: u32, row: u32, params: MagnitudeParams) -> f32 {
    let physical = logical_to_physical(logical, params);
    return textureLoad(magnitudes, vec2<i32>(i32(row), i32(physical)), 0).x;
}

fn bilateral_weight(delta: f32, inv_sigma: f32, spatial_scale: f32) -> f32 {
    let ratio = delta * inv_sigma;
    return spatial_scale * exp(- ratio * ratio);
}

fn accumulate(
    enabled: bool,
    logical: u32,
    row: u32,
    params: MagnitudeParams,
    inv_sigma: f32,
    spatial_scale: f32,
    center: f32,
    accum: vec3<f32>,
) -> vec3<f32> {
    if !enabled {
        return accum;
    }
    let value = sample_magnitude(logical, row, params);
    let weight = bilateral_weight(abs(center - value), inv_sigma, spatial_scale);
    return accum + vec3<f32>(value * weight, value * value * weight, weight);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let dims = uniforms.dims_wrap_flags;
    let capacity = u32(dims.x);
    let height = u32(dims.y);
    let wrap_mask = bitcast<u32>(dims.z);
    let flags = bitcast<u32>(dims.w);

    let state = uniforms.latest_and_count;
    let count = state.y;

    if capacity == 0u || height == 0u || count == 0u {
        return uniforms.background;
    }

    let latest = min(state.x, capacity - 1u);

    let clamped_uv = clamp(input.tex_coords, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));
    let logical_width = count;

    var x_index: u32 = 0u;
    if logical_width > 1u {
        x_index = min(u32(clamped_uv.x * f32(logical_width - 1u) + 0.5), logical_width - 1u);
    }

    let is_pow2 = (flags & FLAG_CAPACITY_POW2) != 0u;
    let is_full = count == capacity;

    var oldest = 0u;
    if is_full {
        let next = latest + 1u;
        if is_pow2 {
            oldest = next & wrap_mask;
        }
        else {
            oldest = next % capacity;
        }
    }

    var row: u32 = 0u;
    if height > 1u {
        row = min(u32(clamped_uv.y * f32(height - 1u) + 0.5), height - 1u);
    }

    let params = MagnitudeParams(capacity, wrap_mask, oldest, is_pow2, is_full);

    let center = sample_magnitude(x_index, row, params);

    let has_left = x_index > 0u;
    let left_logical = select(x_index, x_index - 1u, has_left);
    let has_right = x_index + 1u < logical_width;
    let right_logical = select(x_index, x_index + 1u, has_right);

    let has_up = row > 0u;
    let up_row = select(row, row - 1u, has_up);
    let has_down = row + 1u < height;
    let down_row = select(row, row + 1u, has_down);

    var accum = vec3<f32>(center, center * center, 1.0);
    accum = accumulate(has_left, left_logical, row, params, INV_SIGMA_HORIZONTAL, 1.0, center, accum);
    accum = accumulate(has_right, right_logical, row, params, INV_SIGMA_HORIZONTAL, 1.0, center, accum);
    accum = accumulate(has_up, x_index, up_row, params, INV_SIGMA_VERTICAL, 1.0, center, accum);
    accum = accumulate(has_down, x_index, down_row, params, INV_SIGMA_VERTICAL, 1.0, center, accum);
    accum = accumulate(has_left && has_up, left_logical, up_row, params, INV_SIGMA_DIAGONAL, DIAGONAL_SPATIAL_WEIGHT, center, accum);
    accum = accumulate(has_right && has_up, right_logical, up_row, params, INV_SIGMA_DIAGONAL, DIAGONAL_SPATIAL_WEIGHT, center, accum);
    accum = accumulate(has_left && has_down, left_logical, down_row, params, INV_SIGMA_DIAGONAL, DIAGONAL_SPATIAL_WEIGHT, center, accum);
    accum = accumulate(has_right && has_down, right_logical, down_row, params, INV_SIGMA_DIAGONAL, DIAGONAL_SPATIAL_WEIGHT, center, accum);

    let mean = accum.x / accum.z;
    let variance = max(accum.y / accum.z - mean * mean, 0.0);
    let detail = center - mean;
    let attenuation = clamp((VARIANCE_CLAMP - variance) * INV_VARIANCE_CLAMP, 0.0, 1.0);
    let sharpened = clamp(center + detail * SHARPEN_STRENGTH * attenuation, 0.0, 1.0);

    return sample_palette(sharpened);
}
