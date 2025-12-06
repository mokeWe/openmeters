struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) color : vec4<f32>,
};

fn premultiply(color: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(color.rgb * color.a, color.a);
}

@vertex
fn vs_main(
    @location(0) position : vec2<f32>,
    @location(1) color : vec4<f32>,
) -> VertexOutput {
    var out : VertexOutput;
    out.position = vec4<f32>(position, 0.0, 1.0);
    out.color = premultiply(color);
    return out;
}

@fragment
fn fs_main(in : VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
