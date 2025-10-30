struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) params: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) params: vec3<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4<f32>(input.position, 0.0, 1.0);
    output.color = input.color;
    output.params = input.params.xyz;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let signed_distance = input.params.x;
    let inner = input.params.y;
    let feather = max(input.params.z, 1.0e-4);
    
    // If inner and feather are both zero, this is a filled region (highlight column)
    // so just return the color as-is
    if inner == 0.0 && feather == 1.0e-4 {
        return input.color;
    }
    
    // Otherwise, apply antialiasing for the line
    let dist = abs(signed_distance);
    let coverage = clamp((inner + feather - dist) / feather, 0.0, 1.0);
    let alpha = input.color.a * coverage;
    return vec4<f32>(input.color.rgb, alpha);
}
