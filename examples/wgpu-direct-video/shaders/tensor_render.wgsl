// Tensor rendering shader for WGPU Direct Video Pipeline
// Renders tensor data directly to screen with colormap support

// Vertex shader
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    
    // Full-screen quad vertices
    let x = f32(i32(vertex_index & 1u) * 2 - 1);
    let y = f32(i32(vertex_index & 2u) - 1);
    
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    out.tex_coords = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    
    return out;
}

// Fragment shader with colormap support
@group(0) @binding(0) var tensor_texture: texture_2d<f32>;
@group(0) @binding(1) var tensor_sampler: sampler;

// Colormap function - converts single value to RGB
fn apply_colormap(value: f32, colormap_type: u32) -> vec3<f32> {
    let v = clamp(value, 0.0, 1.0);
    
    switch colormap_type {
        case 0u: { // Viridis
            let c0 = vec3<f32>(0.267004, 0.004874, 0.329415);
            let c1 = vec3<f32>(0.127568, 0.566949, 0.550556);
            let c2 = vec3<f32>(0.993248, 0.906157, 0.143936);
            
            if (v < 0.5) {
                return mix(c0, c1, v * 2.0);
            } else {
                return mix(c1, c2, (v - 0.5) * 2.0);
            }
        }
        case 1u: { // Hot
            if (v < 0.33) {
                return vec3<f32>(v * 3.0, 0.0, 0.0);
            } else if (v < 0.66) {
                return vec3<f32>(1.0, (v - 0.33) * 3.0, 0.0);
            } else {
                return vec3<f32>(1.0, 1.0, (v - 0.66) * 3.0);
            }
        }
        case 2u: { // Cool
            return vec3<f32>(v, 1.0 - v, 1.0);
        }
        default: { // Grayscale
            return vec3<f32>(v, v, v);
        }
    }
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample tensor value
    let tensor_value = textureSample(tensor_texture, tensor_sampler, in.tex_coords).r;
    
    // Apply colormap (hardcoded to hot for now)
    let color = apply_colormap(tensor_value, 1u);
    
    return vec4<f32>(color, 1.0);
}
