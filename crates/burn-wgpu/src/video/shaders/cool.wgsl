// Cool colormap function for WGSL
// Creates cool colors from cyan to magenta

fn cool_colormap(t: f32) -> vec3<f32> {
    let t_clamped = clamp(t, 0.0, 1.0);
    return vec3<f32>(t_clamped, 1.0 - t_clamped, 1.0);
}
