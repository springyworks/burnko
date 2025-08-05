// Jet colormap function for WGSL
// Classic blue-cyan-yellow-red colormap

fn jet_colormap(t: f32) -> vec3<f32> {
    let t_clamped = clamp(t, 0.0, 1.0);
    
    let r = clamp(1.5 - abs(4.0 * t_clamped - 3.0), 0.0, 1.0);
    let g = clamp(1.5 - abs(4.0 * t_clamped - 2.0), 0.0, 1.0);
    let b = clamp(1.5 - abs(4.0 * t_clamped - 1.0), 0.0, 1.0);
    
    return vec3<f32>(r, g, b);
}
