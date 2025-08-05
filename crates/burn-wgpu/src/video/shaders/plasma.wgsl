// Plasma colormap function for WGSL
// High contrast purple-pink-yellow colormap

fn plasma_colormap(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.0585, 0.0078, 0.5206);
    let c1 = vec3<f32>(2.1760, 0.0424, 0.2339);
    let c2 = vec3<f32>(-2.6893, 1.7750, -3.5942);
    let c3 = vec3<f32>(6.1305, -7.1513, 15.9323);
    let c4 = vec3<f32>(-11.1086, 8.4173, -31.6769);
    let c5 = vec3<f32>(10.0623, -3.5668, 25.0109);
    let c6 = vec3<f32>(-3.6581, 0.3216, -6.2044);

    let t_clamped = clamp(t, 0.0, 1.0);
    return c0 + t_clamped * (c1 + t_clamped * (c2 + t_clamped * (c3 + t_clamped * (c4 + t_clamped * (c5 + t_clamped * c6)))));
}
