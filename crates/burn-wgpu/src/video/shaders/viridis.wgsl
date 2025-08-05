// Viridis colormap function for WGSL
// Converts a normalized value [0,1] to Viridis color

fn viridis_colormap(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.2777273, 0.0054374, 0.3340998);
    let c1 = vec3<f32>(0.1050930, 1.4042440, 0.5581840);
    let c2 = vec3<f32>(-0.3308618, 0.2148030, 1.1487750);
    let c3 = vec3<f32>(-4.6340520, -5.7998330, -19.3396750);
    let c4 = vec3<f32>(6.2288510, 14.1798870, 56.6904410);
    let c5 = vec3<f32>(-5.7434270, -13.7441770, -65.3532160);
    let c6 = vec3<f32>(1.6810260, 4.9710710, 26.3124520);

    let t_clamped = clamp(t, 0.0, 1.0);
    return c0 + t_clamped * (c1 + t_clamped * (c2 + t_clamped * (c3 + t_clamped * (c4 + t_clamped * (c5 + t_clamped * c6)))));
}
