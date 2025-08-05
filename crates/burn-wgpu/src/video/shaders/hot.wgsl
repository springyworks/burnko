// Hot colormap function for WGSL
// Creates a heat map from black -> red -> yellow -> white

fn hot_colormap(t: f32) -> vec3<f32> {
    let t_clamped = clamp(t, 0.0, 1.0);
    
    if (t_clamped < 0.33333) {
        // Black to red
        let local_t = t_clamped * 3.0;
        return vec3<f32>(local_t, 0.0, 0.0);
    } else if (t_clamped < 0.66666) {
        // Red to yellow
        let local_t = (t_clamped - 0.33333) * 3.0;
        return vec3<f32>(1.0, local_t, 0.0);
    } else {
        // Yellow to white
        let local_t = (t_clamped - 0.66666) * 3.0;
        return vec3<f32>(1.0, 1.0, local_t);
    }
}
