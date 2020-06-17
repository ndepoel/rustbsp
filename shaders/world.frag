#version 450

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec2 v_tex_uv;
layout(location = 2) in vec2 v_lightmap_uv;
layout(location = 3) in vec3 v_lightgrid_uv;
layout(location = 4) in vec3 v_view_ray;

layout(location = 0) out vec4 f_color;

layout(set = 1, binding = 0) uniform sampler2D mainTex;
layout(set = 1, binding = 1) uniform sampler2D lightmapTex;

layout(constant_id = 0) const bool alpha_mask = false;
layout(constant_id = 1) const float alpha_offset = 0.0;
layout(constant_id = 2) const bool alpha_invert = false;

void main() {
    vec4 texColor = texture(mainTex, v_tex_uv);
    if (alpha_mask)
    {
        float alpha = texColor.a + alpha_offset;
        if (alpha_invert) alpha = 1.0 - alpha;
        if (alpha < 0.5)
            discard;
    }

    vec4 lightmapColor = texture(lightmapTex, v_lightmap_uv);

    //f_color = lightmapColor;   // Just the lightmap
    //f_color = vec4((normalize(v_normal) + vec3(1, 1, 1)) * 0.5, 1.0);    // World-space normals
    f_color = texColor * lightmapColor;
}
