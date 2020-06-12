#version 450

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec2 v_tex_uv;
layout(location = 2) in vec2 v_lightmap_uv;
layout(location = 3) in vec3 v_lightgrid_uv;
layout(location = 4) in vec3 v_view_ray;

layout(location = 0) out vec4 f_color;

layout(set = 1, binding = 0) uniform sampler2D mainTex;
layout(set = 1, binding = 1) uniform sampler2D lightmapTex;

void main() {
    vec4 texColor = texture(mainTex, v_tex_uv);
    vec4 lightmapColor = texture(lightmapTex, v_lightmap_uv);

    //f_color = lightmapColor;   // Just the lightmap
    //f_color = vec4((normalize(v_normal) + vec3(1, 1, 1)) * 0.5, 1.0);    // World-space normals
    f_color = texColor * lightmapColor;
}
