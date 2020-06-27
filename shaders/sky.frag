#version 450

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec2 v_tex_uv;
layout(location = 2) in vec2 v_lightmap_uv;
layout(location = 3) in vec3 v_lightgrid_uv;
layout(location = 4) in vec3 v_view_ray;

layout(location = 0) out vec4 f_color;

layout(set = 1, binding = 0) uniform sampler2D mainTex;

layout(push_constant) uniform VertexMods
{
    vec3 tcmod_u;
    vec3 tcmod_v;
} pc;

vec2 vec_to_latlng(vec3 v)
{
    v = normalize(v);
    float lat = 0.5 + atan(v.y, v.x) / (2.0 * 3.14159265);
    float lng = 0.5 + asin(v.z) / 3.14159265;
    return vec2(lat, lng);
}

void main() {
    // Convert world-space ray to spherical coordinates for a skydome effect
    vec2 uv = vec_to_latlng(v_view_ray);
    
    // Modify sky texture coords to create a scrolling effect
    vec3 uv_dir = vec3(uv - 0.5, 1);
    uv = vec2(dot(uv_dir, pc.tcmod_u), dot(uv_dir, pc.tcmod_v)) + 0.5;

    f_color = texture(mainTex, -uv);
}
