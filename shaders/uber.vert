#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texture_coord;
layout(location = 2) in vec2 lightmap_coord;
layout(location = 3) in vec3 normal;

layout(location = 0) out vec3 v_normal; // World-space normal
layout(location = 1) out vec2 v_tex_uv;
layout(location = 2) out vec2 v_lightmap_uv;
layout(location = 3) out vec3 v_lightgrid_uv;
layout(location = 4) out vec3 v_view_ray;

layout(set = 0, binding = 0) uniform Data {
    mat4 model;
    mat4 view;
    mat4 proj;
    mat4 lightgrid;
} uniforms;

layout(push_constant) uniform VertexMods
{
    float tc_rotate;
    vec2 tc_scroll;
    vec2 tc_scale;
} pc;

void main() {
    vec4 worldpos = uniforms.model * vec4(position, 1.0);
    vec4 viewpos = uniforms.view * worldpos;
    gl_Position = uniforms.proj * viewpos;

    v_normal = transpose(inverse(mat3(uniforms.model))) * normal;

    // Apply texture coordinate rotation
    float sin_rot = sin(pc.tc_rotate);
    float cos_rot = cos(pc.tc_rotate);
    vec2 uv_dir = texture_coord - vec2(0.5, 0.5);
    v_tex_uv.x = cos_rot * uv_dir.x - sin_rot * uv_dir.y;
    v_tex_uv.y = sin_rot * uv_dir.x + cos_rot * uv_dir.y;
    v_tex_uv = vec2(0.5, 0.5) + v_tex_uv;

    v_tex_uv = (v_tex_uv + pc.tc_scroll) * pc.tc_scale;
    v_lightmap_uv = lightmap_coord;
    v_lightgrid_uv = (uniforms.lightgrid * worldpos).xyz;

    // Transform the vertex position into view space, subtract the camera position from it (which is 0 in view space),
    // giving the view ray in view space, then rotate the ray back to world space.
    v_view_ray = transpose(mat3(uniforms.view)) * viewpos.xyz;
}
