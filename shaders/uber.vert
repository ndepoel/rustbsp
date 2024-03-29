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
    float time;
    vec4 vertex_wave;   // x = div, y = base, z = amplitude, w = frequency
    vec3 tcmod_u;
    vec3 tcmod_v;
} pc;

layout(constant_id = 0) const bool apply_deformation = false;

void main() {
    vec3 pos = position;
    if (apply_deformation)
    {
        float phase = (pos.x + pos.y + pos.z) / pc.vertex_wave.x;
        pos = pos + normal * (pc.vertex_wave.y + pc.vertex_wave.z * sin(2.0 * 3.14159265 * pc.vertex_wave.w * pc.time + phase));
    }

    vec4 worldpos = uniforms.model * vec4(pos, 1.0);
    vec4 viewpos = uniforms.view * worldpos;
    gl_Position = uniforms.proj * viewpos;

    v_normal = transpose(inverse(mat3(uniforms.model))) * normal;

    // For rotation to work correctly, we need to move the texture coordinates around a pivot in the center of the texture (i.e. 0.5, 0.5)
    vec2 uv_pivot = vec2(0.5, 0.5);
    vec3 uv_dir = vec3(texture_coord - uv_pivot, 1);
    vec2 tc_scale = vec2(length(vec2(pc.tcmod_u.x, pc.tcmod_v.x)), length(vec2(pc.tcmod_u.y, pc.tcmod_v.y)));   // Isolate the UV scale so we can scale the pivot position accordingly
    v_tex_uv = vec2(dot(uv_dir, pc.tcmod_u), dot(uv_dir, pc.tcmod_v)) + tc_scale * uv_pivot;

    v_lightmap_uv = lightmap_coord;
    v_lightgrid_uv = (uniforms.lightgrid * worldpos).xyz;

    // Transform the vertex position into view space, subtract the camera position from it (which is 0 in view space),
    // giving the view ray in view space, then rotate the ray back to world space.
    v_view_ray = transpose(mat3(uniforms.view)) * viewpos.xyz;
}
