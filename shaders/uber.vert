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

    // Create a transformation that contains all of the texture coordinate modifiers (TODO this can be done on the CPU for all vertices)
    float sin_rot = sin(pc.tc_rotate);
    float cos_rot = cos(pc.tc_rotate);
    vec3 tu = pc.tc_scale.x * vec3(cos_rot, -sin_rot, pc.tc_scroll.x);
    vec3 tv = pc.tc_scale.y * vec3(sin_rot, cos_rot, pc.tc_scroll.y);

    // For rotation to work correctly, we need to move the texture coordinates around a pivot in the center of the texture (i.e. 0.5, 0.5)
    vec3 uv_dir = vec3(texture_coord - 0.5, 1);
    v_tex_uv = vec2(dot(uv_dir, tu), dot(uv_dir, tv)) + 0.5;

    v_lightmap_uv = lightmap_coord;
    v_lightgrid_uv = (uniforms.lightgrid * worldpos).xyz;

    // Transform the vertex position into view space, subtract the camera position from it (which is 0 in view space),
    // giving the view ray in view space, then rotate the ray back to world space.
    v_view_ray = transpose(mat3(uniforms.view)) * viewpos.xyz;
}
