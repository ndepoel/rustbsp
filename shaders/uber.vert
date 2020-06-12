#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texture_coord;
layout(location = 2) in vec2 lightmap_coord;
layout(location = 3) in vec3 normal;

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec2 v_tex_uv;
layout(location = 2) out vec2 v_lightmap_uv;
layout(location = 3) out vec3 v_lightgrid_uv;

layout(set = 0, binding = 0) uniform Data {
    mat4 model;
    mat4 view;
    mat4 proj;
    mat4 lightgrid;
} uniforms;

void main() {
    mat4 modelview = uniforms.view * uniforms.model;
    gl_Position = uniforms.proj * modelview * vec4(position, 1.0);
    v_normal = transpose(inverse(mat3(uniforms.model))) * normal;   // World-space normal
    v_tex_uv = texture_coord;
    v_lightmap_uv = lightmap_coord;
    v_lightgrid_uv = (uniforms.lightgrid * uniforms.model * vec4(position, 1.0)).xyz;
}
