#version 450

layout(vertices = 9) out;   // Patches are defined by a 3x3 grid of control points

layout(location = 0) in vec3 v_normal[];
layout(location = 1) in vec2 v_tex_uv[];
layout(location = 2) in vec2 v_lightmap_uv[];
layout(location = 3) in vec3 v_lightgrid_uv[];

layout(location = 0) out vec3 tc_normal[];
layout(location = 1) out vec2 tc_tex_uv[];
layout(location = 2) out vec2 tc_lightmap_uv[];
layout(location = 3) out vec3 tc_lightgrid_uv[];

void main() {
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    tc_normal[gl_InvocationID] = v_normal[gl_InvocationID];
    tc_tex_uv[gl_InvocationID] = v_tex_uv[gl_InvocationID];
    tc_lightmap_uv[gl_InvocationID] = v_lightmap_uv[gl_InvocationID];
    tc_lightgrid_uv[gl_InvocationID] = v_lightgrid_uv[gl_InvocationID];

    gl_TessLevelInner[0] = 20;
    gl_TessLevelInner[1] = 20;
    gl_TessLevelOuter[0] = 20;
    gl_TessLevelOuter[1] = 20;
    gl_TessLevelOuter[2] = 20;
    gl_TessLevelOuter[3] = 20;
}
