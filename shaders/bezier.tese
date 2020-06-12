#version 450

layout(quads, equal_spacing, cw) in;    // We use quad topology because Quake 3's patches are rectangular

layout(location = 0) in vec3 tc_normal[];
layout(location = 1) in vec2 tc_tex_uv[];
layout(location = 2) in vec2 tc_lightmap_uv[];
layout(location = 3) in vec3 tc_lightgrid_uv[];
layout(location = 4) in vec3 tc_worldpos[];

layout(location = 0) out vec3 te_normal;
layout(location = 1) out vec2 te_tex_uv;
layout(location = 2) out vec2 te_lightmap_uv;
layout(location = 3) out vec3 te_lightgrid_uv;
layout(location = 4) out vec3 te_worldpos;

// Quake 3 patch surfaces are bi-quadratic Bezier surfaces.
// This tessellation shader takes 9 control values per vertex element and evaluates them.
void main() {
    gl_Position = vec4(0, 0, 0, 0);
    te_normal = vec3(0, 0, 0);
    te_tex_uv = vec2(0, 0);
    te_lightmap_uv = vec2(0, 0);
    te_lightgrid_uv = vec3(0, 0, 0);
    te_worldpos = vec3(0, 0, 0);

    vec2 tmp = 1.0 - gl_TessCoord.xy;
    vec3 bx = vec3(tmp.x * tmp.x, 2 * gl_TessCoord.x * tmp.x, gl_TessCoord.x * gl_TessCoord.x);
    vec3 by = vec3(tmp.y * tmp.y, 2 * gl_TessCoord.y * tmp.y, gl_TessCoord.y * gl_TessCoord.y);

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            float b = bx[i] * by[j];
            int n = i * 3 + j;

            gl_Position += gl_in[n].gl_Position * b;
            te_normal += tc_normal[n] * b;
            te_tex_uv += tc_tex_uv[n] * b;
            te_lightmap_uv += tc_lightmap_uv[n] * b;
            te_lightgrid_uv += tc_lightgrid_uv[n] * b;
            te_worldpos += tc_worldpos[n] * b;
        }
    }
}
