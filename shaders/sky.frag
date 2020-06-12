#version 450

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec2 v_tex_uv;
layout(location = 2) in vec2 v_lightmap_uv;
layout(location = 3) in vec3 v_lightgrid_uv;

layout(location = 0) out vec4 f_color;

layout(set = 1, binding = 0) uniform sampler2D mainTex;

layout(push_constant) uniform PushConstantData {
    vec2 unproj_scale;
    vec2 unproj_offset;
    mat4 view_transpose;
    vec2 scroll;
    vec2 scale;
} pc;

vec2 vec_to_latlng(vec3 v)
{
    v = normalize(v);
    float lat = 0.5 + atan(v.y, v.x) / (2.0 * 3.14159265);
    float lng = 0.5 + asin(v.z) / 3.14159265;
    return vec2(lat, lng);
}

void main() {
    // Convert pixel position to a view ray and transform it to world space
    vec3 ray_eye = vec3(gl_FragCoord.xy * pc.unproj_scale + pc.unproj_offset, -1);
    vec3 ray_world = mat3(pc.view_transpose) * ray_eye;

    // Convert world-space ray to spherical coordinates for a skydome effect
    vec2 uv = vec_to_latlng(ray_world);
    
    // Modify sky texture coords as specified by the textures/skies/tim_hell shader
    uv += pc.scroll;
    uv *= pc.scale;

    f_color = texture(mainTex, -uv);
}
