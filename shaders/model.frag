#version 450

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec2 v_tex_uv;
layout(location = 2) in vec2 v_lightmap_uv;
layout(location = 3) in vec3 v_lightgrid_uv;
layout(location = 4) in vec3 v_view_ray;

layout(location = 0) out vec4 f_color;

layout(set = 1, binding = 0) uniform sampler2D mainTex;
layout(set = 1, binding = 1) uniform sampler3D lightgridTexA;
layout(set = 1, binding = 2) uniform sampler3D lightgridTexB;

layout(constant_id = 0) const bool alpha_mask = false;
layout(constant_id = 1) const float alpha_offset = 0.0;
layout(constant_id = 2) const bool alpha_invert = false;
layout(constant_id = 3) const bool baked_lighting = true;

vec3 decode_latlng(float lat, float lng)
{
    return vec3(cos(lat) * sin(lng), sin(lat) * sin(lng), cos(lng));
}

void main() {
    vec4 texColor = texture(mainTex, v_tex_uv);
    if (alpha_mask)
    {
        float alpha = texColor.a + alpha_offset;
        if (alpha_invert) alpha = 1.0 - alpha;
        if (alpha < 0.5)
            discard;
    }

    vec4 lighting = vec4(1, 1, 1, 1);
    if (baked_lighting)
    {
        vec4 lightgridA = texture(lightgridTexA, v_lightgrid_uv);
        vec4 lightgridB = texture(lightgridTexB, v_lightgrid_uv);
        vec3 ambient = lightgridA.rgb;
        vec3 directional = lightgridB.rgb;
        vec3 light_dir = decode_latlng(lightgridA.w, lightgridB.w);
        float brightness = clamp(dot(normalize(v_normal), light_dir), 0.0, 1.0);
        lighting = vec4(ambient + brightness * directional, 1.0);
        //f_color = vec4(lighting, 1.0);    // Just the light grid factor
    }

    f_color = texColor * lighting;
}
