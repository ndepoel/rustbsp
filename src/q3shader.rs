use std::str::Chars;
use std::collections::HashMap;

use cgmath::{ Vector2, Vector3 };

use super::parser;

#[derive(Debug, Default)]
pub struct Shader
{
    pub name: String,
    pub textures: Vec<TextureMap>,
    pub cull: CullMode,
}

#[derive(Debug, PartialEq, Eq)]
pub enum CullMode
{
    None,
    Front,
    Back,
}

impl Default for CullMode
{
    fn default() -> Self { Self::Front }
}

#[derive(Debug, Default)]
pub struct TextureMap
{
    pub map: String,
    pub blend: BlendMode,
    pub mask: AlphaMask,
}

#[derive(Debug, PartialEq, Eq)]
pub enum BlendMode
{
    Opaque,
    Add,
    Multiply,
    AlphaBlend,
}

impl Default for BlendMode
{
    fn default() -> Self { Self::Opaque }
}

#[derive(Debug, PartialEq, Eq)]
pub enum AlphaMask
{
    None,
    Gt0,
    Lt128,
    Ge128,
}

impl Default for AlphaMask
{
    fn default() -> Self { Self::None }
}

#[derive(Debug)]
pub struct TextureCoordModifier
{
    pub scroll: Vector2<f32>,
    pub scale: Vector2<f32>,
}

impl Default for TextureCoordModifier
{
    fn default() -> Self
    {
        Self
        {
            scroll: Vector2::new(0.0, 0.0),
            scale: Vector2::new(1.0, 1.0),
        }
    }
}

fn parse_cull_mode(chars: &mut Chars<'_>) -> CullMode
{
    match parser::next_token(chars)
    {
        Some(token) if token.to_lowercase() == "front" => CullMode::Front,
        Some(token) if token.to_lowercase() == "back" => CullMode::Back,
        Some(token) if token.to_lowercase() == "none" => CullMode::None,
        Some(token) if token.to_lowercase() == "disable" => CullMode::None,
        _ => CullMode::default(),
    }
}

fn parse_blend_func(chars: &mut Chars<'_>) -> BlendMode
{
    match parser::next_token(chars)
    {
        Some(token) if token.to_lowercase() == "add" => BlendMode::Add,
        Some(token) if token.to_lowercase() == "filter" => BlendMode::Multiply,
        Some(token) if token.to_lowercase() == "blend" => BlendMode::AlphaBlend,
        Some(token) if token.to_lowercase() == "gl_one" => match parser::next_token(chars)
        {
            Some(token) if token.to_lowercase() == "gl_zero" => BlendMode::Opaque,
            Some(token) if token.to_lowercase() == "gl_one" => BlendMode::Add,
            _ => BlendMode::default(),
        },
        Some(token) if token.to_lowercase() == "gl_zero" => match parser::next_token(chars)
        {
            Some(token) if token.to_lowercase() == "gl_src_color" => BlendMode::Multiply,
            _ => BlendMode::default(),
        },
        Some(token) if token.to_lowercase() == "gl_dst_color" => match parser::next_token(chars)
        {
            Some(token) if token.to_lowercase() == "gl_zero" => BlendMode::Multiply,
            _ => BlendMode::default(),
        },
        Some(token) if token.to_lowercase() == "gl_src_alpha" => match parser::next_token(chars)
        {
            Some(token) if token.to_lowercase() == "gl_one_minus_src_alpha" => BlendMode::AlphaBlend,
            _ => BlendMode::default(),
        },
        _ => BlendMode::default(),
    }
}

fn parse_alpha_func(chars: &mut Chars<'_>) -> AlphaMask
{
    match parser::next_token(chars)
    {
        Some(token) if token.to_lowercase() == "gt0" => AlphaMask::Gt0,
        Some(token) if token.to_lowercase() == "lt128" => AlphaMask::Lt128,
        Some(token) if token.to_lowercase() == "ge128" => AlphaMask::Ge128,
        _ => AlphaMask::None,
    }
}

fn parse_texture_map(chars: &mut Chars<'_>) -> Option<TextureMap>
{
    let mut texture = TextureMap::default();

    loop
    {
        match parser::next_token(chars)
        {
            Some(token) if token == "}" => break,
            Some(key) if key.to_lowercase() == "map" || key.to_lowercase() == "clampmap" || key.to_lowercase() == "animmap" => texture.map = parser::next_token(chars).unwrap_or_default(),  // TODO: handle animated textures differently
            Some(key) if key.to_lowercase() == "blendfunc" => texture.blend = parse_blend_func(chars),
            Some(key) if key.to_lowercase() == "alphafunc" => texture.mask = parse_alpha_func(chars),
            Some(_) => continue,
            None => break,
        }
    }

    if texture.map.is_empty() { None } else { Some(texture) }
}

fn parse_shader(chars: &mut Chars<'_>) -> Option<Shader>
{
    // A shader definition starts with its name
    let name: String;
    match parser::next_token(chars)
    {
        Some(token) => name = token,
        None => return None,
    }

    // Look for the opening bracket of the definition
    loop
    {
        match parser::next_token(chars)
        {
            Some(token) if token == "{" => break,
            Some(_) => continue,
            None => return None,
        }
    }

    let mut shader = Shader::default();
    shader.name = name;

    loop
    {
        match parser::next_token(chars)
        {
            Some(token) if token == "}" => break,
            Some(token) if token == "{" => { parse_texture_map(chars).and_then(|t| Some(shader.textures.push(t))); },
            Some(token) if token.to_lowercase() == "cull" => shader.cull = parse_cull_mode(chars),
            Some(_) => continue,
            None => break,
        }
    }

    Some(shader)
}

pub fn parse_shaders(string: &str) -> HashMap<String, Shader>
{
    let mut shaders = HashMap::new();
    let mut chars = string.chars();

    loop
    {
        match parse_shader(&mut chars)
        {
            Some(shader) => { shaders.insert(shader.name.clone(), shader); },
            None => return shaders,
        }
    }
}

#[cfg(test)]
mod tests
{
    use super::*;

    const GIRDERS: &str = "textures/base_wall/girders1i_yellofin
    {
        surfaceparm	metalsteps	
            surfaceparm trans	
        surfaceparm alphashadow
        surfaceparm playerclip
           surfaceparm nonsolid
        surfaceparm nomarks	
        cull none
            nopicmip
        {
            map textures/base_wall/girders1i_yellodark_fin.tga
            blendFunc GL_ONE GL_ZERO
            alphaFunc GE128
            depthWrite
            rgbGen identity
        }
        {
            map $lightmap
            rgbGen identity
            blendFunc GL_DST_COLOR GL_ZERO
            depthFunc equal
        }
    }";

    #[test]
    fn test_alphamasked()
    {
        let mut chars = GIRDERS.chars();
        let shdr = parse_shader(&mut chars);
        assert!(shdr.is_some());
        let shader = &shdr.unwrap();

        assert_eq!(CullMode::None, shader.cull);

        assert_eq!(2, shader.textures.len());
        assert_eq!("textures/base_wall/girders1i_yellodark_fin.tga", shader.textures[0].map);
        assert_eq!(BlendMode::Opaque, shader.textures[0].blend);
        assert_eq!(AlphaMask::Ge128, shader.textures[0].mask);

        assert_eq!("$lightmap", shader.textures[1].map);
        assert_eq!(BlendMode::Multiply, shader.textures[1].blend);
        assert_eq!(AlphaMask::None, shader.textures[1].mask);
    }
}
