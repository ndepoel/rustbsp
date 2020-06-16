use std::str::Chars;
use std::collections::HashMap;
use std::path::PathBuf;
use std::cmp::max;

use cgmath::{ Vector2, Vector3 };
use image::{ ImageBuffer, Rgb, Pixel, ImageResult, DynamicImage, RgbaImage, ImageError };
use image::imageops;

use super::parser;
use super::imageops2;

#[derive(Debug, Default)]
pub struct Shader
{
    pub name: String,
    pub textures: Vec<TextureMap>,
    pub cull: CullMode,
}

impl Shader
{
    pub fn load_image(&self) -> ImageResult<RgbaImage>
    {
        let mut iter = self.textures.iter();
        let mut composite: Option<RgbaImage> = None;
        while let Some(tex) = iter.next()
        {
            // Skip environment maps and things like $lightmap
            if tex.map.starts_with("$") || tex.tc_gen != TexCoordGen::Base
            {
                continue;
            }

            let mut img = load_image_file(&tex.map)?;
            let (w, h) = img.dimensions();

            if composite.is_some()
            {
                // Make sure the two images match in size before compositing
                let (cw, ch) = composite.as_ref().unwrap().dimensions();
                if w != cw || h != ch
                {
                    let max_w = max(w, cw);
                    let max_h = max(h, ch);
                    img = imageops::resize(&img, max_w, max_h, imageops::FilterType::Triangle);
                    composite = Some(imageops::resize(composite.as_ref().unwrap(), max_w, max_h, imageops::FilterType::Triangle));
                }

                // Combine the two texture maps together. This is currently very crude, just replacing and alpha blending supported.
                match tex.blend
                {
                    BlendMode::Opaque => { imageops2::alpha_mask(composite.as_mut().unwrap(), &img, 0, 0); },
                    BlendMode::Add => { imageops2::add(composite.as_mut().unwrap(), &img, 0, 0); },
                    BlendMode::Multiply => { imageops2::multiply(composite.as_mut().unwrap(), &img, 0, 0); },
                    BlendMode::AlphaBlend => { imageops::overlay(composite.as_mut().unwrap(), &img, 0, 0); },
                    BlendMode::Ignore => { },
                };
            }
            else
            {
                composite = Some(img);
            }
        }

        match composite
        {
            Some(image) => Ok(image),
            _ => load_image_file(&self.name),
        }
    }

    pub fn is_transparent(&self) -> bool
    {
        // If the first 'proper' texture layer has blending or masking attributes, the shader is considered transparent
        let mut iter = self.textures.iter();
        while let Some(tex) = iter.next()
        {
            // Skip environment maps and things like $lightmap
            if tex.map.starts_with("$") || tex.tc_gen != TexCoordGen::Base { continue; }

            return tex.blend != BlendMode::Opaque || tex.mask != AlphaMask::None;
        }

        false
    }
}

pub fn load_image_file(tex_name: &str) -> ImageResult<RgbaImage>
{
    let extensions = vec!("", "png", "tga", "jpg"); // Check PNG first so we can easily override Quake's TGA or JPG textures with our own substitutes

    let mut file_path = PathBuf::from(tex_name);
    for ext in extensions.iter()
    {
        file_path = file_path.with_extension(ext);
        if file_path.is_file()
        {
            break;
        }
    }

    if file_path.is_file()
    {
        //println!("Loading texture from file: {}", file_path.to_string_lossy());
        Ok(image::open(file_path.to_str().unwrap())?.into_rgba())
    }
    else
    {
        println!("Could not load texture: {}", tex_name);
        Err(ImageError::from(std::io::Error::new(std::io::ErrorKind::NotFound, "Could not find texture image file")))
    }
}

pub fn save_image_file(image: &RgbaImage, name: &str, is_masked: bool, overwrite: bool)
{
    let path = PathBuf::from(format!("export/{}", name)).with_extension("png");
    if path.is_file() && !overwrite
    {
        return;
    }

    let dir = path.parent().unwrap();
    if !dir.is_dir()
    {
        std::fs::create_dir_all(dir).unwrap();
    }

    println!("Saving composited texture to path: {}", path.to_string_lossy());
    if is_masked
    {
        image.save(path).unwrap();
    }
    else
    {
        let rgb_img = image::DynamicImage::ImageRgba8(image.clone()).into_rgb();
        rgb_img.save(path).unwrap();
    }
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
    pub tc_gen: TexCoordGen,
    pub tc_mod: TexCoordModifier,
}

#[derive(Debug, PartialEq, Eq)]
pub enum BlendMode
{
    Opaque,
    Add,
    Multiply,
    AlphaBlend,
    Ignore,
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
pub struct TexCoordModifier
{
    pub scroll: Vector2<f32>,
    pub scale: Vector2<f32>,
}

impl Default for TexCoordModifier
{
    fn default() -> Self
    {
        TexCoordModifier
        {
            scroll: Vector2::new(0.0, 0.0),
            scale: Vector2::new(1.0, 1.0),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum TexCoordGen
{
    Base,
    Lightmap,
    Environment,
}

impl Default for TexCoordGen
{
    fn default() -> Self { Self::Base }
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
        Some(token) if token.to_lowercase() == "gl_add" => BlendMode::Add,
        Some(token) if token.to_lowercase() == "gl_one" => match parser::next_token(chars)
        {
            Some(token) if token.to_lowercase() == "gl_zero" => BlendMode::Opaque,
            Some(token) if token.to_lowercase() == "gl_one" => BlendMode::Add,
            _ => BlendMode::Opaque,
        },
        Some(token) if token.to_lowercase() == "gl_zero" => match parser::next_token(chars)
        {
            Some(token) if token.to_lowercase() == "gl_src_color" => BlendMode::Multiply,
            _ => BlendMode::Ignore,
        },
        Some(token) if token.to_lowercase() == "gl_dst_color" => match parser::next_token(chars)
        {
            Some(token) if token.to_lowercase() == "gl_zero" => BlendMode::Multiply,
            _ => BlendMode::Multiply,
        },
        Some(token) if token.to_lowercase() == "gl_src_alpha" => match parser::next_token(chars)
        {
            Some(token) if token.to_lowercase() == "gl_one_minus_src_alpha" => BlendMode::AlphaBlend,
            _ => BlendMode::default(),
        },
        Some(token) if token.to_lowercase() == "gl_one_minus_src_alpha" => BlendMode::AlphaBlend,
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

fn parse_tc_gen(chars: &mut Chars<'_>) -> TexCoordGen
{
    match parser::next_token(chars)
    {
        Some(token) if token.to_lowercase() == "base" => TexCoordGen::Base,
        Some(token) if token.to_lowercase() == "lightmap" => TexCoordGen::Lightmap,
        Some(token) if token.to_lowercase() == "environment" => TexCoordGen::Environment,
        _ => TexCoordGen::default(),
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
            Some(key) if key.to_lowercase() == "map" || key.to_lowercase() == "clampmap" => texture.map = parser::next_token(chars).unwrap_or_default(),
            Some(key) if key.to_lowercase() == "animmap" =>
            { 
                let _freq = parser::next_token(chars).unwrap_or_default().parse::<f32>().unwrap_or_default();
                texture.map = parser::next_token(chars).unwrap_or_default();
            },
            Some(key) if key.to_lowercase() == "blendfunc" => texture.blend = parse_blend_func(chars),
            Some(key) if key.to_lowercase() == "alphafunc" => texture.mask = parse_alpha_func(chars),
            Some(key) if key.to_lowercase() == "tcgen" => texture.tc_gen = parse_tc_gen(chars),
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

    const BLOCK: &str = "textures/gothic_floor/largerblock3b_ow
    {
    
            {
            map textures/sfx/firegorre.tga
                    tcmod scroll 0 1
                    tcMod turb 0 .25 0 1.6
                    tcmod scale 4 4
                    blendFunc GL_ONE GL_ZERO
                    rgbGen identity
        }
        {
                map textures/gothic_floor/largerblock3b_ow.tga
            blendFunc GL_SRC_ALPHA GL_ONE_MINUS_SRC_ALPHA
                rgbGen identity
        }
            {
            map $lightmap
                    blendFunc GL_DST_COLOR GL_ONE_MINUS_DST_ALPHA
            rgbGen identity
        }
    }";

    const COLUMN: &str = "//textures/gothic_trim/metalbase09_b_blocks15
    //{              
    //	{
    //		map $lightmap
    //		rgbGen identity
    //	}
    //
    //
    //       {
    //		map textures/gothic_trim/metalbase09_b_blocks15.tga
    //                blendFunc GL_dst_color GL_SRC_ALPHA
    //		alphagen lightingspecular
    //		rgbGen identity
    //	}
    //
    //}
    //
    
    textures/gothic_trim/column2c_trans
    {
        qer_editorimage textures/gothic_trim/column2c_test.tga
        surfaceparm nonsolid
        {
            map $lightmap
            rgbGen identity
        
        }
        {
            map textures/gothic_trim/column2c_test.tga  // End-of line comment
            rgbGen identity
            // Adding a random comment here
            blendFunc GL_DST_COLOR GL_ZERO
    
        
        }
    }";

    #[test]
    fn test_alphamasked()
    {
        let mut chars = GIRDERS.chars();
        let sh = parse_shader(&mut chars);
        assert!(sh.is_some());
        let shader = &sh.unwrap();

        assert_eq!(CullMode::None, shader.cull);

        assert_eq!(2, shader.textures.len());
        assert_eq!("textures/base_wall/girders1i_yellodark_fin.tga", shader.textures[0].map);
        assert_eq!(BlendMode::Opaque, shader.textures[0].blend);
        assert_eq!(AlphaMask::Ge128, shader.textures[0].mask);

        assert_eq!("$lightmap", shader.textures[1].map);
        assert_eq!(BlendMode::Multiply, shader.textures[1].blend);
        assert_eq!(AlphaMask::None, shader.textures[1].mask);
    }

    #[test]
    fn test_multilayer()
    {
        let mut chars = BLOCK.chars();
        let sh = parse_shader(&mut chars);
        assert!(sh.is_some());
        let shader = &sh.unwrap();

        assert_eq!(CullMode::Front, shader.cull);
        assert_eq!(3, shader.textures.len());

        assert_eq!("textures/sfx/firegorre.tga", shader.textures[0].map);
        assert_eq!(BlendMode::Opaque, shader.textures[0].blend);
        assert_eq!(AlphaMask::None, shader.textures[0].mask);

        assert_eq!("textures/gothic_floor/largerblock3b_ow.tga", shader.textures[1].map);
        assert_eq!(BlendMode::AlphaBlend, shader.textures[1].blend);
        assert_eq!(AlphaMask::None, shader.textures[1].mask);
    }

    #[test]
    fn test_comments()
    {
        let mut chars = COLUMN.chars();
        let sh = parse_shader(&mut chars);
        assert!(sh.is_some());
        let shader = &sh.unwrap();

        assert_eq!("textures/gothic_trim/column2c_trans", shader.name);
        assert_eq!(CullMode::Front, shader.cull);
        assert_eq!(2, shader.textures.len());

        assert_eq!("$lightmap", shader.textures[0].map);
        assert_eq!("textures/gothic_trim/column2c_test.tga", shader.textures[1].map);
        assert_eq!(BlendMode::Multiply, shader.textures[1].blend);
    }
}
