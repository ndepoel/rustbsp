use std::str::Chars;
use std::collections::HashMap;
use std::path::PathBuf;
use std::cmp::{ min, max };

use cgmath::{ Deg, Vector2, Vector3 };
use image::{ ImageBuffer, Rgba, Pixel, ImageResult, DynamicImage, RgbaImage, ImageError };
use image::imageops;

use super::parser;

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

                // Combine the two texture maps together
                tex.blend.apply(composite.as_mut().unwrap(), &img, 0, 0);
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

    pub fn blend_mode(&self) -> BlendMode
    {
        let mut iter = self.textures.iter();
        let mut first_blend = None;
        while let Some(tex) = iter.next()
        {
            if tex.blend.is_ignore() { continue; }
            if tex.blend.is_opaque() { return tex.blend; }  // If any layer is opaque, we regard the entire surface as opaque
            if !tex.map.starts_with("$") { first_blend = Some(first_blend.unwrap_or(tex.blend)); }  // Otherwise, return the blend mode of the first 'normal' texture layer
        }
        first_blend.unwrap_or_default()
    }

    pub fn alpha_mask(&self) -> AlphaMask
    {
        let mut iter = self.textures.iter();
        while let Some(tex) = iter.next()
        {
            if tex.map.starts_with("$") || tex.blend.is_ignore() { continue; }
            if tex.mask != AlphaMask::None { return tex.mask; } // If any layer is alpha-masked, we consider the entire surface alpha-masked
        }
        AlphaMask::None
    }

    pub fn uses_baked_lighting(&self) -> bool
    {
        let mut iter = self.textures.iter();
        while let Some(tex) = iter.next()
        {
            if tex.map == "$lightmap" || tex.rgb_gen == RgbGen::Entity || tex.rgb_gen == RgbGen::Vertex { return true; }
        }
        false
    }

    pub fn tex_coord_mod(&self) -> TexCoordModifier
    {
        let mut iter = self.textures.iter();
        let mut result = None;
        while let Some(tex) = iter.next()
        {
            if tex.map.starts_with("$") || tex.blend.is_ignore() { continue; }
            result = match result
            {
                Some(_) => return Default::default(),   // If we have multiple blended layers then tcMod will likely do the wrong thing, so do nothing instead
                None => Some(tex.tc_mod),
            };
        }
        result.unwrap_or_default()
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    pub animation: Option<Animation>,
    pub blend: BlendMode,
    pub mask: AlphaMask,
    pub tc_gen: TexCoordGen,
    pub tc_mod: TexCoordModifier,
    pub rgb_gen: RgbGen,
}

#[derive(Debug, Default)]
pub struct Animation
{
    pub frames: Vec<String>,
    pub frequency: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlendMode
{
    pub source: BlendFactor,
    pub destination: BlendFactor,
}

impl Default for BlendMode
{
    fn default() -> Self { Self::replace() }
}

impl BlendMode
{
    fn new(source: BlendFactor, destination: BlendFactor) -> Self
    {
        Self { source: source, destination: destination }
    }

    fn replace() -> Self { Self::new(BlendFactor::One, BlendFactor::Zero) }
    fn add() -> Self { Self::new(BlendFactor::One, BlendFactor::One) }
    fn multiply() -> Self { Self::new(BlendFactor::DstColor, BlendFactor::Zero) }
    fn blend() -> Self { Self::new(BlendFactor::SrcAlpha, BlendFactor::OneMinusSrcAlpha) }

    pub fn apply(&self, bottom: &mut RgbaImage, top: &RgbaImage, x: u32, y: u32)
    {
        let bottom_dims = bottom.dimensions();
        let top_dims = top.dimensions();

        // Crop our top image if we're going out of bounds
        let (range_width, range_height) = imageops::overlay_bounds(bottom_dims, top_dims, x, y);

        for top_y in 0..range_height {
            for top_x in 0..range_width {
                let src = top.get_pixel(top_x, top_y);
                let dst = bottom.get_pixel(x + top_x, y + top_y);

                let src_factor = self.source.apply(&src, &src, &dst);
                let dst_factor = self.destination.apply(&dst, &src, &dst);
                let result = src_factor.map2(&dst_factor, |s, d| min(s as u32 + d as u32, 255) as u8);

                bottom.put_pixel(x + top_x, y + top_y, result);
            }
        }
    }

    pub fn is_ignore(&self) -> bool { self.source == BlendFactor::Zero && self.destination == BlendFactor::One }
    pub fn is_opaque(&self) -> bool { self.source == BlendFactor::One && self.destination == BlendFactor::Zero }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendFactor
{
    One,
    Zero,
    SrcColor,
    OneMinusSrcColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstColor,
    OneMinusDstColor,
    DstAlpha,
    OneMinusDstAlpha,
}

impl Default for BlendFactor
{
    fn default() -> Self { Self::One }
}

impl BlendFactor
{
    fn apply(self, target: &Rgba<u8>, src: &Rgba<u8>, dst: &Rgba<u8>) -> Rgba<u8>
    {
        match self
        {
            Self::One => target.clone(),
            Self::Zero => target.map(|_| 0u8),
            Self::SrcColor => target.map2(src, |t, s| (t as u32 * s as u32 / 255) as u8),
            Self::OneMinusSrcColor => target.map2(src, |t, s| (t as u32 * (255 - s as u32) / 255) as u8),
            Self::SrcAlpha => target.map(|t| (t as u32 * src[3] as u32 / 255) as u8),
            Self::OneMinusSrcAlpha => target.map(|t| (t as u32 * (255 - src[3] as u32) / 255) as u8),
            Self::DstColor => target.map2(dst, |t, d| (t as u32 * d as u32 / 255) as u8),
            Self::OneMinusDstColor => target.map2(dst, |t, d| (t as u32 * (255 - d as u32) / 255) as u8),
            Self::DstAlpha => target.map(|t| (t as u32 * dst[3] as u32 / 255) as u8),
            Self::OneMinusDstAlpha => target.map(|t| (t as u32 * (255 - dst[3] as u32) / 255) as u8),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

impl AlphaMask
{
    pub fn is_masked(self) -> bool { self != Self::None }

    pub fn offset(self) -> f32 
    {
        match self
        {
            Self::Gt0 => 0.4999999,
            _ => 0.0,
        }
    }

    pub fn invert(self) -> bool { self == Self::Lt128 }
}

#[derive(Debug, Clone, Copy)]
pub struct TexCoordModifier
{
    pub rotate: Deg<f32>,
    pub scroll: Vector2<f32>,
    pub scale: Vector2<f32>,
}

impl Default for TexCoordModifier
{
    fn default() -> Self
    {
        TexCoordModifier
        {
            rotate: Deg(0.0),
            scroll: Vector2::new(0.0, 0.0),
            scale: Vector2::new(1.0, 1.0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RgbGen
{
    Identity,
    IdentityLighting,
    Wave,
    Entity,
    OneMinusEntity,
    Vertex,
    OneMinusVertex,
    LightingDiffuse,
}

impl Default for RgbGen
{
    fn default() -> Self { Self::Identity }
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

fn parse_blend_factor(string: &str) -> BlendFactor
{
    match string.to_lowercase().as_str()
    {
        "gl_one" => BlendFactor::One,
        "gl_zero" => BlendFactor::Zero,
        "gl_src_color" => BlendFactor::SrcColor,
        "gl_one_minus_src_color" => BlendFactor::OneMinusSrcColor,
        "gl_src_alpha" => BlendFactor::SrcAlpha,
        "gl_one_minus_src_alpha" => BlendFactor::OneMinusSrcAlpha,
        "gl_dst_color" => BlendFactor::DstColor,
        "gl_one_minus_dst_color" => BlendFactor::OneMinusDstColor,
        "gl_dst_alpha" => BlendFactor::DstAlpha,
        "gl_one_minus_dst_alpha" => BlendFactor::OneMinusDstAlpha,
        _ => BlendFactor::Zero,
    }
}

fn parse_blend_func(chars: &mut Chars<'_>) -> BlendMode
{
    match parser::next_token(chars)
    {
        Some(token) if token.to_lowercase() == "add" || token.to_lowercase() == "gl_add" => BlendMode { source: BlendFactor::One, destination: BlendFactor::One },
        Some(token) if token.to_lowercase() == "filter" => BlendMode { source: BlendFactor::DstColor, destination: BlendFactor::Zero },
        Some(token) if token.to_lowercase() == "blend" => BlendMode { source: BlendFactor::SrcAlpha, destination: BlendFactor::OneMinusSrcAlpha },
        Some(token) => BlendMode { source: parse_blend_factor(&token), destination: parse_blend_factor(&parser::next_token(chars).unwrap_or_default()) },
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

fn next<F>(tokens: &mut std::slice::Iter<String>, default: F) -> F
    where F: std::str::FromStr
{
    tokens.next().cloned().unwrap_or_default().parse::<F>().unwrap_or(default)
}

fn parse_tc_mod(chars: &mut Chars<'_>, tc_mod: &mut TexCoordModifier)
{
    let tokens = parser::tokenize_line(chars);
    let mut iter = tokens.iter();
    match iter.next()
    {
        Some(token) if token.to_lowercase() == "rotate" => tc_mod.rotate = Deg(next(&mut iter, 0.0)),
        Some(token) if token.to_lowercase() == "scroll" => tc_mod.scroll = Vector2::new(next(&mut iter, 0.0), next(&mut iter, 0.0)),
        Some(token) if token.to_lowercase() == "scale" => tc_mod.scale = Vector2::new(next(&mut iter, 1.0), next(&mut iter, 1.0)),
        _ => { },
    }
}

fn parse_rgb_gen(chars: &mut Chars<'_>) -> RgbGen
{
    match parser::next_token(chars)
    {
        Some(token) if token.to_lowercase() == "identity" => RgbGen::Identity,
        Some(token) if token.to_lowercase() == "identitylighting" => RgbGen::IdentityLighting,
        Some(token) if token.to_lowercase() == "wave" => { parser::tokenize_line(chars); RgbGen::Wave },
        Some(token) if token.to_lowercase() == "entity" => RgbGen::Entity,
        Some(token) if token.to_lowercase() == "oneminusentity" => RgbGen::OneMinusEntity,
        Some(token) if token.to_lowercase() == "vertex" || token.to_lowercase() == "exactvertex" => RgbGen::Vertex,
        Some(token) if token.to_lowercase() == "oneminusvertex" => RgbGen::OneMinusVertex,
        Some(token) if token.to_lowercase() == "lightingdiffuse" => RgbGen::LightingDiffuse,
        _ => RgbGen::default(),
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
                let freq = parser::next_token(chars).unwrap_or_default().parse::<f32>().unwrap_or_default();
                let anim = Animation
                {
                    frames: parser::tokenize_line(chars),
                    frequency: freq,
                };
                texture.map = anim.frames.get(0).cloned().unwrap_or_default();
                texture.animation = Some(anim);
            },
            Some(key) if key.to_lowercase() == "blendfunc" => texture.blend = parse_blend_func(chars),
            Some(key) if key.to_lowercase() == "alphafunc" => texture.mask = parse_alpha_func(chars),
            Some(key) if key.to_lowercase() == "tcgen" => texture.tc_gen = parse_tc_gen(chars),
            Some(key) if key.to_lowercase() == "tcmod" => parse_tc_mod(chars, &mut texture.tc_mod),
            Some(key) if key.to_lowercase() == "rgbgen" => texture.rgb_gen = parse_rgb_gen(chars),
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

    const SLIME: &str = "textures/liquids/slime1
	{
	//	*************************************************		
	//	* SLIME Feb 11 1999 				*
	//	* IF YOU CHANGE THIS PLEASE COMMENT THE CHANGE	*
	//	*************************************************	

		// Added to g3map_global texture on May 11, 1999
		qer_editorimage textures/liquids/slime7.tga
		q3map_lightimage textures/liquids/slime7.tga
		q3map_globaltexture
		qer_trans .5

		surfaceparm noimpact
		surfaceparm slime
		surfaceparm nolightmap
		surfaceparm trans		

		q3map_surfacelight 100
		tessSize 32
		cull disable

		deformVertexes wave 100 sin 0 1 .5 .5

		{
			map textures/liquids/slime7c.tga
			tcMod turb .3 .2 1 .05
			tcMod scroll .01 .01
		}
	
		{
			map textures/liquids/slime7.tga
			blendfunc GL_ONE GL_ONE
			tcMod turb .2 .1 1 .05
			tcMod scale .5 .5
			tcMod scroll .01 .01
		}

		{
			map textures/liquids/bubbles.tga
			blendfunc GL_ZERO GL_SRC_COLOR
			tcMod turb .2 .1 .1 .2
			tcMod scale .05 .05
			tcMod scroll .001 .001
		}		

		// 	END
    }";
    
    const GRATE: &str = "textures/base_floor/cybergrate3
    {
        cull disable
        surfaceparm alphashadow
        surfaceparm	metalsteps	
        surfaceparm nomarks
            {
                    map textures/sfx/hologirl.tga
                    blendFunc add
                    tcmod scale  1.2 .5
                    tcmod scroll 3.1 1.1
            
            }
            {
                    map textures/base_floor/cybergrate3.tga
                    alphaFunc GE128
            depthWrite
            }
            {
            map $lightmap
            rgbGen identity
            blendFunc filter
            depthFunc equal
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
        assert_eq!(BlendMode::replace(), shader.textures[0].blend);
        assert_eq!(AlphaMask::Ge128, shader.textures[0].mask);

        assert_eq!("$lightmap", shader.textures[1].map);
        assert_eq!(BlendMode::multiply(), shader.textures[1].blend);
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
        assert_eq!(BlendMode::replace(), shader.textures[0].blend);
        assert_eq!(AlphaMask::None, shader.textures[0].mask);

        assert_eq!("textures/gothic_floor/largerblock3b_ow.tga", shader.textures[1].map);
        assert_eq!(BlendMode::blend(), shader.textures[1].blend);
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
        assert_eq!(BlendMode::multiply(), shader.textures[1].blend);
    }

    #[test]
    fn test_blend()
    {
        let mut chars = SLIME.chars();
        let sh = parse_shader(&mut chars);
        assert!(sh.is_some());
        let shader = &sh.unwrap();

        assert_eq!(CullMode::None, shader.cull);
        assert_eq!(3, shader.textures.len());

        assert_eq!("textures/liquids/slime7c.tga", shader.textures[0].map);
        assert_eq!(BlendMode::replace(), shader.textures[0].blend);
        assert_eq!(AlphaMask::None, shader.textures[0].mask);

        assert_eq!("textures/liquids/slime7.tga", shader.textures[1].map);
        assert_eq!(BlendMode::add(), shader.textures[1].blend);
        assert_eq!(AlphaMask::None, shader.textures[1].mask);

        assert_eq!("textures/liquids/bubbles.tga", shader.textures[2].map);
        assert_eq!(BlendMode::new(BlendFactor::Zero, BlendFactor::SrcColor), shader.textures[2].blend);
        assert_eq!(AlphaMask::None, shader.textures[2].mask);
    }
}
