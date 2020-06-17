extern crate image;

use vulkano::
{
    device::{ Device, Queue },
    command_buffer::{ AutoCommandBufferBuilder, AutoCommandBuffer, DynamicState },
    pipeline::{ GraphicsPipeline, GraphicsPipelineAbstract, GraphicsPipelineCreationError },
    pipeline::blend::{ AttachmentBlend, BlendOp, BlendFactor },
    buffer::{ BufferUsage, ImmutableBuffer, cpu_pool::CpuBufferPool, BufferSlice, BufferAccess },
    framebuffer::{ FramebufferAbstract, Subpass, RenderPassAbstract },
    sync::GpuFuture,
    descriptor::{ DescriptorSet, PipelineLayoutAbstract },
    descriptor::descriptor_set::{ DescriptorSetsCollection, PersistentDescriptorSet, PersistentDescriptorSetBuildError },
    image::{ ImmutableImage, Dimensions, traits::ImageViewAccess, sys::ImageCreationError },
    format::{ Format },
    sampler::{ Sampler, SamplerAddressMode, Filter, MipmapMode },
};
use image::{ ImageBuffer, Rgb, Pixel, RgbaImage };

use std::sync::Arc;
use std::ops::Deref;
use std::path::PathBuf;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;

use cgmath::prelude::*;

use super::vkcore;
use super::bsp;
use super::q3shader;

vulkano::impl_vertex!(bsp::Vertex, position, texture_coord, lightmap_coord, normal);

// Universal vertex shader, including lightmap and lightgrid texture coordinates
mod vs {
    vulkano_shaders::shader!{
        ty: "vertex",
        path: "shaders/uber.vert",
    }
}

// Tesselation control shader for bi-quadratic Bezier curved surfaces
mod tcs {
    vulkano_shaders::shader!{
        ty: "tess_ctrl",
        path: "shaders/bezier.tesc",
    }
}

// Tesselation evaluation shader for bi-quadratic Bezier curved surfaces
mod tes {
    vulkano_shaders::shader!{
        ty: "tess_eval",
        path: "shaders/bezier.tese",
    }
}

// Fragment shader for opaque world geometry lit using a lightmap
mod world_fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        path: "shaders/world.frag",
    }
}

// Fragment shader for opaque model objects lit using the lightgrid
mod model_fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        path: "shaders/model.frag",
    }
}

// Fragment shader for a raycast animated sky surface
mod sky_fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        path: "shaders/sky.frag",
    }
}

struct BspRenderer
{
    device: Arc<Device>,
    queue: Arc<Queue>,

    world: bsp::World,  // BspRenderer will take ownership of World

    pipelines: Pipelines,
    
    vertex_buffer: Arc<ImmutableBuffer<[bsp::Vertex]>>,
    index_buffer: Arc<ImmutableBuffer<[u32]>>,

    vs_uniform_buffer: Arc<CpuBufferPool::<vs::ty::Data>>,

    lightgrid_transform: cgmath::Matrix4::<f32>,

    surface_renderers: Vec<Box<dyn SurfaceRenderer>>,
}

struct Shaders
{
    uber_vert: vs::Shader,
    bezier_tesc: tcs::Shader,
    bezier_tese: tes::Shader,
    world_frag: world_fs::Shader,
    model_frag: model_fs::Shader,
    sky_frag: sky_fs::Shader,
}

struct Pipelines
{
    device: Arc<Device>,
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>, // TODO: currently we're only using one renderpass, but my gut tells me for multi-pass rendering this should go somewhere else
    shaders: Shaders,
    pipelines: HashMap<u64, Arc<dyn GraphicsPipelineAbstract + Send + Sync>>,
}

type VertexSlice = BufferSlice<[bsp::Vertex], Arc<ImmutableBuffer<[bsp::Vertex]>>>;
type IndexSlice = BufferSlice<[u32], Arc<ImmutableBuffer<[u32]>>>;
type Texture = dyn ImageViewAccess + Send + Sync;
type TextureArray = Vec<Arc<Texture>>;

trait SurfaceRenderer
{
    fn is_transparent(&self) -> bool;
    fn draw_surface(&self, builder: &mut AutoCommandBufferBuilder, camera: &vkcore::Camera, dynamic_state: &mut DynamicState, uniforms: Arc<dyn DescriptorSet + Sync + Send>);
}

struct NoopSurfaceRenderer
{
}

struct PlanarSurfaceRenderer
{
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,

    // These type designations are NOT nice, but using a BufferAccess trait here didn't cut it
    vertex_slice: Arc<VertexSlice>,
    index_slice: Arc<IndexSlice>,

    descriptor_set: Arc<dyn DescriptorSet + Sync + Send>,
    is_transparent: bool,
}

struct PatchSurfaceRenderer
{
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    vertex_slice: Arc<VertexSlice>,
    index_buffer: Arc<ImmutableBuffer<[u32]>>,  // This renderer has its own index buffer, to break apart the surface into separate patches of 9 vertices each
    descriptor_set: Arc<dyn DescriptorSet + Sync + Send>,
}

struct SkySurfaceRenderer
{
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    vertex_slice: Arc<VertexSlice>,
    index_slice: Arc<IndexSlice>,
    descriptor_set: Arc<dyn DescriptorSet + Sync + Send>,
}

type MeshSurfaceRenderer = PlanarSurfaceRenderer;   // At the moment these two work identically, but conceptually I'd like to keep them distinct

// We actually might want to pull the renderpass and framebuffer creation into here as well, to allow more flexibility in what and how we render. That's something for later though.
pub fn init(device: Arc<Device>, queue: Arc<Queue>, render_pass: Arc<dyn RenderPassAbstract + Send + Sync>, world: bsp::World, shader_defs: HashMap<String, q3shader::Shader>) -> impl vkcore::RendererAbstract
{
    let mut pipelines = Pipelines::init(device.clone(), render_pass.clone());

    // We upload all of the BSP's vertices and indices to the GPU at once into one giant buffer. Draw calls for individual surfaces will use slices into these buffers.
    let vertex_buffer =
    { 
        let (buf, future) = ImmutableBuffer::from_iter(world.vertices.iter().cloned(), BufferUsage::vertex_buffer(), queue.clone()).unwrap();
        future.flush().unwrap();
        buf
    };

    let index_buffer =
    {
        let (buf, future) = ImmutableBuffer::from_iter(world.indices.iter().cloned(), BufferUsage::index_buffer(), queue.clone()).unwrap();
        future.flush().unwrap();
        buf
    };

    let vs_uniform_buffer = CpuBufferPool::<vs::ty::Data>::new(device.clone(), BufferUsage::uniform_buffer());

    let sampler = Sampler::new(device.clone(), 
        Filter::Linear, Filter::Linear, MipmapMode::Linear,
        SamplerAddressMode::Repeat, SamplerAddressMode::Repeat, SamplerAddressMode::Repeat,
        0.0, 16.0, 0.0, 0.0).unwrap();

    let fallback_tex = create_fallback_texture(queue.clone()).unwrap();

    let mut textures = Vec::with_capacity(world.shaders.len());
    for shader in &world.shaders
    {
        textures.push(match shader_defs.get(shader.name())
        {
            Some(shader_def) => {
                // Some textures are referenced through a shader definition
                match shader_def.load_image()
                {
                    Ok(img) => load_texture(queue.clone(), img).unwrap(),
                    _ => load_texture_file(queue.clone(), shader.name()).unwrap()
                }
            }
            _ => {
                // Some textures are referenced directly by their file name
                load_texture_file(queue.clone(), shader.name()).unwrap()
            }
        });
    }

    let mut lightmaps = Vec::with_capacity(world.lightmaps.len());
    for lightmap in &world.lightmaps
    {
        lightmaps.push(load_lightmap_texture(queue.clone(), &lightmap).unwrap());
    }

    let (dimensions, lightgrid_offset, lightgrid_scale) = world.lightgrid_dimensions();
    let lightgrid_textures = create_lightgrid_textures(queue.clone(), dimensions, &world.light_volumes).unwrap();
    let lightgrid_transform = cgmath::Matrix4::from_nonuniform_scale(lightgrid_scale.x, lightgrid_scale.y, lightgrid_scale.z) * cgmath::Matrix4::from_translation(lightgrid_offset);

    let mut surface_renderers = Vec::with_capacity(world.surfaces.len());
    for surface in &world.surfaces
    {
        let start_vert = surface.first_vertex as usize;
        let end_vert = start_vert + surface.num_vertices as usize;
        let start_index = surface.first_index as usize;
        let end_index = start_index + surface.num_indices as usize;
        let vertex_slice = Arc::new(BufferSlice::from_typed_buffer_access(vertex_buffer.clone()).slice(start_vert .. end_vert).unwrap());
        let index_slice = Arc::new(BufferSlice::from_typed_buffer_access(index_buffer.clone()).slice(start_index .. end_index).unwrap());

        surface_renderers.push(create_surface_renderer(
            &world, &surface, &shader_defs,
            queue.clone(), sampler.clone(), 
            &textures, &lightmaps, &lightgrid_textures, fallback_tex.clone(), 
            &mut pipelines, vertex_slice.clone(), index_slice.clone()
        ).unwrap());
    }

    println!("Created {} distinct pipeline configurations", pipelines.pipelines.len());

    BspRenderer
    { 
        device: device.clone(),
        queue: queue.clone(),
        world: world,
        pipelines: pipelines,
        vertex_buffer: vertex_buffer.clone(),
        index_buffer: index_buffer.clone(),
        vs_uniform_buffer: Arc::new(vs_uniform_buffer),
        lightgrid_transform: lightgrid_transform,
        surface_renderers: surface_renderers,
    }
}

fn create_fallback_texture(queue: Arc<Queue>) -> Result<Arc<Texture>, ImageCreationError>
{
    let (tex, future) = ImmutableImage::from_iter([255u8, 255u8, 255u8, 128u8].iter().cloned(), Dimensions::Dim2d { width: 1, height: 1 }, Format::R8G8B8A8Unorm, queue.clone())?;
    future.flush().unwrap();
    Ok(tex)
}

fn load_texture(queue: Arc<Queue>, img: RgbaImage) -> Result<Arc<Texture>, ImageCreationError>
{
    // let (w, h) = img.dimensions();
    // let img = image::imageops::resize(&img, w * 4, h * 4, image::imageops::FilterType::Lanczos3);
    // let img = image::imageops::unsharpen(&img, 0.7, 2);

    let (w, h) = img.dimensions();
    let (tex, future) = ImmutableImage::from_iter(img.into_raw().iter().cloned(), Dimensions::Dim2d { width: w, height: h }, Format::R8G8B8A8Unorm, queue.clone())?;
    future.flush().unwrap();
    Ok(tex)
}

fn load_texture_file(queue: Arc<Queue>, tex_name: &str) -> Result<Arc<Texture>, ImageCreationError>
{
    match q3shader::load_image_file(tex_name)
    {
        Ok(img) => load_texture(queue.clone(), img),
        Err(_) => create_fallback_texture(queue.clone()),
    }
}

fn load_lightmap_texture(queue: Arc<Queue>, lightmap: &bsp::Lightmap) -> Result<Arc<Texture>, ImageCreationError>
{
    let img = ImageBuffer::from_fn(bsp::LIGHTMAP_WIDTH as u32, bsp::LIGHTMAP_HEIGHT as u32, |x,y| { Rgb(color_shift_lighting(lightmap.image[y as usize][x as usize])).to_rgba() });
    // Perform some image processing to clean up the lightmaps and make them look a bit sharper
    // let img = image::imageops::resize(&img, bsp::LIGHTMAP_WIDTH as u32 * 4, bsp::LIGHTMAP_HEIGHT as u32 * 4, image::imageops::FilterType::Gaussian);
    // let img = image::imageops::unsharpen(&img, 0.7, 2);
    let (w, h) = img.dimensions();
    let (tex, future) = ImmutableImage::from_iter(img.into_raw().iter().cloned(), Dimensions::Dim2d { width: w, height: h }, Format::R8G8B8A8Unorm, queue.clone())?;
    future.flush().unwrap();    // TODO We could probably collect futures and join them all at once instead of going through this sequentially
    Ok(tex)
}

fn create_lightgrid_textures(queue: Arc<Queue>, dimensions: cgmath::Vector3::<usize>, light_volumes: &Vec<bsp::LightVolume>) -> Result<(Arc<Texture>, Arc<Texture>), ImageCreationError>
{
    let (w, h, d) = dimensions.into();
    let grid_size = w * h * d;
    let mut buf = Vec::new();
    buf.resize_with(grid_size * 4, Default::default);

    // The first 3D texture contains ambient light color values and the latitude part of the direction
    for i in 0..grid_size
    {
        let light_volume = &light_volumes[i];
        let ambient = color_shift_lighting(light_volume.ambient);
        buf[i * 4 + 0] = ambient[0];
        buf[i * 4 + 1] = ambient[1];
        buf[i * 4 + 2] = ambient[2];
        buf[i * 4 + 3] = light_volume.direction[1];
    }
    let (tex_a, future) = ImmutableImage::from_iter(buf.iter().cloned(), Dimensions::Dim3d { width: w as u32, height: h as u32, depth: d as u32 }, Format::R8G8B8A8Unorm, queue.clone())?;
    future.flush().unwrap();

    // The second 3D texture contains directional light color values and the longitude part of the direction
    for i in 0..grid_size
    {
        let light_volume = &light_volumes[i];
        let directional = color_shift_lighting(light_volume.directional);
        buf[i * 4 + 0] = directional[0];
        buf[i * 4 + 1] = directional[1];
        buf[i * 4 + 2] = directional[2];
        buf[i * 4 + 3] = light_volume.direction[0];
    }
    let (tex_b, future) = ImmutableImage::from_iter(buf.iter().cloned(), Dimensions::Dim3d { width: w as u32, height: h as u32, depth: d as u32 }, Format::R8G8B8A8Unorm, queue.clone())?;
    future.flush().unwrap();

    Ok((tex_a, tex_b))
}

// This code is converted from the Quake 3 source. Turns out they *do* process the lighting data before loading it into textures,
// which is why the proper Quake 3 look was so hard to replicate before.
fn color_shift_lighting(bytes: [u8; 3]) -> [u8; 3]
{
    let shift = 2;  // You can tweak this to make the lighting appear brighter or darker, but 2 seems to be the default
    let mut r = (bytes[0] as u32) << shift;
    let mut g = (bytes[1] as u32) << shift;
    let mut b = (bytes[2] as u32) << shift;

    // Normalize by color instead of saturating to white
    if (r | g | b) > 255
    {
        let max = if r > g { r } else { g };
        let max = if max > b { max } else { b };
        r = r * 255 / max;
        g = g * 255 / max;
        b = b * 255 / max;
    }

    [r as u8, g as u8, b as u8]
}

fn create_surface_renderer(
    world: &bsp::World, surface: &bsp::Surface, shader_defs: &HashMap<String, q3shader::Shader>,
    queue: Arc<Queue>, sampler: Arc<Sampler>,
    textures: &TextureArray, lightmaps: &TextureArray, lightgrid_textures: &(Arc<Texture>, Arc<Texture>), fallback_tex: Arc<Texture>,
    pipelines: &mut Pipelines, vertex_slice: Arc<VertexSlice>, index_slice: Arc<IndexSlice>) -> Result<Box<dyn SurfaceRenderer>, PersistentDescriptorSetBuildError>
{
    let flags = world.shaders.get(surface.shader_id as usize).and_then(|t| Some(t.surface_flags)).unwrap_or(bsp::SurfaceFlags::empty());
    let shader_def = world.shaders.get(surface.shader_id as usize).and_then(|s| shader_defs.get(s.name()));
    let is_transparent = shader_def.as_ref().map(|s| s.is_transparent()).unwrap_or_default();
    let pipeline = match pipelines.get(surface, flags, shader_def)
    {
        Ok(p) => p,
        Err(_) => { return Ok(Box::new(NoopSurfaceRenderer {})); },
    };

    match surface.surface_type
    {
        bsp::SurfaceType::Planar if flags.contains(bsp::SurfaceFlags::SKY) =>
        {
            let layout = pipeline.descriptor_set_layout(1).unwrap();
            let descriptor_set = Arc::new(PersistentDescriptorSet::start(layout.clone())
                .add_sampled_image(textures.get(surface.shader_id as usize).unwrap_or(&fallback_tex).clone(), sampler.clone()).unwrap()
                .build()?);

            Ok(Box::new(SkySurfaceRenderer
            {
                pipeline: pipeline.clone(),
                vertex_slice: vertex_slice.clone(),
                index_slice: index_slice.clone(),
                descriptor_set: descriptor_set.clone(),
            }))
        },
        bsp::SurfaceType::Planar =>
        {
            let layout = pipeline.descriptor_set_layout(1).unwrap();
            let descriptor_set = Arc::new(PersistentDescriptorSet::start(layout.clone())
                .add_sampled_image(textures.get(surface.shader_id as usize).unwrap_or(&fallback_tex).clone(), sampler.clone()).unwrap()
                .add_sampled_image(lightmaps.get(surface.lightmap_id as usize).unwrap_or(&fallback_tex).clone(), sampler.clone()).unwrap()
                .build()?);

            Ok(Box::new(PlanarSurfaceRenderer
            {
                pipeline: pipeline.clone(),
                vertex_slice: vertex_slice.clone(),
                index_slice: index_slice.clone(),
                descriptor_set: descriptor_set.clone(),
                is_transparent: is_transparent,
            }))
        },
        bsp::SurfaceType::Patch =>
        {
            // The vertex buffer from the BSP created above has all the vertices tightly packed with minimal duplication.
            // Vulkan's tessellation pipeline expects each patch to have a full set of 9 control points, so we have to generate an index list here to provide all of the control points in the right order.
            let index_buffer =
            {
                let mut patch_indices = Vec::new();
                let patch_count = cgmath::Vector2::new((surface.patch_size[0] - 1) / 2, (surface.patch_size[1] - 1) / 2);
                for y in 0..patch_count.y
                {
                    for x in 0..patch_count.x
                    {
                        let start = 2 * y * surface.patch_size[0] + 2 * x;
                        for i in 0..3
                        {
                            for j in 0..3
                            {
                                patch_indices.push((start + i * surface.patch_size[0] + j) as u32);
                            }
                        }
                    }
                }

                let (buf, future) = ImmutableBuffer::from_iter(patch_indices.iter().cloned(), BufferUsage::index_buffer(), queue.clone()).unwrap();
                future.flush().unwrap();
                buf
            };

            let layout = pipeline.descriptor_set_layout(1).unwrap();
            let descriptor_set = Arc::new(PersistentDescriptorSet::start(layout.clone())
                .add_sampled_image(textures.get(surface.shader_id as usize).unwrap_or(&fallback_tex).clone(), sampler.clone()).unwrap()
                .add_sampled_image(lightmaps.get(surface.lightmap_id as usize).unwrap_or(&fallback_tex).clone(), sampler.clone()).unwrap()
                .build()?);

            Ok(Box::new(PatchSurfaceRenderer
            {
                pipeline: pipeline.clone(),
                vertex_slice: vertex_slice.clone(),
                index_buffer: index_buffer.clone(),
                descriptor_set: descriptor_set.clone(),
            }))
        },
        bsp::SurfaceType::Mesh =>
        {
            let layout = pipeline.descriptor_set_layout(1).unwrap();
            let descriptor_set = Arc::new(PersistentDescriptorSet::start(layout.clone())
                .add_sampled_image(textures.get(surface.shader_id as usize).unwrap_or(&fallback_tex).clone(), sampler.clone()).unwrap()
                .add_sampled_image(lightgrid_textures.0.clone(), sampler.clone()).unwrap()
                .add_sampled_image(lightgrid_textures.1.clone(), sampler.clone()).unwrap()
                .build()?);

            Ok(Box::new(MeshSurfaceRenderer
            {
                pipeline: pipeline.clone(),
                vertex_slice: vertex_slice.clone(),
                index_slice: index_slice.clone(),
                descriptor_set: descriptor_set.clone(),
                is_transparent: is_transparent,
            }))
        },
        _ => Ok(Box::new(NoopSurfaceRenderer {}))
    }
}

impl Pipelines
{
    pub fn init(device: Arc<Device>, render_pass: Arc<dyn RenderPassAbstract + Send + Sync>) -> Self
    {
        Self
        {
            device: device.clone(),
            render_pass: render_pass.clone(),
            shaders: Shaders
            {
                uber_vert: vs::Shader::load(device.clone()).unwrap(),
                bezier_tesc: tcs::Shader::load(device.clone()).unwrap(),
                bezier_tese: tes::Shader::load(device.clone()).unwrap(),
                world_frag: world_fs::Shader::load(device.clone()).unwrap(),
                model_frag: model_fs::Shader::load(device.clone()).unwrap(),
                sky_frag: sky_fs::Shader::load(device.clone()).unwrap(),
            },
            pipelines: Default::default(),
        }
    }

    pub fn get(&mut self, surface: &bsp::Surface, surface_flags: bsp::SurfaceFlags, shader_def: Option<&q3shader::Shader>) -> Result<Arc<dyn GraphicsPipelineAbstract + Send + Sync>, GraphicsPipelineCreationError>
    {
        let cull = shader_def.map(|s| s.cull).unwrap_or_default();
        let mask = shader_def.map(|s| s.alpha_mask()).unwrap_or_default();
        let blend = shader_def.map(|s| s.blend_mode()).unwrap_or_default();

        let mut hasher = DefaultHasher::new();
        hasher.write_i32(surface.surface_type as i32);
        hasher.write_u8(surface_flags.contains(bsp::SurfaceFlags::SKY) as u8);
        hasher.write_i32(cull as i32);
        hasher.write_i32(mask as i32);
        hasher.write_i32(blend as i32);
        let hash = hasher.finish();

        match self.pipelines.get(&hash)
        {
            Some(pipeline) => Ok(pipeline.clone()),
            None =>
            {
                let pipeline = self.create(surface.surface_type, surface_flags, cull, mask, blend)?;
                self.pipelines.insert(hash, pipeline.clone());
                Ok(pipeline)
            }
        }
    }

    fn create(&mut self, surface_type: bsp::SurfaceType, surface_flags: bsp::SurfaceFlags, cull: q3shader::CullMode, mask: q3shader::AlphaMask, blend: q3shader::BlendMode) -> Result<Arc<dyn GraphicsPipelineAbstract + Send + Sync>, GraphicsPipelineCreationError>
    {
        // First, setup the basic pipeline with all the standard attributes
        let builder = GraphicsPipeline::start()
            .vertex_input_single_buffer::<bsp::Vertex>()
            .vertex_shader(self.shaders.uber_vert.main_entry_point(), ())
            //.polygon_mode_line()
            .viewports_dynamic_scissors_irrelevant(1)
            .depth_stencil_simple_depth()
            .render_pass(Subpass::from(self.render_pass.clone(), 0).unwrap());

        // Customize the pipeline based on the surface's properties
        let builder = match cull
        {
            q3shader::CullMode::None => builder.cull_mode_disabled(),
            q3shader::CullMode::Front => builder.cull_mode_front(),
            q3shader::CullMode::Back => builder.cull_mode_back(),
        };

        let builder = match blend
        {
            q3shader::BlendMode::Opaque => builder.blend_pass_through().depth_write(true),
            q3shader::BlendMode::AlphaBlend => builder.blend_alpha_blending().depth_write(false),
            q3shader::BlendMode::Add => builder.blend_collective(Self::add_blending()).depth_write(false),
            q3shader::BlendMode::Multiply => builder.blend_collective(Self::multiply_blending()).depth_write(false),
            q3shader::BlendMode::Ignore => builder.blend_collective(AttachmentBlend::ignore_source()).depth_write(false),
        };

        // Finally, attach the correct shaders to the pipeline.
        // We also build the pipeline in here because the builder struct changes type depending on what shaders you attach, so the match's result types would be incompatible otherwise.
        let pipeline = match surface_type
        {
            bsp::SurfaceType::Planar if surface_flags.contains(bsp::SurfaceFlags::SKY) =>
            {
                builder
                    .fragment_shader(self.shaders.sky_frag.main_entry_point(), ())
                    .build(self.device.clone())?
            },
            bsp::SurfaceType::Planar =>
            {
                let sc = world_fs::SpecializationConstants { alpha_mask: mask.is_masked() as u32, alpha_offset: mask.offset(), alpha_invert: mask.invert() as u32 };
                builder
                    .fragment_shader(self.shaders.world_frag.main_entry_point(), sc)
                    .build(self.device.clone())?
            },
            bsp::SurfaceType::Patch =>
            {
                let sc = world_fs::SpecializationConstants { alpha_mask: mask.is_masked() as u32, alpha_offset: mask.offset(), alpha_invert: mask.invert() as u32 };
                builder
                    .tessellation_shaders(self.shaders.bezier_tesc.main_entry_point(), (), self.shaders.bezier_tese.main_entry_point(), ())
                    .patch_list(9)
                    .fragment_shader(self.shaders.world_frag.main_entry_point(), sc)
                    .build(self.device.clone())?
            },
            bsp::SurfaceType::Mesh =>
            {
                let sc = model_fs::SpecializationConstants { alpha_mask: mask.is_masked() as u32, alpha_offset: mask.offset(), alpha_invert: mask.invert() as u32 };
                builder
                    .fragment_shader(self.shaders.model_frag.main_entry_point(), sc)
                    .build(self.device.clone())?
            },
            _ => { return Err(GraphicsPipelineCreationError::WrongShaderType); }
        };

        Ok(Arc::new(pipeline))
    }

    #[inline]
    fn add_blending() -> AttachmentBlend {
        AttachmentBlend {
            enabled: true,
            color_op: BlendOp::Add,
            color_source: BlendFactor::One,
            color_destination: BlendFactor::One,
            alpha_op: BlendOp::Add,
            alpha_source: BlendFactor::One,
            alpha_destination: BlendFactor::One,
            mask_red: true,
            mask_green: true,
            mask_blue: true,
            mask_alpha: true,
        }
    }

    #[inline]
    fn multiply_blending() -> AttachmentBlend {
        AttachmentBlend {
            enabled: true,
            color_op: BlendOp::Add,
            color_source: BlendFactor::Zero,
            color_destination: BlendFactor::SrcColor,
            alpha_op: BlendOp::Add,
            alpha_source: BlendFactor::Zero,
            alpha_destination: BlendFactor::SrcAlpha,
            mask_red: true,
            mask_green: true,
            mask_blue: true,
            mask_alpha: true,
        }
    }
}

struct RenderState
{
    drawn_surfaces: Vec<bool>,
    transparents: Vec<usize>,
}

impl RenderState
{
    fn new(num_surfaces: usize) -> Self
    {
        let mut drawn_surfaces = Vec::new();
        drawn_surfaces.resize_with(num_surfaces, Default::default);
        Self
        {
            drawn_surfaces: drawn_surfaces,
            transparents: Default::default(),
        }
    }
}

impl vkcore::RendererAbstract for BspRenderer
{
    // This will probably morph into a function that returns a bunch of CommandBuffers to execute eventually
    fn draw(&self, camera: &vkcore::Camera, framebuffer: Arc<dyn FramebufferAbstract + Send + Sync>, dynamic_state: &mut DynamicState) -> AutoCommandBuffer
    {
        let leaf_index = self.world.leaf_at_position(camera.position);
        let cam_leaf = &self.world.leafs[leaf_index];

        let uniforms =
        {
            let uniform_data = vs::ty::Data
            {
                model: cgmath::Matrix4::from_scale(1.0).into(), // Just an identity matrix; the world doesn't move
                view: camera.view_matrix().into(),
                proj: camera.projection_matrix().into(),
                lightgrid: self.lightgrid_transform.into(),
            };

            Arc::new(self.vs_uniform_buffer.next(uniform_data).unwrap())
        };

        // For the uniform vertex data we need to update the descriptor set once every frame. This can be reused for all static objects.
        let layout = self.pipelines.pipelines.iter().next().unwrap().1.descriptor_set_layout(0).unwrap(); // TODO this is REALLY unsafe :grimacing:
        let uniform_set = Arc::new(PersistentDescriptorSet::start(layout.clone())
            .add_buffer(uniforms.clone()).unwrap()
            .build().unwrap()
        );

        // The command buffer contains the instructions to be executed to render things specifically for this frame:
        // a single draw call contains the pipeline (i.e. material) to use, the vertex buffer (and indices) to use, and the dynamic rendering parameters to be passed to the shaders.
        let clear_values = vec!([0.1921, 0.3019, 0.4745, 1.0].into(), (1f32, 1u32).into());
        let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(self.device.clone(), self.queue.family()).unwrap();
        builder.begin_render_pass(framebuffer.clone(), false, clear_values).unwrap();

        // Recursively draw the BSP tree starting at node 0, while keeping track of which surfaces have already been rendered.
        let mut render_state = RenderState::new(self.world.surfaces.len());
        self.draw_node(0, camera, cam_leaf.cluster, &mut render_state, &mut builder, dynamic_state, uniform_set.clone());

        // Models are not part of the tree and need to be rendered separately.
        for model in self.world.models.iter().skip(1)   // Model 0 appears to be a special model containing ALL surfaces, which we clearly do not want to render
        {
            // This is a rather crude visibility check using only the model's center point but it works well enough
            let model_leaf = self.world.leaf_at_position((model.mins + model.maxs) * 0.5);
            if self.world.cluster_visible(cam_leaf.cluster, self.world.leafs[model_leaf].cluster)
            {
                self.draw_model(model, camera, &mut render_state, &mut builder, dynamic_state, uniform_set.clone());
            }
        }

        // Leafs are rendered front-to-back, so in order to draw transparent surfaces back-to-front we need to iterate through them in reverse.
        for surface_index in render_state.transparents.into_iter().rev()
        {
            self.draw_transparent(surface_index, camera, &mut builder, dynamic_state, uniform_set.clone());
        }

        builder.end_render_pass().unwrap();
        builder.build().unwrap()
    }
}

impl BspRenderer
{
    fn draw_node(&self, node_index: i32, camera: &vkcore::Camera, cluster: i32, render_state: &mut RenderState, builder: &mut AutoCommandBufferBuilder, dynamic_state: &mut DynamicState, uniforms: Arc<dyn DescriptorSet + Sync + Send>)
    {
        if node_index < 0
        {
            let leaf_index = !node_index as usize;
            let leaf = &self.world.leafs[leaf_index];
            if self.world.cluster_visible(cluster, leaf.cluster)
            {
                self.draw_leaf(leaf, camera, render_state, builder, dynamic_state, uniforms.clone());
            }
            return;
        }

        let node = &self.world.nodes[node_index as usize];

        let first: i32;
        let last: i32;
        let plane = &self.world.planes[node.plane as usize];

        if plane.point_distance(camera.position) >= 0.0
        {
            first = node.front;
            last = node.back;
        }
        else
        {
            first = node.back;
            last = node.front;
        }

        self.draw_node(first, camera, cluster, render_state, builder, dynamic_state, uniforms.clone());
        self.draw_node(last, camera, cluster, render_state, builder, dynamic_state, uniforms.clone());
    }

    fn draw_leaf(&self, leaf: &bsp::Leaf, camera: &vkcore::Camera, render_state: &mut RenderState, builder: &mut AutoCommandBufferBuilder, dynamic_state: &mut DynamicState, uniforms: Arc<dyn DescriptorSet + Sync + Send>)
    {
        for leaf_surf_index in leaf.first_surface..(leaf.first_surface + leaf.num_surfaces)
        {
            let surface_index = self.world.leaf_surfaces[leaf_surf_index as usize] as usize;
            if render_state.drawn_surfaces[surface_index]
            {
                // Make sure we draw each surface only once
                continue;
            }

            render_state.drawn_surfaces[surface_index] = true;

            let surface = &self.world.surfaces[surface_index];
            let texture = &self.world.shaders[surface.shader_id as usize];
            if texture.surface_flags.contains(bsp::SurfaceFlags::NODRAW) || texture.content_flags.intersects(bsp::ContentFlags::WATER | bsp::ContentFlags::FOG)
            {
                continue;
            }

            let renderer = &self.surface_renderers[surface_index];
            if renderer.is_transparent()
            {
                render_state.transparents.push(surface_index);
                continue;
            }

            renderer.draw_surface(builder, camera, dynamic_state, uniforms.clone());
        }
    }

    fn draw_model(&self, model: &bsp::Model, camera: &vkcore::Camera, render_state: &mut RenderState, builder: &mut AutoCommandBufferBuilder, dynamic_state: &mut DynamicState, uniforms: Arc<dyn DescriptorSet + Sync + Send>)
    {
        for model_surf_index in model.first_surface..(model.first_surface + model.num_surfaces)
        {
            let surface_index = model_surf_index as usize;
            if render_state.drawn_surfaces[surface_index]
            {
                // Make sure we draw each surface only once
                continue;
            }

            render_state.drawn_surfaces[surface_index] = true;

            let renderer = &self.surface_renderers[surface_index];
            if renderer.is_transparent()
            {
                // TODO: add surface to transparents list, should insert in the correct position based on camera distance
                continue;
            }

            renderer.draw_surface(builder, camera, dynamic_state, uniforms.clone());
        }
    }

    fn draw_transparent(&self, surface_index: usize, camera: &vkcore::Camera, builder: &mut AutoCommandBufferBuilder, dynamic_state: &mut DynamicState, uniforms: Arc<dyn DescriptorSet + Sync + Send>)
    {
        let renderer = &self.surface_renderers[surface_index];
        renderer.draw_surface(builder, camera, dynamic_state, uniforms.clone());
    }
}

impl SurfaceRenderer for NoopSurfaceRenderer
{
    fn is_transparent(&self) -> bool { false }

    fn draw_surface(&self, _builder: &mut AutoCommandBufferBuilder, _camera: &vkcore::Camera, _dynamic_state: &mut DynamicState, _uniforms: Arc<dyn DescriptorSet + Sync + Send>)
    {
    }
}

impl SurfaceRenderer for PlanarSurfaceRenderer
{
    fn is_transparent(&self) -> bool { self.is_transparent }

    fn draw_surface(&self, builder: &mut AutoCommandBufferBuilder, _camera: &vkcore::Camera, dynamic_state: &mut DynamicState, uniforms: Arc<dyn DescriptorSet + Sync + Send>)
    {
        // TODO Building secondary command buffers per surface or leaf would probably speed this up a whole lot => tried it, but you can't pass dynamic state or per-frame uniforms to a pre-built secondary command buffer :/
        // TODO This could possibly be done more efficiently using indirect drawing instead of using buffer slices, but I'm getting stuck with Vulkano's arcane type requirements
        // TODO Look if SyncCommandBufferBuilder can be a valid alternative (split up state binding and draw calls)
        let sets = (uniforms.clone(), self.descriptor_set.clone());
        builder.draw_indexed(self.pipeline.clone(), &dynamic_state, vec!(self.vertex_slice.clone()), self.index_slice.clone(), sets, ()).unwrap();
    }
}

impl SurfaceRenderer for PatchSurfaceRenderer
{
    fn is_transparent(&self) -> bool { false }

    fn draw_surface(&self, builder: &mut AutoCommandBufferBuilder, _camera: &vkcore::Camera, dynamic_state: &mut DynamicState, uniforms: Arc<dyn DescriptorSet + Sync + Send>)
    {
        let sets = (uniforms.clone(), self.descriptor_set.clone());
        builder.draw_indexed(self.pipeline.clone(), &dynamic_state, vec!(self.vertex_slice.clone()), self.index_buffer.clone(), sets, ()).unwrap();
    }
}

impl SurfaceRenderer for SkySurfaceRenderer
{
    fn is_transparent(&self) -> bool { false }

    fn draw_surface(&self, builder: &mut AutoCommandBufferBuilder, camera: &vkcore::Camera, dynamic_state: &mut DynamicState, uniforms: Arc<dyn DescriptorSet + Sync + Send>)
    {
        let sets = (uniforms.clone(), self.descriptor_set.clone());
        
        let pc = sky_fs::ty::PushConstantData
        {
            scroll: [0.05 * camera.time, 0.1 * camera.time],
            scale: [3.0, 2.0],
        };

        builder.draw_indexed(self.pipeline.clone(), &dynamic_state, vec!(self.vertex_slice.clone()), self.index_slice.clone(), sets, pc).unwrap();
    }
}
