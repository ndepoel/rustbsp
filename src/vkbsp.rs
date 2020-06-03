extern crate image;

use vulkano::
{
    device::{ Device, Queue },
    command_buffer::{ AutoCommandBufferBuilder, AutoCommandBuffer, DynamicState },
    pipeline::{ GraphicsPipeline, GraphicsPipelineAbstract },
    buffer::{ BufferUsage, ImmutableBuffer, cpu_pool::CpuBufferPool, BufferSlice, BufferAccess },
    framebuffer::{ FramebufferAbstract, Subpass, RenderPassAbstract },
    sync::GpuFuture,
    descriptor::{ DescriptorSet, PipelineLayoutAbstract },
    descriptor::descriptor_set::{ DescriptorSetsCollection, PersistentDescriptorSet },
    image::{ ImmutableImage, Dimensions, traits::ImageViewAccess },
    format::{ Format },
    sampler::{ Sampler, SamplerAddressMode, Filter, MipmapMode },
};
use image::{ ImageBuffer, Rgb, Pixel };

use std::sync::Arc;
use std::time::Instant;
use std::f32::consts;
use std::ops::Deref;

use cgmath::prelude::*;

use super::vkcore;
use super::bsp;
use super::entity;

vulkano::impl_vertex!(bsp::Vertex, position, texture_coord, lightmap_coord, normal);

mod vs {
    vulkano_shaders::shader!{
        ty: "vertex",
        src: "#version 450
            layout(location = 0) in vec3 position;
            layout(location = 1) in vec2 texture_coord;
            layout(location = 2) in vec2 lightmap_coord;
            layout(location = 3) in vec3 normal;

            layout(location = 0) out vec3 v_normal;
            layout(location = 1) out vec2 v_tex_uv;
            layout(location = 2) out vec2 v_light_uv;
            layout(location = 3) out vec3 v_lightdir;

            layout(set = 0, binding = 0) uniform Data {
                mat4 model;
                mat4 view;
                mat4 proj;
            } uniforms;

            void main() {
                mat4 modelview = uniforms.view * uniforms.model;
                gl_Position = uniforms.proj * modelview * vec4(position, 1.0);
                v_tex_uv = texture_coord;
                v_light_uv = lightmap_coord;
                v_normal = transpose(inverse(mat3(modelview))) * normal;
                v_lightdir = mat3(uniforms.view) * normalize(vec3(0.2, -0.5, -1));
            }
        "
    }
}

mod fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        src: "#version 450
            layout(location = 0) in vec3 v_normal;
            layout(location = 1) in vec2 v_tex_uv;
            layout(location = 2) in vec2 v_light_uv;
            layout(location = 3) in vec3 v_lightdir;

            layout(location = 0) out vec4 f_color;

            layout(set = 1, binding = 0) uniform sampler2D mainTex;
            layout(set = 1, binding = 1) uniform sampler2D lightTex;

            void main() {
                float diffuse = clamp(dot(normalize(v_normal), v_lightdir), 0.0, 1.0);
                vec4 texColor = texture(mainTex, v_tex_uv);
                vec4 lightColor = texture(lightTex, v_light_uv);

                //f_color = (0.3 + diffuse) * texColor;     // Main texture with some fake directional lighting
                f_color = lightColor;   // Just the lightmap
                //f_color = vec4((v_normal + vec3(1, 1, 1)) * 0.5, 1.0);    // View-space normals
            }
        "
    }
}

pub struct BspRenderer
{
    device: Arc<Device>,
    queue: Arc<Queue>,

    world: bsp::World,  // BspRenderer will take ownership of World

    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    vertex_buffer: Arc<ImmutableBuffer<[bsp::Vertex]>>,
    index_buffer: Arc<ImmutableBuffer<[u32]>>,

    vs_uniform_buffer: Arc<CpuBufferPool::<vs::ty::Data>>,

    texture: Arc<dyn ImageViewAccess + Send + Sync>,
    sampler: Arc<Sampler>,

    cam_pos: cgmath::Vector3<f32>,
    rotation_start: Instant,

    lightmaps: Vec<Arc<dyn ImageViewAccess + Send + Sync>>,
    surface_data: Vec<SurfaceData>,
}

struct SurfaceData
{
    // Not sure if we actually need to store these here. They're referenced in the descriptor set, and they should be managed centrally.
    main_texture: Arc<dyn ImageViewAccess + Send + Sync>,
    light_texture: Arc<dyn ImageViewAccess + Send + Sync>,

    // TODO: also add pipeline reference here so we can vary shaders per surface (e.g. sky shader)

    // These type designations are NOT nice, but using a BufferAccess trait here didn't cut it
    vertex_slice: Arc<BufferSlice<[bsp::Vertex], Arc<ImmutableBuffer<[bsp::Vertex]>>>>,
    index_slice: Arc<BufferSlice<[u32], Arc<ImmutableBuffer<[u32]>>>>,

    descriptor_set: Arc<dyn DescriptorSet + Sync + Send>,
}

// We actually might want to pull the renderpass and framebuffer creation into here as well, to allow more flexibility in what and how we render. That's something for later though.
pub fn init(device: Arc<Device>, queue: Arc<Queue>, render_pass: Arc<dyn RenderPassAbstract + Send + Sync>, world: bsp::World) -> BspRenderer
{
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

    let vs = vs::Shader::load(device.clone()).unwrap();
    let fs = fs::Shader::load(device.clone()).unwrap();

    let vs_uniform_buffer = CpuBufferPool::<vs::ty::Data>::new(device.clone(), BufferUsage::uniform_buffer());

    let texture =
    {
        let img = image::open("image_img.png").unwrap().into_rgba();
        let (w, h) = img.dimensions();
        let (tex, future) = ImmutableImage::from_iter(img.into_raw().iter().cloned(), Dimensions::Dim2d { width: w, height: h }, Format::R8G8B8A8Srgb, queue.clone()).unwrap();
        future.flush().unwrap();
        tex
    };

    let mut lightmaps = Vec::<Arc<dyn ImageViewAccess + Send + Sync>>::with_capacity(world.lightmaps.len());
    for lightmap in &world.lightmaps
    {
        lightmaps.push({
            let img = ImageBuffer::from_fn(bsp::LIGHTMAP_WIDTH as u32, bsp::LIGHTMAP_HEIGHT as u32, |x,y| { Rgb(lightmap.image[y as usize][x as usize]).to_rgba() });
            let (w, h) = img.dimensions();
            let (tex, future) = ImmutableImage::from_iter(img.into_raw().iter().cloned(), Dimensions::Dim2d { width: w, height: h }, Format::R8G8B8A8Unorm, queue.clone()).unwrap();
            future.flush().unwrap();    // TODO We could probably collect futures and join them all at once instead of going through this sequentially
            tex
        });
    }

    let sampler = Sampler::new(device.clone(), 
        Filter::Linear, Filter::Linear, MipmapMode::Linear, 
        SamplerAddressMode::ClampToEdge, SamplerAddressMode::ClampToEdge, SamplerAddressMode::ClampToEdge, 
        0.0, 1.0, 0.0, 0.0).unwrap();

    // A pipeline is sort of a description of a single material: it determines which shaders to use and sets up the static rendering parameters
    let pipeline = Arc::new(GraphicsPipeline::start()
        .vertex_input_single_buffer::<bsp::Vertex>()
        .vertex_shader(vs.main_entry_point(), ())
        .triangle_list()
        .viewports_dynamic_scissors_irrelevant(1)
        .fragment_shader(fs.main_entry_point(), ())
        .depth_stencil_simple_depth()
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap());

    let texture_layout = pipeline.descriptor_set_layout(1).unwrap();

    let mut surface_data = Vec::with_capacity(world.surfaces.len());
    for surface in &world.surfaces
    {
        surface_data.push({
            let start_vert = surface.first_vertex as usize;
            let end_vert = start_vert + surface.num_vertices as usize;
            let vertex_slice = Arc::new(BufferSlice::from_typed_buffer_access(vertex_buffer.clone()).slice(start_vert .. end_vert).unwrap());
    
            let start_index = surface.first_index as usize;
            let end_index = start_index + surface.num_indices as usize;
            let index_slice = Arc::new(BufferSlice::from_typed_buffer_access(index_buffer.clone()).slice(start_index .. end_index).unwrap());

            let descriptor_set = Arc::new(PersistentDescriptorSet::start(texture_layout.clone())
                .add_sampled_image(texture.clone(), sampler.clone()).unwrap()
                .add_sampled_image(lightmaps.get(surface.lightmap_id as usize).unwrap_or(&lightmaps[0]).clone(), sampler.clone()).unwrap()  // TODO: handle incorrect lightmap index more gracefully
                .build().unwrap()
            );

            SurfaceData
            {
                main_texture: texture.clone(),  // TODO: actually load the required texture image
                light_texture: texture.clone(),
                vertex_slice: vertex_slice.clone(),
                index_slice: index_slice.clone(),
                descriptor_set: descriptor_set.clone(),
            }
        });
    }

    let entities = entity::parse_entities(&world.entities);
    let cam_pos = match entities.iter().find(|ent| ent.class_name == "info_player_deathmatch")
    {
        Some(ent) => ent.origin + cgmath::Vector3::new(0.0, 0.0, 70.0),
        None => cgmath::Vector3::new(300.0, 40.0, 540.0)
    };

    BspRenderer
    { 
        device: device.clone(), queue: queue.clone(),
        world: world,
        pipeline: pipeline.clone(),
        vertex_buffer: vertex_buffer.clone(), index_buffer: index_buffer.clone(),
        vs_uniform_buffer: Arc::new(vs_uniform_buffer),
        texture: Arc::new(texture),
        sampler: sampler.clone(),
        cam_pos: cam_pos,
        rotation_start: Instant::now(),
        lightmaps: lightmaps,
        surface_data: surface_data,
    }
}

impl vkcore::RendererAbstract for BspRenderer
{
    // This will probably morph into a function that returns a bunch of CommandBuffers to execute eventually
    fn draw(&self, framebuffer: Arc<dyn FramebufferAbstract + Send + Sync>, dynamic_state: &mut DynamicState) -> AutoCommandBuffer
    {
        // let center = (self.world.nodes[0].mins + self.world.nodes[0].maxs) / 2;
        // let cam_pos = cgmath::Vector3::new(center.x as f32, center.y as f32, center.z as f32);

        // let leaf_index = self.world.leaf_at_position(cam_pos);
        // let leaf = &self.world.leafs[leaf_index];
        // let center = (leaf.mins + leaf.maxs) / 2;
        // let cam_pos = cgmath::Vector3::new(center.x as f32, center.y as f32, center.z as f32);

        //let cam_pos = cgmath::Vector3::new(-25.0, 300.0, 268.0);
        // let cam_pos = cgmath::Vector3::new(300.0, 40.0, 540.0);

        let leaf_index = self.world.leaf_at_position(self.cam_pos);
        let cam_leaf = &self.world.leafs[leaf_index];

        let time = self.rotation_start.elapsed().as_secs_f32();
        let angle = time * 30.0;

        // Start off with a view matrix that moves us from Vulkan's coordinate system to Quake's (+Z is up)
        let mut view = cgmath::Matrix4::from_cols(
            cgmath::Vector4::new(1.0, 0.0, 0.0, 0.0),
            cgmath::Vector4::new(0.0, 0.0, 1.0, 0.0),
            cgmath::Vector4::new(0.0, 1.0, 0.0, 0.0),
            cgmath::Vector4::new(0.0, 0.0, 0.0, 1.0));

        view = view * cgmath::Matrix4::from_angle_y(cgmath::Deg(0.0));      // Roll
        view = view * cgmath::Matrix4::from_angle_x(cgmath::Deg(180.0));    // Pitch
        view = view * cgmath::Matrix4::from_angle_z(cgmath::Deg(-angle));   // Yaw
        view = view * cgmath::Matrix4::from_translation(-self.cam_pos);

        let uniforms =
        {
            let uniform_data = vs::ty::Data
            {
                model: cgmath::Matrix4::from_scale(1.0).into(), // Just an identity matrix; the world doesn't move
                view: view.into(),
                // TODO derive aspect ratio from viewport (not doing that right now as I'm going to move viewport out of dynamic state anyway)
                proj: cgmath::perspective(cgmath::Deg(60.0), 16.0/9.0, 5.0, 10000.0).into(),
            };

            Arc::new(self.vs_uniform_buffer.next(uniform_data).unwrap())
        };

        // For the uniform vertex data we need to update the descriptor set once every frame. This can be reused for all static objects.
        let layout = self.pipeline.descriptor_set_layout(0).unwrap();
        let uniform_set = Arc::new(PersistentDescriptorSet::start(layout.clone())
            .add_buffer(uniforms.clone()).unwrap()
            .build().unwrap()
        );

        // The command buffer contains the instructions to be executed to render things specifically for this frame:
        // a single draw call contains the pipeline (i.e. material) to use, the vertex buffer (and indices) to use, and the dynamic rendering parameters to be passed to the shaders.
        let clear_values = vec!([0.1921, 0.3019, 0.4745, 1.0].into(), 1f32.into());
        let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(self.device.clone(), self.queue.family()).unwrap()
            .begin_render_pass(framebuffer.clone(), false, clear_values).unwrap();

        // Using our clever visitor pattern with closures here is trickier than it seems, because we only want to borrow data to the traversal algorithm, 
        // but the builder pattern used by Vulkano ends up moving the builder variable into the closure with no way to move it back out again. This requires a different approach.
        // let mut leaf_indices = Vec::new();
        // let draw_leaf = |index: usize, _leaf: &bsp::Leaf| leaf_indices.push(index);
        // self.world.traverse_front_to_back(cam_pos, |_index, _node| true, draw_leaf);

        // for leaf_index in leaf_indices.into_iter()
        // {
        //     let leaf = &self.world.leafs[leaf_index];
        //     builder = self.draw_leaf(leaf, builder, dynamic_state, uniforms.clone());
        // }

        let mut drawn_surfaces = Vec::new();
        drawn_surfaces.resize_with(self.world.surfaces.len(), Default::default);
        builder = self.draw_node(0, self.cam_pos, cam_leaf.cluster, &mut drawn_surfaces, builder, dynamic_state, uniform_set.clone());

        builder.end_render_pass().unwrap()
            .build().unwrap()
    }
}

impl BspRenderer
{
    fn draw_node(&self, node_index: i32, position: cgmath::Vector3<f32>, cluster: i32, drawn_surfaces: &mut Vec<bool>, builder: AutoCommandBufferBuilder, dynamic_state: &mut DynamicState, uniforms: Arc<dyn DescriptorSet + Sync + Send>) -> AutoCommandBufferBuilder
    {
        let mut builder = builder;

        if node_index < 0
        {
            let leaf_index = !node_index as usize;
            let leaf = &self.world.leafs[leaf_index];
            if self.world.cluster_visible(cluster, leaf.cluster)
            {
                builder = self.draw_leaf(leaf, drawn_surfaces, builder, dynamic_state, uniforms.clone());
            }
            return builder;
        }

        let node = &self.world.nodes[node_index as usize];

        let first: i32;
        let last: i32;
        let plane = &self.world.planes[node.plane as usize];

        if plane.point_distance(position) >= 0.0
        {
            first = node.front;
            last = node.back;
        }
        else
        {
            first = node.back;
            last = node.front;
        }

        builder = self.draw_node(first, position, cluster, drawn_surfaces, builder, dynamic_state, uniforms.clone());
        builder = self.draw_node(last, position, cluster, drawn_surfaces, builder, dynamic_state, uniforms.clone());
        builder
    }

    fn draw_leaf(&self, leaf: &bsp::Leaf, drawn_surfaces: &mut Vec<bool>, builder: AutoCommandBufferBuilder, dynamic_state: &mut DynamicState, uniforms: Arc<dyn DescriptorSet + Sync + Send>) -> AutoCommandBufferBuilder
    {
        let mut builder = builder;

        for leaf_surf_index in leaf.first_surface..(leaf.first_surface + leaf.num_surfaces)
        {
            let surface_index = self.world.leaf_surfaces[leaf_surf_index as usize] as usize;
            if drawn_surfaces[surface_index]
            {
                // Make sure we draw each surface only once
                continue;
            }

            let surface = &self.world.surfaces[surface_index];
            let texture = &self.world.textures[surface.texture_id as usize];
            if (surface.surface_type != bsp::SurfaceType::Planar && surface.surface_type != bsp::SurfaceType::Mesh) || texture.surface_flags.contains(bsp::SurfaceFlags::SKY)
            {
                // Patch surfaces will be rendered separately (using tessellation shaders) and sky surfaces require a different set of shaders
                continue;
            }

            let surface = &self.surface_data[surface_index];
            builder = self.draw_surface(surface, builder, dynamic_state, uniforms.clone());
            drawn_surfaces[surface_index] = true;
        }

        builder
    }

    fn draw_surface(&self, surface: &SurfaceData, builder: AutoCommandBufferBuilder, dynamic_state: &mut DynamicState, uniforms: Arc<dyn DescriptorSet + Sync + Send>) -> AutoCommandBufferBuilder
    {
        // TODO Building secondary command buffers per surface or leaf would probably speed this up a whole lot => tried it, but you can't pass dynamic state or per-frame uniforms to a pre-built secondary command buffer :/
        // TODO This could possibly be done more efficiently using indirect drawing instead of using buffer slices, but I'm getting stuck with Vulkano's arcane type requirements
        let sets = (uniforms.clone(), surface.descriptor_set.clone());
        builder.draw_indexed(self.pipeline.clone(), &dynamic_state, vec!(surface.vertex_slice.clone()), surface.index_slice.clone(), sets, ()).unwrap()
    }
}
