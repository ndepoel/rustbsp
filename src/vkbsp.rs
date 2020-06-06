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
use std::ops::Deref;

use cgmath::prelude::*;

use super::vkcore;
use super::bsp;

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
            }
        "
    }
}

mod tcs {
    vulkano_shaders::shader!{
        ty: "tess_ctrl",
        src: "#version 450
            layout(vertices = 9) out;   // Patches are defined by a 3x3 grid of control points

            layout(location = 0) in vec3 v_normal[];
            layout(location = 1) in vec2 v_tex_uv[];
            layout(location = 2) in vec2 v_light_uv[];

            layout(location = 0) out vec3 tc_normal[];
            layout(location = 1) out vec2 tc_tex_uv[];
            layout(location = 2) out vec2 tc_light_uv[];

            void main() {
                gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
                tc_normal[gl_InvocationID] = v_normal[gl_InvocationID];
                tc_tex_uv[gl_InvocationID] = v_tex_uv[gl_InvocationID];
                tc_light_uv[gl_InvocationID] = v_light_uv[gl_InvocationID];

                gl_TessLevelInner[0] = 20;
                gl_TessLevelInner[1] = 20;
                gl_TessLevelOuter[0] = 20;
                gl_TessLevelOuter[1] = 20;
                gl_TessLevelOuter[2] = 20;
                gl_TessLevelOuter[3] = 20;
            }
        "
    }
}

// Quake 3 patch surfaces are bi-quadratic Bezier surfaces.
// This tessellation shader takes 9 control values per vertex element and evaluates them.
mod tes {
    vulkano_shaders::shader!{
        ty: "tess_eval",
        src: "#version 450
            layout(quads, equal_spacing, cw) in;    // We use quad topology because Quake 3's patches are rectangular

            layout(location = 0) in vec3 tc_normal[];
            layout(location = 1) in vec2 tc_tex_uv[];
            layout(location = 2) in vec2 tc_light_uv[];
        
            layout(location = 0) out vec3 te_normal;
            layout(location = 1) out vec2 te_tex_uv;
            layout(location = 2) out vec2 te_light_uv;

            void main() {
                gl_Position = vec4(0, 0, 0, 0);
                te_normal = vec3(0, 0, 0);
                te_tex_uv = vec2(0, 0);
                te_light_uv = vec2(0, 0);

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
                        te_light_uv += tc_light_uv[n] * b;
                    }
                }
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

            layout(location = 0) out vec4 f_color;

            layout(set = 1, binding = 0) uniform sampler2D mainTex;
            layout(set = 1, binding = 1) uniform sampler2D lightTex;

            void main() {
                vec4 texColor = texture(mainTex, v_tex_uv);
                vec4 lightColor = texture(lightTex, v_light_uv);

                f_color = lightColor;   // Just the lightmap
                //f_color = vec4((v_normal + vec3(1, 1, 1)) * 0.5, 1.0);    // View-space normals
            }
        "
    }
}

struct BspRenderer
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

    lightmaps: Vec<Arc<dyn ImageViewAccess + Send + Sync>>,
    surface_renderers: Vec<Box<dyn SurfaceRenderer>>,
}

trait SurfaceRenderer
{
    fn draw_surface(&self, builder: AutoCommandBufferBuilder, dynamic_state: &mut DynamicState, uniforms: Arc<dyn DescriptorSet + Sync + Send>) -> AutoCommandBufferBuilder;
}

struct NoopSurfaceRenderer
{
}

struct PlanarSurfaceRenderer
{
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,

    // These type designations are NOT nice, but using a BufferAccess trait here didn't cut it
    vertex_slice: Arc<BufferSlice<[bsp::Vertex], Arc<ImmutableBuffer<[bsp::Vertex]>>>>,
    index_slice: Arc<BufferSlice<[u32], Arc<ImmutableBuffer<[u32]>>>>,

    descriptor_set: Arc<dyn DescriptorSet + Sync + Send>,
}

struct PatchSurfaceRenderer
{
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    vertex_slice: Arc<BufferSlice<[bsp::Vertex], Arc<ImmutableBuffer<[bsp::Vertex]>>>>,
    index_buffer: Arc<ImmutableBuffer<[u32]>>,  // This renderer has its own index buffer, to break apart the surface into separate patches of 9 vertices each
    descriptor_set: Arc<dyn DescriptorSet + Sync + Send>,
}

// We actually might want to pull the renderpass and framebuffer creation into here as well, to allow more flexibility in what and how we render. That's something for later though.
pub fn init(device: Arc<Device>, queue: Arc<Queue>, render_pass: Arc<dyn RenderPassAbstract + Send + Sync>, world: bsp::World) -> impl vkcore::RendererAbstract
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
    let tcs = tcs::Shader::load(device.clone()).unwrap();
    let tes = tes::Shader::load(device.clone()).unwrap();
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
            // Perform some image processing to clean up the lightmaps and make them look a bit sharper
            let img = image::imageops::resize(&img, bsp::LIGHTMAP_WIDTH as u32 * 4, bsp::LIGHTMAP_HEIGHT as u32 * 4, image::imageops::FilterType::Gaussian);
            let img = image::imageops::unsharpen(&img, 0.7, 2);
            let (w, h) = img.dimensions();
            let (tex, future) = ImmutableImage::from_iter(img.into_raw().iter().cloned(), Dimensions::Dim2d { width: w, height: h }, Format::R8G8B8A8Unorm, queue.clone()).unwrap();
            future.flush().unwrap();    // TODO We could probably collect futures and join them all at once instead of going through this sequentially
            tex
        });
    }

    let sampler = Sampler::new(device.clone(), 
        Filter::Linear, Filter::Linear, MipmapMode::Linear, 
        SamplerAddressMode::ClampToEdge, SamplerAddressMode::ClampToEdge, SamplerAddressMode::ClampToEdge, 
        0.0, 16.0, 0.0, 0.0).unwrap();

    // A pipeline is sort of a description of a single material: it determines which shaders to use and sets up the static rendering parameters
    // TODO create separate pipelines for planar surfaces, patches, meshes, sky
    let pipeline = Arc::new(GraphicsPipeline::start()
        .vertex_input_single_buffer::<bsp::Vertex>()
        .vertex_shader(vs.main_entry_point(), ())
        .triangle_list()
        //.polygon_mode_line()
        .cull_mode_front()
        .viewports_dynamic_scissors_irrelevant(1)
        .fragment_shader(fs.main_entry_point(), ())
        .depth_stencil_simple_depth()
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap());

    let patch_pipeline = Arc::new(GraphicsPipeline::start()
        .vertex_input_single_buffer::<bsp::Vertex>()
        .vertex_shader(vs.main_entry_point(), ())
        .tessellation_shaders(tcs.main_entry_point(), (), tes.main_entry_point(), ())
        .patch_list(9)
        //.polygon_mode_line()
        .cull_mode_front()
        .viewports_dynamic_scissors_irrelevant(1)
        .fragment_shader(fs.main_entry_point(), ())
        .depth_stencil_simple_depth()
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap());

    let mut surface_renderers = Vec::<Box<dyn SurfaceRenderer>>::with_capacity(world.surfaces.len());
    for surface in &world.surfaces
    {
        surface_renderers.push(
        { 
            match surface.surface_type
            {
                bsp::SurfaceType::Planar | bsp::SurfaceType::Mesh =>
                {
                    let start_vert = surface.first_vertex as usize;
                    let end_vert = start_vert + surface.num_vertices as usize;
                    let vertex_slice = Arc::new(BufferSlice::from_typed_buffer_access(vertex_buffer.clone()).slice(start_vert .. end_vert).unwrap());
            
                    let start_index = surface.first_index as usize;
                    let end_index = start_index + surface.num_indices as usize;
                    let index_slice = Arc::new(BufferSlice::from_typed_buffer_access(index_buffer.clone()).slice(start_index .. end_index).unwrap());

                    let layout = pipeline.descriptor_set_layout(1).unwrap();
                    let descriptor_set = Arc::new(PersistentDescriptorSet::start(layout.clone())
                        .add_sampled_image(texture.clone(), sampler.clone()).unwrap()   // TODO: actually load the required texture image
                        .add_sampled_image(lightmaps.get(surface.lightmap_id as usize).unwrap_or(&lightmaps[0]).clone(), sampler.clone()).unwrap()  // TODO: handle incorrect lightmap index more gracefully
                        .build().unwrap()
                    );

                    Box::new(PlanarSurfaceRenderer
                    {
                        pipeline: pipeline.clone(),
                        vertex_slice: vertex_slice.clone(),
                        index_slice: index_slice.clone(),
                        descriptor_set: descriptor_set.clone(),
                    })
                },
                bsp::SurfaceType::Patch =>
                {
                    let start_vert = surface.first_vertex as usize;
                    let end_vert = start_vert + surface.num_vertices as usize;
                    let vertex_slice = Arc::new(BufferSlice::from_typed_buffer_access(vertex_buffer.clone()).slice(start_vert .. end_vert).unwrap());

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

                    let layout = patch_pipeline.descriptor_set_layout(1).unwrap();
                    let descriptor_set = Arc::new(PersistentDescriptorSet::start(layout.clone())
                        .add_sampled_image(texture.clone(), sampler.clone()).unwrap()   // TODO: actually load the required texture image
                        .add_sampled_image(lightmaps.get(surface.lightmap_id as usize).unwrap_or(&lightmaps[0]).clone(), sampler.clone()).unwrap()  // TODO: handle incorrect lightmap index more gracefully
                        .build().unwrap()
                    );

                    Box::new(PatchSurfaceRenderer
                    {
                        pipeline: patch_pipeline.clone(),
                        vertex_slice: vertex_slice.clone(),
                        index_buffer: index_buffer.clone(),
                        descriptor_set: descriptor_set.clone(),
                    })
                },
                _ => Box::new(NoopSurfaceRenderer {})
            }
        });
    }

    BspRenderer
    { 
        device: device.clone(), queue: queue.clone(),
        world: world,
        pipeline: pipeline.clone(),
        vertex_buffer: vertex_buffer.clone(), index_buffer: index_buffer.clone(),
        vs_uniform_buffer: Arc::new(vs_uniform_buffer),
        texture: Arc::new(texture),
        sampler: sampler.clone(),
        lightmaps: lightmaps,
        surface_renderers: surface_renderers,
    }
}

impl vkcore::RendererAbstract for BspRenderer
{
    // This will probably morph into a function that returns a bunch of CommandBuffers to execute eventually
    fn draw(&self, camera: &vkcore::Camera, framebuffer: Arc<dyn FramebufferAbstract + Send + Sync>, dynamic_state: &mut DynamicState) -> AutoCommandBuffer
    {
        let leaf_index = self.world.leaf_at_position(camera.position);
        let cam_leaf = &self.world.leafs[leaf_index];

        let viewport = &dynamic_state.viewports.as_ref().unwrap()[0];
        let uniforms =
        {
            let uniform_data = vs::ty::Data
            {
                model: cgmath::Matrix4::from_scale(1.0).into(), // Just an identity matrix; the world doesn't move
                view: camera.to_view_matrix().into(),
                proj: cgmath::perspective(cgmath::Deg(60.0), viewport.dimensions[0] / viewport.dimensions[1], 5.0, 10000.0).into(),
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
        builder = self.draw_node(0, camera.position, cam_leaf.cluster, &mut drawn_surfaces, builder, dynamic_state, uniform_set.clone());

        for model in self.world.models.iter().skip(1)   // Model 0 appears to be a special model containing ALL surfaces, which we clearly do not want to render
        {
            // This is a rather crude visibility check using only the model's center point but it works well enough
            let model_leaf = self.world.leaf_at_position((model.mins + model.maxs) * 0.5);
            if self.world.cluster_visible(cam_leaf.cluster, self.world.leafs[model_leaf].cluster)
            {
                builder = self.draw_model(model, &mut drawn_surfaces, builder, dynamic_state, uniform_set.clone());
            }
        }

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

            drawn_surfaces[surface_index] = true;

            let surface = &self.world.surfaces[surface_index];
            let texture = &self.world.textures[surface.texture_id as usize];
            if texture.surface_flags.contains(bsp::SurfaceFlags::SKY)
            {
                // Patch surfaces will be rendered separately (using tessellation shaders) and sky surfaces require a different set of shaders. Meshes don't use lightmaps so would also require a different shader.
                continue;
            }

            let renderer = &self.surface_renderers[surface_index];
            builder = renderer.draw_surface(builder, dynamic_state, uniforms.clone());
        }

        builder
    }

    fn draw_model(&self, model: &bsp::Model, drawn_surfaces: &mut Vec<bool>, builder: AutoCommandBufferBuilder, dynamic_state: &mut DynamicState, uniforms: Arc<dyn DescriptorSet + Sync + Send>) -> AutoCommandBufferBuilder
    {
        let mut builder = builder;

        for model_surf_index in model.first_surface..(model.first_surface + model.num_surfaces)
        {
            let surface_index = model_surf_index as usize;
            if drawn_surfaces[surface_index]
            {
                // Make sure we draw each surface only once
                continue;
            }

            drawn_surfaces[surface_index] = true;

            let renderer = &self.surface_renderers[surface_index];
            builder = renderer.draw_surface(builder, dynamic_state, uniforms.clone());
        }

        builder
    }
}

impl SurfaceRenderer for NoopSurfaceRenderer
{
    fn draw_surface(&self, builder: AutoCommandBufferBuilder, _dynamic_state: &mut DynamicState, _uniforms: Arc<dyn DescriptorSet + Sync + Send>) -> AutoCommandBufferBuilder
    {
        builder
    }
}

impl SurfaceRenderer for PlanarSurfaceRenderer
{
    fn draw_surface(&self, builder: AutoCommandBufferBuilder, dynamic_state: &mut DynamicState, uniforms: Arc<dyn DescriptorSet + Sync + Send>) -> AutoCommandBufferBuilder
    {
        let sets = (uniforms.clone(), self.descriptor_set.clone());
        builder.draw_indexed(self.pipeline.clone(), &dynamic_state, vec!(self.vertex_slice.clone()), self.index_slice.clone(), sets, ()).unwrap()
    }
}

impl SurfaceRenderer for PatchSurfaceRenderer
{
    fn draw_surface(&self, builder: AutoCommandBufferBuilder, dynamic_state: &mut DynamicState, uniforms: Arc<dyn DescriptorSet + Sync + Send>) -> AutoCommandBufferBuilder
    {
        let sets = (uniforms.clone(), self.descriptor_set.clone());
        builder.draw_indexed(self.pipeline.clone(), &dynamic_state, vec!(self.vertex_slice.clone()), self.index_buffer.clone(), sets, ()).unwrap()
    }
}
