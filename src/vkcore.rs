use vulkano::{
    instance::{ Instance, PhysicalDevice },
    device::{ Device, Features },
    swapchain::{ Swapchain, SurfaceTransform, PresentMode, ColorSpace, FullscreenExclusive },
    swapchain,
    image::{ SwapchainImage, attachment::AttachmentImage },
    command_buffer::{ AutoCommandBuffer, DynamicState },
    framebuffer::{ Framebuffer, FramebufferAbstract, RenderPassAbstract },
    sync::{ GpuFuture, FlushError },
    sync,
    pipeline::{ viewport::Viewport },
    format::Format,
};
use winit::{
    event_loop::{ EventLoop, ControlFlow },
    window::{ WindowBuilder, Window, Fullscreen },
    event::{ Event, WindowEvent, VirtualKeyCode, DeviceEvent, ElementState },
};
use vulkano_win::VkSurfaceBuild;

use std::sync::Arc;
use std::time::Instant;
use std::f32::consts;

use cgmath::prelude::*;

use super::vkbsp;
use super::bsp;
use super::entity;

pub trait RendererAbstract
{
    fn draw(&self, camera: &Camera, framebuffer: Arc<dyn FramebufferAbstract + Send + Sync>, dynamic_state: &mut DynamicState) -> AutoCommandBuffer;
}

pub struct Camera
{
    pub position: cgmath::Vector3<f32>,
    pub rotation: cgmath::Vector3<f32>, // Pitch, roll, yaw

    mouse_sensitivity: f32,
    movement_speed: f32,
}

impl Camera
{
    pub fn to_view_matrix(&self) -> cgmath::Matrix4<f32>
    {
        // Start off with a view matrix that moves us from Vulkan's coordinate system to Quake's (+Z is up)
        let view = cgmath::Matrix4::from_cols(
            cgmath::Vector4::new(1.0, 0.0, 0.0, 0.0),
            cgmath::Vector4::new(0.0, 0.0, 1.0, 0.0),
            cgmath::Vector4::new(0.0, 1.0, 0.0, 0.0),
            cgmath::Vector4::new(0.0, 0.0, 0.0, 1.0));

        view *
            cgmath::Matrix4::from_angle_y(cgmath::Deg(-self.rotation.y)) *  // Roll
            cgmath::Matrix4::from_angle_x(cgmath::Deg(-self.rotation.x)) *  // Pitch
            cgmath::Matrix4::from_angle_z(cgmath::Deg(-self.rotation.z)) *  // Yaw
            cgmath::Matrix4::from_translation(-self.position)
    }

    pub fn to_quaternion(&self) -> cgmath::Quaternion<f32>
    {
        cgmath::Quaternion::from_angle_z(cgmath::Deg(self.rotation.z)) *
        cgmath::Quaternion::from_angle_x(cgmath::Deg(self.rotation.x)) *
        cgmath::Quaternion::from_angle_y(cgmath::Deg(self.rotation.y))
    }
}

pub fn init(world: bsp::World, fullscreen: bool)
{
    // Create the Vulkan instance with whatever is required to draw to a window
    let instance = 
    {   
        let extensions = vulkano_win::required_extensions();
        Instance::new(None, &extensions, None)
            .expect("failed to create instance")
    };

    // Query the available graphics devices and find a queue that supports graphics
    let physical = PhysicalDevice::enumerate(&instance).next()
        .expect("no device available");

    println!("{:?}", physical);

    for family in physical.queue_families() 
    {
        println!("Found a queue family {:?} with {:?} queue(s)", family, family.queues_count());
    }
    
    let queue_family = physical.queue_families()
        .find(|&q| q.supports_graphics())
        .expect("couldn't find a graphical queue family");

    let (device, mut queues) = 
    {
        let device_ext = vulkano::device::DeviceExtensions
        {
            khr_swapchain: true,
            .. vulkano::device::DeviceExtensions::none()
        };
        let features = Features
        {
            sampler_anisotropy: true,
            tessellation_shader: true,
            image_cube_array: true,
            fill_mode_non_solid: true,
            .. Features::none()
        };
        Device::new(physical, &features, &device_ext,
                    [(queue_family, 0.5)].iter().cloned()).expect("failed to create device")
    };
    
    let queue = queues.next().unwrap();
    println!("Device: {:?}, queue: {:?}", device, queue);   // These two are what we will be using the most: a virtual device and a command queue that allows rendering graphics

    // Create Vulkan window and a surface to render to
    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .with_inner_size(winit::dpi::PhysicalSize::new(1920, 1080))
        .with_title("Rust BSP (Vulkan)")
        .build_vk_surface(&event_loop, instance.clone()).unwrap();

    surface.window().set_cursor_grab(true).unwrap();
    surface.window().set_cursor_visible(false);
    surface.window().set_fullscreen(if fullscreen { Some(Fullscreen::Borderless(surface.window().primary_monitor())) } else { None });

    // Create a swapchain for double buffered rendering
    let (swapchain, images) = 
    {
        let caps = surface.capabilities(physical)
            .expect("failed to get surface capabilities");
    
        println!("Supported formats:");
        for fmt in caps.supported_formats.iter()
        {
            println!("{:?}", fmt);
        }

        let dimensions = caps.current_extent.unwrap_or([1920, 1080]);
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;   // NOTE this now defaults to Unorm format. Not sure if we should prefer sRGB instead.
        println!("Dimensions: {:?}, alpha: {:?}, format: {:?}", dimensions, alpha, format);

        Swapchain::new(device.clone(), surface.clone(),
            caps.min_image_count, format, dimensions, 1, caps.supported_usage_flags, &queue,
            SurfaceTransform::Identity, alpha, PresentMode::Fifo, FullscreenExclusive::Default,
            true, ColorSpace::SrgbNonLinear)
            .expect("failed to create swapchain")
    };

    // Create render pass; I'm not a fan of these magic Vulkano macros, as I have no idea what's actually happening here
    let render_pass = Arc::new(vulkano::single_pass_renderpass!(
        device.clone(),
        attachments:
        {
            color:
            {
                load: Clear,
                store: Store,
                format: swapchain.format(),
                samples: 1,
            },
            depth:
            {
                load: Clear,
                store: DontCare,
                format: Format::D16Unorm,
                samples: 1,
            }
        },
        pass:
        {
            color: [color],
            depth_stencil: {depth}
        }
    ).unwrap());

    let entities = entity::parse_entities(&world.entities);
    let (cam_pos, cam_rot) = match entities.iter().find(|ent| ent.class_name == "info_player_deathmatch")
    {
        Some(ent) => (ent.origin + cgmath::Vector3::new(0.0, 0.0, 70.0), cgmath::Vector3::new(180.0, 0.0, ent.angle - 90.0)),
        None => (cgmath::Vector3::new(300.0, 40.0, 540.0), cgmath::Vector3::new(180.0, 0.0, 0.0))
    };

    let renderer = vkbsp::init(device.clone(), queue.clone(), render_pass.clone(), world);

    // Create framebuffers from the images we created along with our swapchain
    let mut dynamic_state = DynamicState::none();
    let framebuffers = window_size_dependent_setup(device.clone(), &images, render_pass.clone(), &mut dynamic_state);
    let mut previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>);

    let mut camera = Camera
    { 
        position: cam_pos,
        rotation: cam_rot,
        mouse_sensitivity: 0.08,
        movement_speed: 250.0,
    };
    let start_time = Instant::now();
    let mut prev_time = 0.0;
    let mut movement = cgmath::Vector3::new(0.0, 0.0, 0.0);
    let mut movement_multiplier = 1.0;

    event_loop.run(move |event, _, control_flow| 
    {
        *control_flow = ControlFlow::Poll;

        match event
        {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } =>
            {
                *control_flow = ControlFlow::Exit;
            },
            Event::WindowEvent { event: WindowEvent::Resized(_), .. } => { },   // Ignoring this for now, normally you'd recreate the swapchain and framebuffers here
            Event::MainEventsCleared =>
            {
                let time = start_time.elapsed().as_secs_f32();
                let time_delta = time - prev_time;
                prev_time = time;

                previous_frame_end.as_mut().unwrap().cleanup_finished();

                let (image_num, _suboptimal, acquire_future) = swapchain::acquire_next_image(swapchain.clone(), None).unwrap();  // TODO: match result and handle errors (including out-of-date swapchain)

                if !movement.is_zero()
                {
                    camera.position += camera.to_quaternion() * movement.normalize() * time_delta * camera.movement_speed * movement_multiplier;
                }

                // Build the command buffer; apparently building the command buffer on each frame IS expected (good to know)
                // This would typically be delegated to another function where the actual setup of whatever you want to render would happen.
                let command_buffer = renderer.draw(&camera, framebuffers[image_num].clone(), &mut dynamic_state);

                // Actually execute the command buffer on a queue and present the framebuffer to the screen
                let future = previous_frame_end.take().unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer).unwrap()
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                    .then_signal_fence_and_flush();

                match future
                {
                    Ok(future) => { previous_frame_end = Some(Box::new(future) as Box<_>); },
                    // TODO also handle out-of-date errors to recreate the swapchain and framebuffers
                    Err(e) =>
                    {
                        println!("Failed to flush future: {:?}", e);
                        previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
                    }
                }
            },
            Event::WindowEvent { event: WindowEvent::KeyboardInput { input, .. }, .. } =>
            {
                match input.virtual_keycode
                {
                    Some(VirtualKeyCode::Escape) => { *control_flow = ControlFlow::Exit; }
                    _ => ()
                };
            },
            Event::DeviceEvent { event: DeviceEvent::MouseMotion { delta }, .. } =>
            {
                camera.rotation += cgmath::Vector3::new(-delta.1 as f32, 0.0, -delta.0 as f32) * camera.mouse_sensitivity;
            },
            Event::DeviceEvent { event: DeviceEvent::Key(input), .. } =>
            {
                let value = if input.state == ElementState::Pressed { 1.0 } else { 0.0 };
                match input.virtual_keycode
                {
                    Some(VirtualKeyCode::W) => movement.y = -value,
                    Some(VirtualKeyCode::S) => movement.y = value,
                    Some(VirtualKeyCode::A) => movement.x = -value,
                    Some(VirtualKeyCode::D) => movement.x = value,
                    Some(VirtualKeyCode::Q) => movement.z = value,
                    Some(VirtualKeyCode::E) => movement.z = -value,
                    Some(VirtualKeyCode::LShift) => movement_multiplier = 1.0 + value * 0.7,
                    _ => ()
                };
            }
            _ => ()
        }
    });
}

fn window_size_dependent_setup(
    device: Arc<Device>,
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();

    // This is fairly familiar; setting up a viewport based on the dimensions of the framebuffer we're rendering to
    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0 .. 1.0,
    };
    dynamic_state.viewports = Some(vec!(viewport));

    let depth_buffer = AttachmentImage::transient(device.clone(), dimensions, Format::D16Unorm).unwrap();

    // This seems to create and bind framebuffers to each of the swapchain images
    // For multi-pass rendering I guess we'd create multiple framebuffers, one for each pass, with size and format appropriate for that pass
    images.iter().map(|image| {
        Arc::new(
            Framebuffer::start(render_pass.clone())
                .add(image.clone()).unwrap()
                .add(depth_buffer.clone()).unwrap()
                .build().unwrap()
        ) as Arc<dyn FramebufferAbstract + Send + Sync>
    }).collect::<Vec<_>>()
}
