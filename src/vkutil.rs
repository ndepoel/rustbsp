use vulkano::
{
    device::{ Device, Queue },
    image::{ ImmutableImage, ImageDimensions, Dimensions, MipmapsCount, ImageUsage, ImageLayout, traits::ImageViewAccess, sys::ImageCreationError },
    format::{ Format },
    sampler::{ Filter },
    command_buffer::{ AutoCommandBuffer, AutoCommandBufferBuilder, CommandBuffer, CommandBufferExecFuture },
    buffer::{ CpuAccessibleBuffer, BufferUsage },
    sync::{ GpuFuture, NowFuture },
};
use image::{ ImageBuffer, Rgb, Pixel, RgbaImage };
use image::imageops;

use std::sync::Arc;

pub fn load_texture_nomipmap(queue: Arc<Queue>, img: RgbaImage) -> Result<(Arc<dyn ImageViewAccess + Send + Sync>, CommandBufferExecFuture<NowFuture, AutoCommandBuffer>), ImageCreationError>
{
    let (w, h) = img.dimensions();
    let (tex, future) = ImmutableImage::from_iter(img.into_raw().iter().cloned(), Dimensions::Dim2d { width: w, height: h }, Format::R8G8B8A8Unorm, queue.clone())?;
    Ok((tex, future))
}

pub fn load_texture_mipmapped(queue: Arc<Queue>, img: RgbaImage) -> Result<(Arc<dyn ImageViewAccess + Send + Sync>, CommandBufferExecFuture<NowFuture, AutoCommandBuffer>), ImageCreationError>
{
    let (w, h) = img.dimensions();
    let source = CpuAccessibleBuffer::from_iter(
        queue.device().clone(),
        BufferUsage::transfer_source(),
        false,
        img.into_raw().iter().cloned(),
    )?;

    let usage = ImageUsage {
        transfer_destination: true,
        transfer_source: true,
        sampled: true,
        ..ImageUsage::none()
    };
    let layout = ImageLayout::ShaderReadOnlyOptimal;

    let dimensions = Dimensions::Dim2d { width: w, height: h };
    let (tex, init) = ImmutableImage::uninitialized(
        queue.device().clone(),
        dimensions,
        Format::R8G8B8A8Unorm,
        MipmapsCount::Log2,
        usage,
        layout,
        queue.device().active_queue_families(),
    )?;

    let mut builder = AutoCommandBufferBuilder::new(queue.device().clone(), queue.family())?;
    builder
        .copy_buffer_to_image_dimensions(source, init, [0, 0, 0], dimensions.width_height_depth(), 0, dimensions.array_layers_with_cube(), 0)
        .unwrap();

    let dims = ImageDimensions::Dim2d { width: w, height: h, cubemap_compatible: false, array_layers: 1 };
    for mip in 1..tex.clone().mipmap_levels()
    {
        let src_dims = dims.mipmap_dimensions(mip - 1).unwrap();
        let dst_dims = dims.mipmap_dimensions(mip).unwrap();

        // TODO: this currently breaks as there is a conflict between 'source' and 'destination' referencing the same image buffer
        builder
            .blit_image(
                tex.clone(), [0; 3], [src_dims.width() as i32, src_dims.height() as i32, 1], 0, mip - 1,    // Source (the previous mip level)
                tex.clone(), [0; 3], [dst_dims.width() as i32, dst_dims.height() as i32, 1], 0, mip,        // Destination (the current, smaller mip level)
                1, Filter::Linear)
            .unwrap();
    }

    let command_buffer = builder.build().unwrap();

    let future = match command_buffer.execute(queue) {
        Ok(f) => f,
        Err(_) => unreachable!(),
    };

    Ok((tex, future))
}
