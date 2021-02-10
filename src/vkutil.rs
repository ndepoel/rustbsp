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
    let (tex, future) = ImmutableImage::from_iter(img.into_raw().iter().cloned(), Dimensions::Dim2d { width: w, height: h }, MipmapsCount::One, Format::R8G8B8A8Unorm, queue.clone())?;
    Ok((tex, future))
}

pub fn load_texture_mipmapped(queue: Arc<Queue>, img: RgbaImage) -> Result<(Arc<dyn ImageViewAccess + Send + Sync>, CommandBufferExecFuture<NowFuture, AutoCommandBuffer>), ImageCreationError>
{
    let (w, h) = img.dimensions();
    let (tex, future) = ImmutableImage::from_iter(img.into_raw().iter().cloned(), Dimensions::Dim2d { width: w, height: h }, MipmapsCount::Log2, Format::R8G8B8A8Unorm, queue.clone())?;
    Ok((tex, future))
}
