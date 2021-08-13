use vulkano::
{
    device::{ Device, Queue },
    image::{ ImmutableImage, ImageDimensions, MipmapsCount, ImageUsage, ImageLayout, ImageViewAbstract, view::ImageView, sys::ImageCreationError },
    format::{ Format },
    sampler::{ Filter },
    command_buffer::{ PrimaryAutoCommandBuffer, AutoCommandBufferBuilder, CommandBufferExecFuture },
    buffer::{ CpuAccessibleBuffer, BufferUsage },
    sync::{ GpuFuture, NowFuture },
};
use image::{ ImageBuffer, Rgb, Pixel, RgbaImage };
use image::imageops;

use std::sync::Arc;

pub fn load_texture_nomipmap(queue: Arc<Queue>, img: RgbaImage) -> Result<(Arc<dyn ImageViewAbstract + Send + Sync>, CommandBufferExecFuture<NowFuture, PrimaryAutoCommandBuffer>), ImageCreationError>
{
    let (w, h) = img.dimensions();
    let (tex, future) = ImmutableImage::from_iter(img.into_raw().iter().cloned(), ImageDimensions::Dim2d { width: w, height: h, array_layers: 1 }, MipmapsCount::One, Format::R8G8B8A8Unorm, queue.clone())?;
    Ok((ImageView::new(tex).unwrap(), future))
}

pub fn load_texture_mipmapped(queue: Arc<Queue>, img: RgbaImage) -> Result<(Arc<dyn ImageViewAbstract + Send + Sync>, CommandBufferExecFuture<NowFuture, PrimaryAutoCommandBuffer>), ImageCreationError>
{
    let (w, h) = img.dimensions();
    let (tex, future) = ImmutableImage::from_iter(img.into_raw().iter().cloned(), ImageDimensions::Dim2d { width: w, height: h, array_layers: 1 }, MipmapsCount::Log2, Format::R8G8B8A8Unorm, queue.clone())?;
    Ok((ImageView::new(tex).unwrap(), future))
}
