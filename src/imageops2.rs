// Additional image operations to extend the image::imageops module
use image::{ RgbaImage, Rgba };
use image::imageops;

use std::cmp::min;

pub fn add(bottom: &mut RgbaImage, top: &RgbaImage, x: u32, y: u32)
{
    let bottom_dims = bottom.dimensions();
    let top_dims = top.dimensions();

    // Crop our top image if we're going out of bounds
    let (range_width, range_height) = imageops::overlay_bounds(bottom_dims, top_dims, x, y);

    for top_y in 0..range_height {
        for top_x in 0..range_width {
            let p = top.get_pixel(top_x, top_y);
            let mut bottom_pixel = bottom.get_pixel(x + top_x, y + top_y).clone();
            bottom_pixel[0] = min(bottom_pixel[0] as u32 + p[0] as u32, 255) as u8;
            bottom_pixel[1] = min(bottom_pixel[1] as u32 + p[1] as u32, 255) as u8;
            bottom_pixel[2] = min(bottom_pixel[2] as u32 + p[2] as u32, 255) as u8;

            bottom.put_pixel(x + top_x, y + top_y, bottom_pixel);
        }
    }
}

pub fn multiply(bottom: &mut RgbaImage, top: &RgbaImage, x: u32, y: u32)
{
    let bottom_dims = bottom.dimensions();
    let top_dims = top.dimensions();

    // Crop our top image if we're going out of bounds
    let (range_width, range_height) = imageops::overlay_bounds(bottom_dims, top_dims, x, y);

    for top_y in 0..range_height {
        for top_x in 0..range_width {
            let p = top.get_pixel(top_x, top_y);
            let mut bottom_pixel = bottom.get_pixel(x + top_x, y + top_y).clone();
            bottom_pixel[0] = (bottom_pixel[0] as u32 * p[0] as u32 / 255) as u8;
            bottom_pixel[1] = (bottom_pixel[1] as u32 * p[1] as u32 / 255) as u8;
            bottom_pixel[2] = (bottom_pixel[2] as u32 * p[2] as u32 / 255) as u8;

            bottom.put_pixel(x + top_x, y + top_y, bottom_pixel);
        }
    }
}

pub fn alpha_mask(bottom: &mut RgbaImage, top: &RgbaImage, x: u32, y: u32)
{
    let bottom_dims = bottom.dimensions();
    let top_dims = top.dimensions();

    // Crop our top image if we're going out of bounds
    let (range_width, range_height) = imageops::overlay_bounds(bottom_dims, top_dims, x, y);

    for top_y in 0..range_height {
        for top_x in 0..range_width {
            let p = top.get_pixel(top_x, top_y);
            if p[3] < 128   // TODO: allow the masking func to be provided as an argument
            {
                bottom.put_pixel(x + top_x, y + top_y, p.clone());
            }
        }
    }
}
