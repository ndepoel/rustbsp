#![allow(dead_code)]
#![allow(unused_imports)]

use std::fs::File;
use std::env;
use std::io::{Result, Error, ErrorKind};
use std::io::prelude::*;

use image::{ImageBuffer, Rgb};

mod bsp;
mod math;
mod entity;

fn main() -> Result<()>
{
    let args: Vec<String> = env::args().collect();
    if args.len() < 2
    {
        return Err(Error::new(ErrorKind::InvalidInput, "Please provide a BSP file."));
    }

    let mut file = File::open(&args[1])?;
    
    let world = bsp::load_world(&mut file)?;

    //println!("{:?}", world.vertices);
    //println!("{:?}", world.faces);

    for tex in &world.textures
    {
        println!("Texture '{}' type {}", tex.name(), tex.texture_type);
    }

    // Dump lightmaps to raw image files
    for i in 0..world.lightmaps.len()
    {
        let lm = &world.lightmaps[i];
        let img = ImageBuffer::from_fn(128, 128, |x,y| { Rgb(lm.image[x as usize][y as usize]) });
        img.save(format!("lightmap-{}.png", i)).unwrap();
    }

    println!("Entities: {}", world.entities);

    println!("World summary:\n{}", world);

    // let v = math::Vector3::default();   // Should be leaf 1882 for q3dm13.bsp
    let v = math::Vector3::new(-25.0, 300.0, 268.0);    // Should be leaf 2740 for q3dm13.bsp
    println!("Leaf at position {} = index {}", v, world.leaf_at_position(&v));

    // Traverse the BSP tree and print visited leafs using a lambda expression
    world.traverse_front_to_back(&v, |_index, _node| true, |index, _leaf| print!("{} ", index));
    println!("");

    let entities = entity::parse_entities(&world.entities);
    println!("{:?}", entities);

    Ok(())
}
