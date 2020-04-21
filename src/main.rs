#![allow(dead_code)]

use std::fs::File;

mod bsp;
mod math;

fn main() -> std::io::Result<()>
{
    let mut file = File::open("q3dm13.bsp")?;
    
    let world = bsp::load_world(&mut file)?;

    //println!("{:?}", world.vertices);
    //println!("{:?}", world.faces);

    for tex in &world.textures
    {
        println!("Texture '{}' type {}", tex.name(), tex.texture_type);
    }

    // TODO: dump lightmaps to raw image files
    // for i in 0..world.lightmaps.len()
    // {
    //     let lm = &world.lightmaps[i];
    // }

    println!("Entities: {}", world.entities);

    Ok(())
}
