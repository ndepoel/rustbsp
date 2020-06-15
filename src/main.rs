#![allow(dead_code)]
#![allow(unused_imports)]

#[macro_use]
extern crate bitflags;
extern crate glob;

use std::fs::{ File, read_to_string };
use std::env;
use std::io::{ Error, ErrorKind };
use std::io::prelude::*;

use cgmath::Vector3;
use image::{ImageBuffer, Rgb};
use glob::glob;

mod bsp;
mod parser;
mod entity;
mod q3shader;
mod vkcore;
mod vkbsp;

fn main() -> Result<(), Box<dyn std::error::Error>>
{
    let args: Vec<String> = env::args().collect();
    if args.len() < 2
    {
        return Err(Box::new(Error::new(ErrorKind::InvalidInput, "Please provide a BSP file.")));
    }

    let mut file = File::open(&args[1])?;
    let world = bsp::load_world(&mut file)?;

    let entities = entity::parse_entities(&world.entities);
    println!("Parsed {} entities", entities.len());

    let shaders = read_shader_files("scripts/*.shader")?;
    println!("Parsed {} shaders", shaders.len());

    let map_name = match entities.iter().find(|ent| ent.class_name == "worldspawn").and_then(|ent| ent.properties.get("message"))
    {
        Some(msg) => msg,
        None => "Unknown",
    };
    println!("Map name: {}", map_name);

    vkcore::init(world, entities, false);

    Ok(())
}

fn read_shader_files(glob_pattern: &str) -> Result<Vec<q3shader::Shader>, Box::<dyn std::error::Error>>
{
    let mut shaders = Vec::new();
    for entry in glob(glob_pattern)?
    {
        match entry
        {
            Ok(path) => 
            {
                let text = read_to_string(path)?;
                shaders.append(&mut q3shader::parse_shaders(&text));
            },
            _ => continue,
        }
    }

    Ok(shaders)
}
