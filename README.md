## Introduction

## How To Use

## Compiling

## Features
* Loading of the complete Quake 3 BSP file structure, plus implementation of several tree traversal and querying algorithms.
* Rendering of BSP surfaces through Vulkan, using pre-compiled GLSL vertex and fragment shader programs.
* Static lighting through both pre-baked lightmaps and lightgrid. The latter is implemented through a 3D texture sampled in the fragment shader, allowing for per-pixel accurate lighting. This improves upon Quake 3 which applied lighting only per object or per vertex.
* Rendering of Quake 3's trademark Bézier curved surfaces through the use of tessellation shaders. This as opposed to Quake 3's original CPU-based tessellation methods, making better use of modern GPU features.
* Sky
* Transparencies
* Hidden-surface removal through pre-calculated Potentially Visible Sets and view-frustum culling.
* Entities
* Limited parsing and interpretation of Quake 3's "shader" scripts, which are more like what we'd call material scripts these days.
* Compositing of multiple texture layers according to instructions from the shader scripts.
* Limited support for texture coordinate modifier effects, animated textures and vertex deformation effects.

## Missing Features
* Proper support for multi-layer shader effects. Currently the multiple layers are composited together into a single texture and then combined with static lighting in the fragment shader, which sometimes produces incorrect results and prohibits certain texture animation effects. Implementing this properly would require either multiple draw calls per surface (that's how Quake 3 did it back in the day) which is not terribly efficient and would noticeably impact performance, or dynamically translate Q3's shader scripts into GLSL shaders and compile those at load-time. The latter would be a more modern and elegant solution, but it would also be a lot of work and I'm not quite sure how well it would fit within Vulkano's framework. Either way it's not something that's a priority; the current solution of pre-blending texture layers and only selectively enabling animation effects works well enough for me.
* A more accurate sky implementation. I haven't completely figured out how Quake 3 implemented its sky, but it involves some funky vertex and texture coordinate manipulation to build a partial skybox on-the-fly, a technique which I don't think would translate well to GLSL shaders. Either way, I'm reasonably happy with the current skydome implementation, which looks convincing enough so long as you don't look straight up.
* Collision detection. The BSP data structure contains bounding boxes for efficient tracing of ray collisions throughout the world. I've decided to not implement this yet because I've done it before and it's mostly just a lot of boring maths programming. I might implement this at some point but it's low priority.
* Texture mipmap generation. Yeah I know, this is a really stupid feature to be missing, but it's much more difficult to implement this using Vulkano than it ought to be. Fortunately Quake 3's textures are fairly low-res, so they don't tend to alias that quickly.
* Volumetric fog rendering

## What This Project Is
* A toy project primarily intended to learn Rust, and to a lesser extent to learn some Vulkan.
* An opportunity to experiment with and learn about more modern graphics rendering techniques.

## What This Project Is NOT
* An accurate recreation of Quake 3's graphics engine. It's a free interpretation using modern techniques on modern hardware, with some corners cut for convience.
* A high-performance game engine. The choice of Vulkano coupled with the outdated layout of Quake 3's maps rules out the potential for this project to ever become a proper game, but that was also never the point.

## Some Words On Vulkano
Someone looking at this project for the first time might wonder: why the choice for Vulkano as the graphics library? To be fair, I've asked myself this question several times over the course of its implementation, after running into several performance issues and functional limitations with Vulkano. But after all is said and done, I still stand by my original decision and think it was the right choice for this project. Let me explain.

First of all, the primary goal of this project is and always has been to learn Rust and to touch upon as many Rustic idioms as possible. The goal was not to implement any new groundbreaking graphics techniques, though the option was always there. Hence I was not very interested in making use of low-level graphics libraries that are just simple FFI-wrappers around a C library with loads of unsafe code. I wanted something that used idiomatic Rust with a primarily safe API. My eye first fell upon Glium, which is a higher abstraction Rustic wrapper for OpenGL. OpenGL being my first choice for graphics API as I was already familiar with that and it's conceptually similar to the previous BSP renderer implementations that I've made. An alternative Vulkan renderer might have been an option further down the line. This would allow me to focus on solving just one specific problem, namely learning Rust and all of its intricacies.

However upon diving into Glium, I discovered that it had already long been abandoned by its original author, citing the irrelevance of OpenGL and the overlap with other Vulkan-based systems. Additionally, Glium introduced enough new concepts on top of bare OpenGL that I would have to spend some time anyway learning how to use it properly. And if I'm going to spend time learning a framework, I'd rather spend it on one that isn't regarded as obsolete by its original author. So based on their advice, I had a look at Vulkano instead and pledged to spend a bit of time going through its guides and learning the basics of the Vulkan API. Vulkano provides a similar high-level abstracted view of the underlying low-level API as Glium, with plenty of fascinating idiomatic Rust constructs to study and most importantly, a primarily safe API. It would allow me to get on with the implementation of my Rust BSP Viewer without getting bogged down too much by the technical micro-details of pure Vulkan programming. It seemed to fit the bill quite well.



## References

Below is a list of all the research and reference materials used in the creation of this project. Not included: lots of Googling StackOverflow.

### Rust
* Rust by Example: https://doc.rust-lang.org/stable/rust-by-example/index.html
* The Rust Reference: https://doc.rust-lang.org/reference/introduction.html
* The Rust Standard Library: https://doc.rust-lang.org/std/

### Vulkan
* Vulkano Guide: https://vulkano.rs/guide/introduction
* Vulkano API: https://docs.rs/vulkano/0.19.0/vulkano/
* Vulkano Examples: https://github.com/vulkano-rs/vulkano-examples
* Sascha Willems' Vulkan Examples: https://github.com/SaschaWillems/Vulkan
* Vulkan Cookbook: https://github.com/PacktPublishing/Vulkan-Cookbook
* Vulkan - A Specification (Tessellation): https://www.khronos.org/registry/vulkan/specs/1.1-extensions/html/chap22.html

### Quake 3
* Unofficial Quake 3 BSP Format: http://www.hyrtwol.dk/wolfenstein/unofficial_quake3_bsp_format.md
* Unofficial Quake 3 Map Specs: http://www.mralligator.com/q3/
* Rendering Quake 3 Maps: http://graphics.cs.brown.edu/games/quake/quake3.html
* Q3A Shader Manual: http://toolz.nexuizninjaz.com/shader/
* Description of MD3 Format: https://icculus.org/~phaethon/q3a/formats/md3format.html

### Glium
* Glium post-mortem: https://users.rust-lang.org/t/glium-post-mortem/7063
