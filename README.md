## Introduction
Welcome to RustBSP! This is my first serious project developing anything with the Rust programming language. Its intention was to implement something I'm familiar with (loading and rendering Quake 3 maps) while learning something I am unfamiliar with (Rust, and also a bit of Vulkan). This allowed me to really get invested in Rust and dig into its specific semantics and idioms, by building something significantly more complex than your typical "Hello World" program, but without having to figure out what it is that I'm building first. It gave me a clear vision and kept the learning process focused.

In case the title doesn't give it away already, RustBSP allows you to load in and render BSP map files intended for the Quake 3 engine. It's something I have made before while in college, using C/C++ and OpenGL as the programming language and rendering API of choice. Remaking this today allowed me to make use of more modern rendering techiques, which was a fun exercise in itself. It also means I was able to implement more features and effects than I've ever done before.

Although the project's source is clearly the result of a learning process and some of the choices I've made along the way are questionable, I'm still quite happy with how this turned out. It was a valuable learning exercise in more ways than one, and it's fun to simply zoom around Quake's levels looking at all the fancy effects. However I don't think I'll be continuing with this project anytime soon. Right now the source code is on the cusp of becoming a mess, so it'll need a major design overhaul if I ever wanted to add any of the features I have been thinking about.

## What This Project Is
* A toy project primarily intended to learn Rust, and to a lesser extent to learn some Vulkan.
* An opportunity to experiment with and learn more about modern graphics rendering techniques.

## What This Project Is NOT
* An accurate recreation of Quake 3's graphics engine. It's a free interpretation using modern techniques on modern hardware, with some corners cut for convience.
* A high-performance game engine. The choice of Vulkano coupled with the outdated structure of Quake 3's maps rules out the potential for this project to ever become a proper game, but that was also never the point.

## Compiling
In the basis, compiling this project is just a matter of running:
```
cargo build
```
However as this project makes use of Vulkano which in turn makes use of `shaderc-rs`, some third-party tools need to be installed first as also explained by [Vulkano's README](https://github.com/vulkano-rs/vulkano).

Download and install the following tools:
* [CMake](https://cmake.org/download/)
* [Ninja](https://github.com/ninja-build/ninja/releases)
* [Python](https://www.python.org/downloads/) (or install it through the Windows Store)

As Ninja is a zipped executable download and it needs to be available on `PATH`, what I simply did was unzip the executable into CMake's `bin` directory.
Vulkano's README recommends installing `msys2` and installing the above tools using `pacman` but you really don't need to do any of that.

After installing the above, `cargo build` should download and compile all the dependencies and build the application as expected. During development I often ran:
```
cargo run --release maps\[somemap].bsp
```
to compile and test the application in one go. A release build will obviously take longer to compile and does not provide as much valuable debugging info, but it does run a heck of a lot faster than a debug build.

## How To Use
RustBSP requires maps and data files from Quake 3 or a compatible mod to function. For copyright reasons I cannot include any art assets with this project, but I have included a single map (q3dm1.bsp) which was released as open source and that can be loaded as-is without any textures.

If you want to see all the features that RustBSP has to offer, you will need a copy of Quake 3 and extract its data files to RustBSP's working directory. Quake 3 can be purchased for cheap on Steam or GOG. Open the game's installation directory and look for the .pk3 files inside the baseq3 directory. These are regular ZIP files that can be opened with any archiving tool such as WinZip or 7zip. From these archives, you will want to extract the following directories into RustBSP's working directory:

* env
* gfx
* maps
* models
* scripts
* textures

When this is done, simply run the RustBSP executable with the .bsp file you want to load as its single startup argument, for example:
```
rustbsp.exe maps\q3dm13.bsp
```
RustBSP should automatically find and load any of the asset files that are required to render the map in full.

If you're working from RustBSP's source directory and are compiling the project yourself, you can also instead use the `cargo run` command as mentioned in the previous section.

## Features
* Loading of the complete Quake 3 BSP file structure, plus implementation of several tree traversal and querying algorithms.
* Rendering of BSP surfaces through Vulkan, using pre-compiled GLSL vertex and fragment shader programs.
* Static lighting through both pre-baked lightmaps and lightgrid. The latter is implemented through a 3D texture sampled in the fragment shader, allowing for per-pixel accurate lighting. This improves upon Quake 3 which applied lighting only per object or per vertex.
* Rendering of Quake 3's trademark Bézier curved surfaces through the use of tessellation shaders. This as opposed to Quake 3's original CPU-based tessellation methods, making better use of modern GPU features.
* Animated sky dome using view ray casting in the fragment shader.
* Support for alpha masked and alpha blended transparent surfaces.
* Hidden-surface removal through pre-calculated Potentially Visible Sets and view-frustum culling.
* Parsing of entity data with a couple of examples on how to use the resulting entities (most notably for selecting a spawn location).
* Limited parsing and interpretation of Quake 3's "shader" scripts, which are more like what we'd call material scripts these days.
* Compositing of multiple texture layers according to instructions from the shader scripts.
* Limited support for texture coordinate modifier effects, animated textures and vertex deformation effects.

## Missing Features
* Proper support for multi-layer shader effects. Currently the multiple layers are composited together into a single texture and then combined with static lighting in the fragment shader, which produces incorrect results sometimes and prohibits certain texture animation effects. Implementing this properly would require either multiple draw calls per surface (how Quake 3 did it back in the day) which is not terribly efficient and would noticeably impact performance, or dynamically translate Q3's shader scripts into GLSL shaders and compile those at load-time. The latter would be a more modern and elegant solution, but it would also be a lot of work and I'm not quite sure how well it would fit within Vulkano's framework. Either way it's not something that's a priority; the current solution of pre-blending texture layers and only selectively enabling animation effects works well enough for me.
* A more accurate sky implementation. I haven't completely figured out how Quake 3 implemented its sky, but it involves some funky vertex and texture coordinate manipulation to build a partial skybox on-the-fly, a technique which I don't think would translate well to GLSL shaders. Either way, I'm reasonably happy with the current skydome implementation, which looks convincing enough so long as you don't look straight up.
* Collision detection. The BSP data structure contains bounding boxes for efficient tracing of ray collisions throughout the world. I've decided to not implement this yet as I've done it before and it's mostly just a lot of boring math programming. I might implement this at some point but it's low priority.
* Volumetric fog rendering. Another key selling point from Quake 3 that I don't think would translate well to Vulkano and GLSL. I haven't really looked into this much; quite possibly I could think of a modern reinterpretation of this technique that would look accurate enough, but it's rather low priority.

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
