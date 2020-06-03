use std::fs::File;
use std::io::prelude::*;
use std::io::SeekFrom;
use std::io::{Error, ErrorKind};
use std::mem::{size_of, transmute};
use std::fmt;
use std::cmp;

use cgmath::{Vector2, Vector3};

// Enums in Rust are integer-counted by default like most other languages, starting at 0
enum LumpType
{
    Entities,
    Textures,
    Planes,
    Nodes,
    Leafs,
    LeafSurfaces,
    LeafBrushes,
    Models,
    Brushes,
    BrushSides,
    Vertices,
    Indices,
    Fogs,
    Surfaces,
    Lightmaps,
    LightVolumes,
    Visibility,

    MaxLumps
}

#[derive(Debug)]    // This creates a debug representation of the struct, allowing it to be easily printed as a string
#[repr(C)]  // This ensures the struct will be layed out sequentially and with padding rules from C, making them suitable for use with memory mapped files
struct Header
{
    id: [u8; 4],
    version: i32,
}

#[derive(Debug)]
#[repr(C)]
struct Lump
{
    offset: i32,
    length: i32,
}

#[derive(Debug, Default, Clone, Copy)]
#[repr(C)]
pub struct Vertex
{
    // We're using plain arrays here instead of cgmath types, because this allows us to use this data structure directly to create Vulkan vertex buffers
    pub position: [f32; 3],
    pub texture_coord: [f32; 2],
    pub lightmap_coord: [f32; 2],
    pub normal: [f32; 3],
    pub color: [u8; 4],
}

#[derive(Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum SurfaceType
{
    Bad = 0,
    Planar = 1,
    Patch = 2,
    Mesh = 3,
    Flare = 4,
}

#[derive(Debug)]
#[repr(C)]
pub struct Surface
{
    pub texture_id: i32,
    fog_id: i32,
    pub surface_type: SurfaceType,
    pub first_vertex: i32,
    pub num_vertices: i32,
    pub first_index: i32,
    pub num_indices: i32,
    pub lightmap_id: i32,
    lightmap_corner: Vector2<i32>,
    lightmap_size: Vector2<i32>,
    lightmap_pos: Vector3<f32>,
    lightmap_vecs: [Vector3<f32>; 2],  // for patches, [0] and [1] are lodbounds
    normal: Vector3<f32>,
    patch_size: Vector2<i32>,
}

bitflags!
{
    pub struct SurfaceFlags: u32
    {
        const NODAMAGE      = 0x00000001;
        const SLICK         = 0x00000002;
        const SKY           = 0x00000004;
        const LADDER        = 0x00000008;
        const NOIMPACT      = 0x00000010;
        const NOMARKS       = 0x00000020;
        const FLESH         = 0x00000040;
        const NODRAW        = 0x00000080;
        const HINT          = 0x00000100;
        const SKIP          = 0x00000200;
        const NOLIGHTMAP    = 0x00000400;
        const POINTLIGHT    = 0x00000800;
        const METALSTEPS    = 0x00001000;
        const NOSTEPS       = 0x00002000;
        const NONSOLID      = 0x00004000;
        const LIGHTFILTER   = 0x00008000;
        const ALPHASHADOW   = 0x00010000;
        const NODLIGHT      = 0x00020000;
        const DUST          = 0x00040000;
    }
}

bitflags!
{
    pub struct ContentFlags: u32
    {
        const SOLID         = 0x00000001;
        const LAVA          = 0x00000008;
        const SLIME         = 0x00000010;
        const WATER         = 0x00000020;
        const FOG           = 0x00000040;
        const NOTTEAM1      = 0x00000080;
        const NOTTEAM2      = 0x00000100;
        const NOBOTCLIP     = 0x00000200;
        const AREAPORTAL    = 0x00008000;
        const PLAYERCLIP    = 0x00010000;
        const MONSTERCLIP   = 0x00020000;
        const TELEPORTER    = 0x00040000;
        const JUMPPAD       = 0x00080000;
        const CLUSTERPORTAL = 0x00100000;
        const DONOTENTER    = 0x00200000;
        const BOTCLIP       = 0x00400000;
        const MOVER         = 0x00800000;
        const ORIGIN        = 0x01000000;
        const BODY          = 0x02000000;
        const CORPSE        = 0x04000000;
        const DETAIL        = 0x08000000;
        const STRUCTURAL    = 0x10000000;
        const TRANSLUCENT   = 0x20000000;
        const TRIGGER       = 0x40000000;
        const NODROP        = 0x80000000;
    }
}

#[repr(C)]
pub struct Texture
{
    name: [u8; 64], // Arrays are always fixed-size and have to be known at compile-time. Also #[derive(Debug)] only supports arrays up to 32 in length.
    pub surface_flags: SurfaceFlags,
    pub content_flags: ContentFlags,
}

impl Texture
{
    pub fn name(&self) -> &str
    {
        // Note: this doesn't properly handle UTF8 errors, but since BSP files only contain plain ASCII strings... ¯\_(ツ)_/¯
        std::str::from_utf8(&self.name).unwrap().trim_end_matches(char::is_control)
    }
}

pub const LIGHTMAP_WIDTH: usize = 128;
pub const LIGHTMAP_HEIGHT: usize = 128;

#[repr(C)]
pub struct Lightmap
{
    pub image: [[[u8; 3]; LIGHTMAP_HEIGHT]; LIGHTMAP_WIDTH],   // 3-dimensional array, I wonder how well this will work in practice...
}

#[derive(Debug)]
#[repr(C)]
pub struct Node
{
    pub plane: i32,
    pub front: i32,
    pub back: i32,
    pub mins: Vector3<i32>,
    pub maxs: Vector3<i32>,
}

#[derive(Debug)]
#[repr(C)]
pub struct Leaf
{
    pub cluster: i32,
    area: i32,
    pub mins: Vector3<i32>,
    pub maxs: Vector3<i32>,
    pub first_surface: i32,
    pub num_surfaces: i32,
    first_brush: i32,
    num_brushes: i32,
}

#[derive(Debug)]
#[repr(C)]
pub struct Plane
{
    normal: Vector3<f32>,
    distance: f32,
}

impl Plane
{
    pub fn point_distance(&self, point: Vector3<f32>) -> f32
    {
        cgmath::dot(self.normal, point) - self.distance
    }
}

#[derive(Debug, Default)]
#[repr(C)]
pub struct VisDataInfo
{
    num_vectors: i32,
    vector_size: i32,
}

#[derive(Debug)]
#[repr(C)]
pub struct Brush
{
    first_side: i32,
    num_sides: i32,
    texture_id: i32,
}

#[derive(Debug)]
#[repr(C)]
pub struct BrushSide
{
    plane_index: i32,
    texture_id: i32,
}

#[derive(Debug)]
#[repr(C)]
pub struct Model
{
    mins: Vector3<f32>,
    maxs: Vector3<f32>,
    first_surface: i32,
    num_surfaces: i32,
    first_brush: i32,
    num_brushes: i32,
}

#[repr(C)]
pub struct Fog
{
    shader: [u8; 64],
    brush_index: i32,
    visible_side: i32,
}

impl Fog
{
    pub fn shader_name(&self) -> &str
    {
        // Note: this doesn't properly handle UTF8 errors, but since BSP files only contain plain ASCII strings... ¯\_(ツ)_/¯
        std::str::from_utf8(&self.shader).unwrap().trim_end_matches(char::is_control)
    }
}

#[derive(Debug)]
#[repr(C)]
pub struct LightVolume
{
    ambient: [u8; 3],
    directional: [u8; 3],
    direction: [u8; 2],
}

// TODO: Bezier Patches/Faces

#[derive(Default)]  // This allows us to create an empty instance without having to initialize all the fields manually.
pub struct World
{
    pub vertices: Vec<Vertex>,
    pub surfaces: Vec<Surface>,
    pub textures: Vec<Texture>,
    pub lightmaps: Vec<Lightmap>,
    pub nodes: Vec<Node>,
    pub leafs: Vec<Leaf>,
    pub leaf_surfaces: Vec<i32>,
    pub planes: Vec<Plane>,
    pub visdata_info: VisDataInfo,
    pub visdata_vecs: Vec<u8>,
    pub entities: String,
    pub brushes: Vec<Brush>,
    pub leaf_brushes: Vec<i32>,
    pub brush_sides: Vec<BrushSide>,
    pub models: Vec<Model>,
    pub indices: Vec<u32>,
    pub fogs: Vec<Fog>,
    pub light_volumes: Vec<LightVolume>,
    // TODO: bezier faces
}

impl World
{
    pub fn tree_depth(&self) -> u32
    {
        self.tree_depth_impl(0, 0)
    }

    fn tree_depth_impl(&self, node_index: i32, depth: u32) -> u32
    {
        if node_index < 0
        {
            return depth;
        }

        let node = &self.nodes[node_index as usize];
        let front = self.tree_depth_impl(node.front, depth + 1);
        let back = self.tree_depth_impl(node.back, depth + 1);
        cmp::max(front, back)
    }

    pub fn leaf_at_position(&self, pos: Vector3<f32>) -> usize
    {
        let mut index = 0;
        while index >= 0    // Positive index means we're still looking at a tree node, not a leaf
        {
            let node = &self.nodes[index as usize];
            let plane = &self.planes[node.plane as usize];

            if plane.point_distance(pos) >= 0.0
            {
                index = node.front;
            }
            else
            {
                index = node.back;
            }
        }

        // Leaf indexes are stored as negative two's complement values, so convert them here
        !index as usize
    }

    pub fn traverse_front_to_back(&self, position: Vector3<f32>, visit_node: fn(usize, &Node) -> bool, visit_leaf: fn(usize, &Leaf))
    {
        let front_to_back = |plane: &Plane| plane.point_distance(position) >= 0.0;
        self.traverse_impl(0, &front_to_back, visit_node, visit_leaf);
    }

    pub fn traverse_back_to_front(&self, position: Vector3<f32>, visit_node: fn(usize, &Node) -> bool, visit_leaf: fn(usize, &Leaf))
    {
        let back_to_front = |plane: &Plane| plane.point_distance(position) < 0.0;
        self.traverse_impl(0, &back_to_front, visit_node, visit_leaf);
    }

    fn traverse_impl(&self, node_index: i32, front_first: &dyn Fn(&Plane) -> bool, visit_node: fn(usize, &Node) -> bool, visit_leaf: fn(usize, &Leaf))
    {
        if node_index < 0
        {
            let leaf_index = !node_index as usize;
            visit_leaf(leaf_index, &self.leafs[leaf_index]);
            return;
        }

        let node = &self.nodes[node_index as usize];
        if !visit_node(node_index as usize, node)
        {
            // Allow traversal of this side of the tree to be aborted, e.g. if the node's bounds fall outside of view
            return;
        }

        let first: i32;
        let last: i32;
        let plane = &self.planes[node.plane as usize];

        if front_first(plane)
        {
            first = node.front;
            last = node.back;
        }
        else
        {
            first = node.back;
            last = node.front;
        }

        self.traverse_impl(first, front_first, visit_node, visit_leaf);
        self.traverse_impl(last, front_first, visit_node, visit_leaf);
    }

    pub fn cluster_visible(&self, cluster: i32, test: i32) -> bool
    {
        if cluster < 0 { return true; } // If we're outside the map, everything will be considered visible

        // PVS data contains cross-references between all leafs, tightly packed into bit strings.
        // To decode it requires some nasty wrangling with bitwire operators that I'm not going to bother trying to understand right now.
        let vec_index = cluster * self.visdata_info.vector_size + (test >> 3);
        let vec = &self.visdata_vecs[vec_index as usize];
        (vec & (1 << (test & 7))) != 0
    }
}

impl fmt::Display for World
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
    {
        writeln!(f, "{} vertices", self.vertices.len())?;
        writeln!(f, "{} surfaces", self.surfaces.len())?;
        writeln!(f, "{} textures", self.textures.len())?;
        writeln!(f, "{} lightmaps", self.lightmaps.len())?;
        writeln!(f, "{} nodes", self.nodes.len())?;
        writeln!(f, "{} leafs", self.leafs.len())?;
        writeln!(f, "{} leaf surfaces", self.leaf_surfaces.len())?;
        writeln!(f, "{} planes", self.planes.len())?;
        writeln!(f, "{} visdata bytes", self.visdata_vecs.len())?;
        writeln!(f, "{} entity chars", self.entities.len())?;
        writeln!(f, "{} brushes", self.brushes.len())?;
        writeln!(f, "{} leaf brushes", self.leaf_brushes.len())?;
        writeln!(f, "{} brush sides", self.brush_sides.len())?;
        writeln!(f, "{} models", self.models.len())?;
        writeln!(f, "{} indices", self.indices.len())?;
        writeln!(f, "{} fogs", self.fogs.len())?;
        writeln!(f, "{} light volumes", self.light_volumes.len())?;
        
        write!(f, "BSP tree depth: {}", self.tree_depth())
    }
}

// size_of<> cannot be used with generic type arguments, so instead of using generic functions we have to resort to macros here.
// It's not too bad actually; these macros came out pretty clean and are just as easy to use as a generic function.
// Also pardon the use of transmute() here which is just about the most unsafe thing you can do in Rust, but these data formats were 
// originally designed for C and are meant to be memory mapped. Transmute is the most natural way of replicating this.
macro_rules! read_struct
{
    ($f:expr, $item:ty) =>
    {{
        let result: $item =
        {
            let mut buf = [0u8; size_of::<$item>()];    // Creates a byte array of fixed size, filled with zeroes
            $f.read_exact(&mut buf[..])?;
            unsafe { transmute(buf) }   // Not very idiomatic Rust, is it?
        };

        result
    }};
}

macro_rules! read_array
{
    ($f:expr, $count:expr, $item:ty) =>
    {{
        let items: [$item; $count as usize] =
        {
            let mut buf = [0u8; size_of::<$item>() * $count as usize];
            $f.read_exact(&mut buf[..])?;
            unsafe { transmute(buf) }
        };

        items
    }};
}

// Reading and adding items one-by-one is not as efficient as mapping an entire array from file into memory at once
// but since arrays must have a size that is known at compile-time, we can only use Vecs here.
// We can probably do something hacky by getting a pointer to the Vec's buffer and mapping the file contents directly to that,
// but I don't want to stray TOO far from idiomatic Rust straight away.
macro_rules! read_lump_vec
{
    ($f:expr, $lump:expr, $item:ty) =>
    {{
        $f.seek(SeekFrom::Start($lump.offset as u64))?;
        let num_items = $lump.length as usize / size_of::<$item>();
        let mut items: Vec<$item> = Vec::with_capacity(num_items);
    
        for _ in 0..num_items
        {
            let it = read_struct!($f, $item);
            items.push(it);
        }

        items
    }};
}

// Interestingly, Rust has both String and str types for dealing with strings.
// From what I understand, the difference is very similar to the difference between Vec and arrays.
// As in: one is used for long-term storage of dynamic data (String/Vec) while the other is meant more for short-term use within a limited scope (str/array).
macro_rules! read_lump_str
{
    ($f:expr, $lump:expr) =>
    {{
        $f.seek(SeekFrom::Start($lump.offset as u64))?;
        let mut buffer = String::new();
        $f.take($lump.length as u64).read_to_string(&mut buffer)?;  // I like this. take() gives you a new reader based on the provided length, and then you can just read that to its end.
        buffer
    }}
}

pub fn load_world(file: &mut File) -> std::io::Result<World>
{
    println!("Entities = {}, Textures = {}, MaxLumps = {}", LumpType::Entities as i32, LumpType::Textures as i32, LumpType::MaxLumps as i32);

    let header = read_struct!(file, Header);
    println!("{:?}", header);

    let id = std::str::from_utf8(&header.id);
    if id.is_err() || id.unwrap() != "IBSP" || header.version != 0x2E
    {
        return Err(Error::new(ErrorKind::InvalidData, "Invalid BSP header"));
    }

    let lumps = read_array!(file, LumpType::MaxLumps, Lump);
    println!("{:?}", lumps);

    let mut world = World::default();

    let lump = &lumps[LumpType::Vertices as usize];
    world.vertices = read_lump_vec!(file, lump, Vertex);

    let lump = &lumps[LumpType::Surfaces as usize];
    world.surfaces = read_lump_vec!(file, lump, Surface);

    let lump = &lumps[LumpType::Textures as usize];
    world.textures = read_lump_vec!(file, lump, Texture);

    let lump = &lumps[LumpType::Lightmaps as usize];
    world.lightmaps = read_lump_vec!(file, lump, Lightmap);

    let lump = &lumps[LumpType::Nodes as usize];
    world.nodes = read_lump_vec!(file, lump, Node);

    let lump = &lumps[LumpType::Leafs as usize];
    world.leafs = read_lump_vec!(file, lump, Leaf);

    let lump = &lumps[LumpType::LeafSurfaces as usize];
    world.leaf_surfaces = read_lump_vec!(file, lump, i32);

    let lump = &lumps[LumpType::Planes as usize];
    world.planes = read_lump_vec!(file, lump, Plane);

    let lump = &lumps[LumpType::Visibility as usize];
    file.seek(SeekFrom::Start(lump.offset as u64))?;
    world.visdata_info = read_struct!(file, VisDataInfo);
    file.take(world.visdata_info.num_vectors as u64 * world.visdata_info.vector_size as u64).read_to_end(&mut world.visdata_vecs)?;

    let lump = &lumps[LumpType::Entities as usize];
    world.entities = read_lump_str!(file, lump);

    let lump = &lumps[LumpType::Brushes as usize];
    world.brushes = read_lump_vec!(file, lump, Brush);

    let lump = &lumps[LumpType::LeafBrushes as usize];
    world.leaf_brushes = read_lump_vec!(file, lump, i32);

    let lump = &lumps[LumpType::BrushSides as usize];
    world.brush_sides = read_lump_vec!(file, lump, BrushSide);

    let lump = &lumps[LumpType::Models as usize];
    world.models = read_lump_vec!(file, lump, Model);

    let lump = &lumps[LumpType::Indices as usize];
    world.indices = read_lump_vec!(file, lump, u32);

    let lump = &lumps[LumpType::Fogs as usize];
    world.fogs = read_lump_vec!(file, lump, Fog);

    let lump = &lumps[LumpType::LightVolumes as usize];
    world.light_volumes = read_lump_vec!(file, lump, LightVolume);

    Ok(world)
}
