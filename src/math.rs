use std::ops;

#[derive(Debug, Default)]
#[repr(C)]
pub struct Vector2
{
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Default)]
#[repr(C)]
pub struct Vector2i
{
    pub x: i32,
    pub y: i32,
}

#[derive(Debug, Default)]
#[repr(C)]
pub struct Vector3
{
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

// Just some basic 3D math routines for Vector3. Pretty self-explanatory.
impl Vector3
{
    pub fn new() -> Vector3 { Vector3 { x: 0.0, y: 0.0, z: 0.0 } }

    pub fn copy(src: &Vector3) -> Vector3 { Vector3 { x: src.x, y: src.y, z: src.z } }

    pub fn set(&mut self, x: f32, y: f32, z: f32) { self.x = x; self.y = y; self.z = z; }

    pub fn length_sqr(&self) -> f32 { self.x * self.x + self.y * self.y + self.z * self.z }

    pub fn length(&self) -> f32 { (self.x * self.x + self.y * self.y + self.z * self.z).sqrt() }

    pub fn normalize(&mut self)
    {
        let len = self.length();
        self.x = self.x / len;
        self.y = self.y / len;
        self.z = self.z / len;
    }

    pub fn normalized(&self) -> Vector3
    {
        let len = self.length();
        Vector3 { x: self.x / len, y: self.y / len, z: self.z / len }
    }

    pub fn dot(&self, other: &Vector3) -> f32 { self.x * other.x + self.y * other.y + self.z * other.z }

    pub fn mul_add(&self, scale: f32, other: &Vector3) -> Vector3
    {
        Vector3 
        {
            x: self.x.mul_add(scale, other.x),
            y: self.y.mul_add(scale, other.y),
            z: self.z.mul_add(scale, other.z),
        }
    }

    pub fn distance_sqr(&self, other: &Vector3) -> f32
    {
        let dx = other.x - self.x;
        let dy = other.y - self.y;
        let dz = other.z - self.z;
        dx * dx + dy * dy + dz * dz
    }

    pub fn distance(&self, other: &Vector3) -> f32
    {
        let dx = other.x - self.x;
        let dy = other.y - self.y;
        let dz = other.z - self.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

// Operator overloads for Vector3. Again, pretty self-explanatory.
impl ops::Add for Vector3
{
    type Output = Vector3;

    fn add(self, other: Vector3) -> Vector3
    {
        Vector3 { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z }
    }
}

impl ops::Sub for Vector3
{
    type Output = Vector3;

    fn sub(self, other: Vector3) -> Vector3
    {
        Vector3 { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z }
    }
}

impl ops::Mul<f32> for Vector3
{
    type Output = Vector3;

    fn mul(self, scale: f32) -> Vector3
    {
        Vector3 { x: self.x * scale, y: self.y * scale, z: self.z * scale }
    }
}

impl ops::Neg for Vector3
{
    type Output = Vector3;

    fn neg(self) -> Vector3
    {
        Vector3 { x: -self.x, y: -self.y, z: -self.z }
    }
}

#[derive(Debug, Default)]
#[repr(C)]
pub struct Vector3i
{
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

#[derive(Debug, Default)]
#[repr(C)]
pub struct Color3
{
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

#[derive(Debug, Default)]
#[repr(C)]
pub struct Color4
{
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}
