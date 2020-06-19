use cgmath::{ Vector3, Matrix4 };
use cgmath::prelude::*;

#[derive(Debug, Default)]
pub struct Frustum
{
    planes: [Plane; PlaneType::MaxPlaneTypes as usize],
}

#[derive(Debug)]
struct Plane
{
    normal: Vector3<f32>,
    distance: f32,
}

#[repr(usize)]
enum PlaneType
{
    Right = 0,
    Left,
    Top,
    Bottom,
    Back,
    Front,

    MaxPlaneTypes
}

impl Frustum
{
    // I really ought to try and understand sometime how and why this works... but I can't be bothered right now
    pub fn update(&mut self, projection: &Matrix4<f32>, view: &Matrix4<f32>)
    {
        let clip = projection * view;

        let mut plane = &mut self.planes[PlaneType::Right as usize];
        plane.normal.x = clip.x.w - clip.x.x;
        plane.normal.y = clip.y.w - clip.y.x;
        plane.normal.z = clip.z.w - clip.z.x;
        plane.distance = clip.w.w - clip.w.x;
        plane.normalize();

        let mut plane = &mut self.planes[PlaneType::Left as usize];
        plane.normal.x = clip.x.w + clip.x.x;
        plane.normal.y = clip.y.w + clip.y.x;
        plane.normal.z = clip.z.w + clip.z.x;
        plane.distance = clip.w.w + clip.w.x;
        plane.normalize();

        let mut plane = &mut self.planes[PlaneType::Top as usize];
        plane.normal.x = clip.x.w - clip.x.y;
        plane.normal.y = clip.y.w - clip.y.y;
        plane.normal.z = clip.z.w - clip.z.y;
        plane.distance = clip.w.w - clip.w.y;
        plane.normalize();

        let mut plane = &mut self.planes[PlaneType::Bottom as usize];
        plane.normal.x = clip.x.w + clip.x.y;
        plane.normal.y = clip.y.w + clip.y.y;
        plane.normal.z = clip.z.w + clip.z.y;
        plane.distance = clip.w.w + clip.w.y;
        plane.normalize();

        let mut plane = &mut self.planes[PlaneType::Back as usize];
        plane.normal.x = clip.x.w - clip.x.z;
        plane.normal.y = clip.y.w - clip.y.z;
        plane.normal.z = clip.z.w - clip.z.z;
        plane.distance = clip.w.w - clip.w.z;
        plane.normalize();

        let mut plane = &mut self.planes[PlaneType::Front as usize];
        plane.normal.x = clip.x.w + clip.x.z;
        plane.normal.y = clip.y.w + clip.y.z;
        plane.normal.z = clip.z.w + clip.z.z;
        plane.distance = clip.w.w + clip.w.z;
        plane.normalize();
    }

    pub fn point_inside(&self, point: Vector3<f32>) -> bool
    {
        self.sphere_inside(point, 0.0)
    }

    pub fn sphere_inside(&self, center: Vector3<f32>, radius: f32) -> bool
    {
        for i in 0..(PlaneType::MaxPlaneTypes as usize)
        {
            if self.planes[i].point_distance(center) <= -radius
            {
                return false;
            }
        }

        true
    }

    pub fn box_inside(&self, mins: Vector3<f32>, maxs: Vector3<f32>) -> bool
    {
        for i in 0..(PlaneType::MaxPlaneTypes as usize)
        {
            let plane = &self.planes[i];
            if plane.point_distance(Vector3::new(mins.x, mins.y, mins.z)) > 0.0 { continue; }
            if plane.point_distance(Vector3::new(mins.x, mins.y, maxs.z)) > 0.0 { continue; }
            if plane.point_distance(Vector3::new(mins.x, maxs.y, mins.z)) > 0.0 { continue; }
            if plane.point_distance(Vector3::new(mins.x, maxs.y, maxs.z)) > 0.0 { continue; }
            if plane.point_distance(Vector3::new(maxs.x, mins.y, mins.z)) > 0.0 { continue; }
            if plane.point_distance(Vector3::new(maxs.x, mins.y, maxs.z)) > 0.0 { continue; }
            if plane.point_distance(Vector3::new(maxs.x, maxs.y, mins.z)) > 0.0 { continue; }
            if plane.point_distance(Vector3::new(maxs.x, maxs.y, maxs.z)) > 0.0 { continue; }

            return false;
        }

        true
    }

    pub fn box_inside_i32(&self, mins: Vector3<i32>, maxs: Vector3<i32>) -> bool
    {
        self.box_inside(Vector3::new(mins.x as f32, mins.y as f32, mins.z as f32), Vector3::new(maxs.x as f32, maxs.y as f32, maxs.z as f32))
    }
}

impl Default for Plane
{
    fn default() -> Self { Self { normal: Vector3::unit_z(), distance: 0.0 } }
}

impl Plane
{
    fn normalize(&mut self)
    {
        let div = 1.0 / self.normal.magnitude();
        self.normal *= div;
        self.distance *= div;
    }

    fn point_distance(&self, point: Vector3<f32>) -> f32
    {
        point.dot(self.normal) + self.distance
    }
}
