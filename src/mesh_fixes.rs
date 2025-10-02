use std::mem;
use nalgebra::{Matrix3, Vector3};
use crate::shared::Halfedge;

#[inline]
fn flip_halfedge(halfedge: i32) -> i32
{
	let tri = halfedge / 3;
	let vert = 2 - (halfedge - 3 * tri);
	3 * tri + vert
}

pub fn transform_normal(transform: Matrix3::<f64>, mut normal: Vector3<f64>) -> Vector3<f64>
{
	normal = (transform * normal).normalize();
	if normal.x.is_nan()
	{
		return Vector3::zeros();
	}
	
	normal
}

pub struct FlipTris<'a>
{
	pub halfedge: &'a mut [Halfedge],
}

impl<'a> FlipTris<'a>
{
	pub fn call(&mut self, tri: usize)
	{
		let tmp = self.halfedge[3 * tri];
		self.halfedge[3 * tri] = self.halfedge[3 * tri + 2];
		self.halfedge[3 * tri + 2] = tmp;
		
		for i in 0..3
		{
			mem::swap(&mut self.halfedge[3 * tri + i].start_vert, &mut self.halfedge[3 * tri + i].end_vert);
			self.halfedge[3 * tri + i].paired_halfedge =
					flip_halfedge(self.halfedge[3 * tri + i].paired_halfedge);
		}
	}
}
