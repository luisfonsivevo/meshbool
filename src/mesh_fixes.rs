use crate::shared::Halfedge;
use nalgebra::{Matrix3, Vector3, Vector4};
use std::mem;

#[inline]
fn flip_halfedge(halfedge: i32) -> i32 {
	let tri = halfedge / 3;
	let vert = 2 - (halfedge - 3 * tri);
	3 * tri + vert
}

pub fn transform_normal(transform: Matrix3<f64>, mut normal: Vector3<f64>) -> Vector3<f64> {
	normal = (transform * normal).normalize();
	if normal.x.is_nan() {
		return Vector3::zeros();
	}

	normal
}

pub struct TransformTangents<'a> {
	pub tangent: &'a mut [Vector4<f64>],
	pub edge_offset: i32,
	pub transform: Matrix3<f64>,
	pub invert: bool,
	pub old_tangents: &'a [Vector4<f64>],
	pub halfedge: &'a [Halfedge],
}

impl<'a> TransformTangents<'a> {
	pub fn call(&mut self, edge_out: i32) {
		let edge_in = if self.invert {
			self.halfedge[flip_halfedge(edge_out) as usize].paired_halfedge
		} else {
			edge_out
		};
		self.tangent[(edge_out + self.edge_offset) as usize] = (self.transform
			* self.old_tangents[edge_in as usize].xyz())
		.push(self.old_tangents[edge_in as usize].w);
	}
}

pub struct FlipTris<'a> {
	pub halfedge: &'a mut [Halfedge],
}

impl<'a> FlipTris<'a> {
	pub fn call(&mut self, tri: usize) {
		let tmp = self.halfedge[3 * tri];
		self.halfedge[3 * tri] = self.halfedge[3 * tri + 2];
		self.halfedge[3 * tri + 2] = tmp;

		for i in 0..3 {
			mem::swap(
				&mut self.halfedge[3 * tri + i].start_vert,
				&mut self.halfedge[3 * tri + i].end_vert,
			);
			self.halfedge[3 * tri + i].paired_halfedge =
				flip_halfedge(self.halfedge[3 * tri + i].paired_halfedge);
		}
	}
}
