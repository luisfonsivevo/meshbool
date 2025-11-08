use crate::common::AABB;
use crate::utils::{K_PRECISION, mat3, next3_usize};
use core::f64;
use nalgebra::{Matrix2x3, Matrix3, Matrix3x4, Point3, Vector3};
use std::ops::MulAssign;

#[inline]
pub fn max_epsilon(min_epsilon: f64, bbox: &AABB) -> f64 {
	let epsilon = min_epsilon.max(K_PRECISION * bbox.scale());
	if epsilon.is_finite() { epsilon } else { -1.0 }
}

#[inline]
pub fn next_halfedge(mut current: i32) -> i32 {
	current += 1;
	if current % 3 == 0 {
		current -= 3;
	}
	current
}

pub fn normal_transform(transform: &Matrix3x4<f64>) -> Matrix3<f64> {
	mat3(transform)
		.transpose()
		.try_inverse()
		.unwrap_or_else(|| Matrix3::from_element(f64::NAN))
}

///By using the closest axis-aligned projection to the normal instead of a
///projection along the normal, we avoid introducing any rounding error.
#[inline]
pub fn get_axis_aligned_projection(normal: Vector3<f64>) -> Matrix2x3<f64> {
	let abs_normal = normal.abs();
	let (xyz_max, mut projection) = if abs_normal.z > abs_normal.x && abs_normal.z > abs_normal.y {
		(normal.z, Matrix2x3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
	} else if abs_normal.y > abs_normal.x {
		(normal.y, Matrix2x3::new(0.0, 0.0, 1.0, 1.0, 0.0, 0.0))
	} else {
		(normal.x, Matrix2x3::new(0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
	};

	if xyz_max < 0.0 {
		projection.row_mut(0).mul_assign(-1.0);
	}
	projection
}

#[inline]
pub fn get_barycentric(v: &Point3<f64>, tri_pos: &Matrix3<f64>, tolerance: f64) -> Vector3<f64> {
	let edges = Matrix3::from_columns(&[
		tri_pos.column(2) - tri_pos.column(1),
		tri_pos.column(0) - tri_pos.column(2),
		tri_pos.column(1) - tri_pos.column(0),
	]);

	let d2 = Vector3::new(
		edges.column(0).magnitude_squared(),
		edges.column(1).magnitude_squared(),
		edges.column(2).magnitude_squared(),
	);

	let long_side = if d2[0] > d2[1] && d2[0] > d2[2] {
		0
	} else if d2[1] > d2[2] {
		1
	} else {
		2
	};

	let cross_p = edges.column(0).cross(&edges.column(1));
	let area2 = cross_p.magnitude_squared();
	let tol2 = tolerance.powi(2);

	let mut uvw = Vector3::default();
	for i in 0..3 {
		let dv = v - tri_pos.column(i);
		if dv.coords.magnitude_squared() < tol2 {
			// Return exactly equal if within tolerance of vert.
			uvw[i] = 1.0;
			return uvw;
		}
	}

	if d2[long_side] < tol2
	//point
	{
		return Vector3::new(1.0, 0.0, 0.0);
	} else if area2 > d2[long_side] * tol2
	//triangle
	{
		for i in 0..3 {
			let j = next3_usize(i);
			let cross_pv = edges.column(i).cross(&(v.coords - tri_pos.column(j)));
			let area_2v = cross_pv.magnitude_squared();
			// Return exactly equal if within tolerance of edge.
			uvw[i] = if area_2v < d2[i] * tol2 {
				0.0
			} else {
				cross_pv.dot(&cross_p)
			};
		}

		uvw /= uvw[0] + uvw[1] + uvw[2];
		return uvw;
	} else
	//line
	{
		let next_v = next3_usize(long_side);
		let alpha = (v - tri_pos.column(next_v))
			.coords
			.dot(&edges.column(long_side))
			/ d2[long_side];
		uvw[long_side] = 0.0;
		uvw[next_v] = 1.0 - alpha;
		let last_v = next3_usize(next_v);
		uvw[last_v] = alpha;
		return uvw;
	}
}

///The fundamental component of the halfedge data structure used for storing and
///operating on the Manifold.
#[derive(Default, Clone, Copy, Debug)]
pub struct Halfedge {
	pub start_vert: i32,
	pub end_vert: i32,
	pub paired_halfedge: i32,
	pub prop_vert: i32,
}

impl Halfedge {
	pub fn is_forward(&self) -> bool {
		self.start_vert < self.end_vert
	}
}

#[derive(Copy, Clone, Debug)]
pub struct TriRef {
	/// The unique ID of the mesh instance of this triangle. If .meshID and .tri
	/// match for two triangles, then they are coplanar and came from the same
	/// face.
	pub mesh_id: i32,
	/// The OriginalID of the mesh this triangle came from. This ID is ideal for
	/// reapplying properties like UV coordinates to the output mesh.
	pub original_id: i32,
	/// Probably the triangle index of the original triangle this was part of:
	/// Mesh.triVerts[tri], but it's an input, so just pass it along unchanged.
	pub face_id: i32,
	/// Triangles with the same coplanar ID are coplanar.
	pub coplanar_id: i32,
}

impl TriRef {
	pub fn same_face(&self, other: &TriRef) -> bool {
		self.mesh_id == other.mesh_id
			&& self.coplanar_id == other.coplanar_id
			&& self.face_id == other.face_id
	}
}
