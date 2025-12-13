use nalgebra::{Matrix2x3, Point3, Vector2, Vector3};

use crate::AABB;
use crate::collider::Recorder;
use crate::meshboolimpl::MeshBoolImpl;
use crate::shared::{Halfedge, get_axis_aligned_projection, next_halfedge};
use crate::utils::ccw;

struct CheckHalfedges<'a> {
	halfedges: &'a [Halfedge],
}

impl<'a> CheckHalfedges<'a> {
	fn call(&self, edge: usize) -> bool {
		let halfedge = self.halfedges[edge];
		if halfedge.start_vert == -1 && halfedge.end_vert == -1 && halfedge.paired_halfedge == -1 {
			return true;
		}

		if self.halfedges[next_halfedge(edge as i32) as usize].start_vert == -1
			|| self.halfedges[next_halfedge(next_halfedge(edge as i32)) as usize].start_vert == -1
		{
			return false;
		}

		if halfedge.paired_halfedge == -1 {
			return false;
		}

		let paired = self.halfedges[halfedge.paired_halfedge as usize];
		let mut good = true;
		good &= paired.paired_halfedge == edge as i32;
		good &= halfedge.start_vert != halfedge.end_vert;
		good &= halfedge.start_vert == paired.end_vert;
		good &= halfedge.end_vert == paired.start_vert;
		good
	}
}

struct CheckCCW<'a> {
	halfedges: &'a [Halfedge],
	vert_pos: &'a [Point3<f64>],
	tri_normal: &'a [Vector3<f64>],
	tol: f64,
}

impl<'a> CheckCCW<'a> {
	fn call(&self, face: usize) -> bool {
		if self.halfedges[3 * face].paired_halfedge < 0 {
			return true;
		}

		let projection: Matrix2x3<f64> = get_axis_aligned_projection(self.tri_normal[face].clone());
		let mut v: [Vector2<f64>; 3] = Default::default();
		for i in [0, 1, 2] {
			v[i] =
				projection * self.vert_pos[self.halfedges[3 * face + i].start_vert as usize].coords;
		}

		let ccw: i32 = ccw(v[0].into(), v[1].into(), v[2].into(), self.tol.abs());
		if self.tol > 0.0 { ccw >= 0 } else { ccw == 0 }
	}
}

impl MeshBoolImpl {
	/**
		* Returns true if this manifold is in fact an oriented even manifold and all of
		* the data structures are consistent.
		*/
	pub fn is_manifold(&self) -> bool {
		if self.halfedge.len() == 0 {
			return true;
		}
		if self.halfedge.len() % 3 != 0 {
			return false;
		}
		(0..self.halfedge.len()).all(|edge| {
			CheckHalfedges {
				halfedges: &self.halfedge,
			}
			.call(edge)
		})
	}

	///Returns true if this manifold is in fact an oriented 2-manifold and all of
	///the data structures are consistent.
	pub fn is_2_manifold(&self) -> bool {
		if !self.is_manifold() {
			return false;
		}

		let mut halfedge = self.halfedge.clone();
		halfedge.sort_by_key(|edge| (edge.start_vert, edge.end_vert));

		(0..(2 * self.num_edge() - 1)).all(|edge| {
			let h = halfedge[edge];
			if h.start_vert == -1 && h.end_vert == -1 && h.paired_halfedge == -1 {
				return true;
			}

			h.start_vert != halfedge[edge + 1].start_vert
				|| h.end_vert != halfedge[edge + 1].end_vert
		})
	}

	///Returns true if all triangles are CCW relative to their triNormals_.
	pub fn matches_tri_normals(&self) -> bool {
		if self.halfedge.len() == 0 || self.face_normal.len() != self.num_tri() {
			return true;
		}
		let c_ccw = CheckCCW {
			halfedges: &self.halfedge,
			vert_pos: &self.vert_pos,
			tri_normal: &self.face_normal,
			tol: 2.0 * self.epsilon,
		};
		return (0..self.num_tri()).all(|i| c_ccw.call(i));
	}

	///Returns the number of triangles that are colinear within epsilon_.
	pub fn num_degenerate_tris(&self) -> usize {
		if self.halfedge.len() == 0 || self.face_normal.len() != self.num_tri() {
			return 1;
		}
		let c_ccw = CheckCCW {
			halfedges: &self.halfedge,
			vert_pos: &self.vert_pos,
			tri_normal: &self.face_normal,
			tol: -1.0 * self.epsilon / 2.0,
		};
		return (0..self.num_tri()).filter(|i| c_ccw.call(*i)).count();
	}

	pub fn calculate_bbox(&mut self) {
		self.bbox.min = self.vert_pos.iter().fold(
			Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
			|a, &b| {
				if a.x.is_nan() {
					return b;
				}
				if b.x.is_nan() {
					return a;
				}
				a.inf(&b)
			},
		);

		self.bbox.max = self.vert_pos.iter().fold(
			Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
			|a, &b| {
				if a.x.is_nan() {
					return b;
				}
				if b.x.is_nan() {
					return a;
				}
				a.sup(&b)
			},
		);
	}

	///Determines if all verts are finite. Checking just the bounding box dimensions
	///is insufficient as it ignores NaNs.
	pub fn is_finite(&self) -> bool {
		!self
			.vert_pos
			.iter()
			.any(|v| v.iter().any(|f| !f.is_finite()))
	}

	///Returns the minimum gap between two manifolds. Returns a double between
	///0 and searchLength.
	pub fn min_gap(&self, other: &Self, search_length: f64) -> f64 {
		let mut face_box_other: Vec<AABB> = vec![];
		let mut face_morton_other: Vec<u32> = vec![];

		other.get_face_box_morton(&mut face_box_other, &mut face_morton_other);

		for aabb in face_box_other.iter_mut() {
			*aabb = AABB::new(
				(aabb.min.coords - Vector3::repeat(search_length)).into(),
				(aabb.max.coords + Vector3::repeat(search_length)).into(),
			);
		}

		let mut recorder = MinDistanceRecorder::new(&self, other);
		self.collider
			.collisions_from_slice::<_, MinDistanceRecorder>(&face_box_other, &mut recorder, false);
		let min_distance_squared = recorder.get().min(search_length * search_length);
		return min_distance_squared.sqrt();
	}
}

struct MinDistanceRecorder<'a> {
	this: &'a MeshBoolImpl,
	other: &'a MeshBoolImpl,
	result: f64,
}

impl Recorder for MinDistanceRecorder<'_> {
	fn record(&mut self, tri_other: i32, tri: i32) {
		let min_distance = &mut self.result;

		let mut p: [Vector3<f64>; 3] = Default::default();
		let mut q: [Vector3<f64>; 3] = Default::default();

		for j in [0, 1, 2] {
			p[j] = self.this.vert_pos[self.this.halfedge[3 * tri as usize + j].start_vert as usize]
				.coords;
			q[j] = self.other.vert_pos
				[self.other.halfedge[3 * tri_other as usize + j].start_vert as usize]
				.coords;
		}
		*min_distance =
			min_distance.min(crate::tri_dis::distance_triangle_triangle_squared(&p, &q));
	}
}

impl<'a> MinDistanceRecorder<'a> {
	fn get(&self) -> f64 {
		return self.result;
	}

	fn new(this: &'a MeshBoolImpl, other: &'a MeshBoolImpl) -> Self {
		Self {
			this,
			other,
			result: core::f64::INFINITY,
		}
	}
}
