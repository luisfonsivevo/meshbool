use nalgebra::{Point3, Vector3};

use crate::AABB;
use crate::collider::Recorder;
use crate::meshboolimpl::MeshBoolImpl;
use crate::shared::{Halfedge, next_halfedge};

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
