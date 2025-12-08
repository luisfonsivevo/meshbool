use nalgebra::{Matrix2x3, Point3, Vector2, Vector3};

use crate::AABB;
use crate::collider::Recorder;
use crate::meshboolimpl::MeshBoolImpl;
use crate::shared::{Halfedge, get_axis_aligned_projection, next_halfedge};
use crate::utils::{atomic_add_f64, ccw};

struct CheckHalfedges<'a> {
	halfedges: &'a [Halfedge],
}

#[derive(Eq, PartialEq)]
pub enum Property {
	Volume,
	SurfaceArea,
}

struct CurvatureAngles<'a> {
	mean_curvature: &'a mut [f64],
	gaussian_curvature: &'a mut [f64],
	area: &'a mut [f64],
	degree: &'a mut [f64],
	halfedge: &'a [Halfedge],
	vert_pos: &'a [Point3<f64>],
	tri_normal: &'a [Vector3<f64>],
}

impl<'a> CurvatureAngles<'a> {
	pub fn call(&mut self, tri: usize) {
		let mut edge: [Vector3<f64>; 3] = Default::default();
		let mut edge_length = Vector3::repeat(0.0_f64);
		for i in [0, 1, 2] {
			let start_vert: i32 = self.halfedge[3 * tri + i].start_vert;
			let end_vert: i32 = self.halfedge[3 * tri + i].end_vert;
			edge[i] = self.vert_pos[end_vert as usize] - self.vert_pos[start_vert as usize];
			edge_length[i] = edge[i].norm();
			edge[i] /= edge_length[i];
			let neighbor_tri: i32 = self.halfedge[3 * tri + i].paired_halfedge / 3;
			let dihedral: f64 = 0.25
				* edge_length[i]
				* self.tri_normal[tri]
					.cross(&self.tri_normal[neighbor_tri as usize])
					.dot(&edge[i])
					.asin();
			unsafe {
				atomic_add_f64(&mut self.mean_curvature[start_vert as usize], dihedral);
				atomic_add_f64(&mut self.mean_curvature[end_vert as usize], dihedral);
				atomic_add_f64(&mut self.degree[start_vert as usize], 1.0);
			}
		}

		let mut phi = Vector3::<f64>::default();
		phi[0] = (-edge[2].dot(&edge[0])).acos();
		phi[1] = (-edge[0].dot(&edge[1])).acos();
		phi[2] = core::f64::consts::PI - phi[0] - phi[1];
		let area3: f64 = edge_length[0] * edge_length[1] * edge[0].cross(&edge[1]).norm() / 6.0;

		for i in [0, 1, 2] {
			let vert: i32 = self.halfedge[3 * tri + i].start_vert;
			unsafe {
				atomic_add_f64(&mut self.gaussian_curvature[vert as usize], -phi[i]);
				atomic_add_f64(&mut self.area[vert as usize], area3);
			}
		}
	}
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

	pub fn get_property(&self, prop: Property) -> f64 {
		if self.is_empty() {
			return 0.0;
		}

		let volume = |tri: usize| {
			let v: Vector3<f64> = self.vert_pos[self.halfedge[3 * tri].start_vert as usize].coords;
			let cross_p: Vector3<f64> = (self.vert_pos
				[self.halfedge[3 * tri + 1].start_vert as usize]
				- v)
				.coords
				.cross(&(self.vert_pos[self.halfedge[3 * tri + 2].start_vert as usize] - v).coords);
			cross_p.dot(&v) / 6.0
		};

		let area = |tri: usize| {
			let v: Vector3<f64> = self.vert_pos[self.halfedge[3 * tri].start_vert as usize].coords;
			(self.vert_pos[self.halfedge[3 * tri + 1].start_vert as usize] - v)
				.coords
				.cross(&(self.vert_pos[self.halfedge[3 * tri + 2].start_vert as usize] - v).coords)
				.norm() / 2.0
		};

		// Kahan summation
		let mut value: f64 = 0.0;
		let mut value_compensation: f64 = 0.0;
		for i in 0..self.num_tri() {
			let value1: f64 = if prop == Property::SurfaceArea {
				area(i)
			} else {
				volume(i)
			};
			let t: f64 = value + value1;
			value_compensation += (value - t) + value1;
			value = t;
		}
		value += value_compensation;
		return value;
	}

	pub fn calculate_curvature(&mut self, gaussian_idx: i32, mean_idx: i32) {
		if self.is_empty() {
			return;
		}
		if gaussian_idx < 0 && mean_idx < 0 {
			return;
		}
		let mut vert_mean_curvature: Vec<f64> = vec![0.0; self.num_vert()];
		let mut vert_gaussian_curvature: Vec<f64> = vec![core::f64::consts::TAU; self.num_vert()];
		let mut vert_area: Vec<f64> = vec![0.0; self.num_vert()];
		let mut degree: Vec<f64> = vec![0.0; self.num_vert()];
		{
			let mut ca = CurvatureAngles {
				mean_curvature: &mut vert_mean_curvature,
				gaussian_curvature: &mut vert_gaussian_curvature,
				area: &mut vert_area,
				degree: &mut degree,
				halfedge: &self.halfedge,
				vert_pos: &self.vert_pos,
				tri_normal: &self.face_normal,
			};
			(0..self.num_tri()).for_each(|i| ca.call(i));
		}
		(0..self.num_vert()).for_each(|vert| {
			let factor: f64 = degree[vert] / (6.0 * vert_area[vert]);
			vert_mean_curvature[vert] *= factor;
			vert_gaussian_curvature[vert] *= factor;
		});

		let old_num_prop: i32 = self.num_prop() as i32;
		let num_prop: i32 = old_num_prop.max(gaussian_idx.max(mean_idx) + 1);
		let old_properties = self.properties.clone();
		self.properties = vec![0.0; num_prop as usize * self.num_prop_vert()];
		self.num_prop = num_prop;

		let mut counters: Vec<u8> = vec![0; self.num_prop_vert()];
		(0..self.num_tri()).for_each(|tri| {
			for i in [0, 1, 2] {
				let edge = &self.halfedge[3 * tri + i];
				let vert: i32 = edge.start_vert;
				let prop_vert: i32 = edge.prop_vert;

				let old = unsafe {
					use core::sync::atomic::{AtomicU8, Ordering};
					let ptr = &mut counters[prop_vert as usize] as *const u8 as *const AtomicU8;

					// Convert to a shared reference
					let atomic_ref = &*ptr;

					atomic_ref.swap(1u8, Ordering::SeqCst)
				};
				if old == 1 {
					continue;
				}

				for p in 0..old_num_prop {
					self.properties[(num_prop * prop_vert + p) as usize] =
						old_properties[(old_num_prop * prop_vert + p) as usize];
				}

				if gaussian_idx >= 0 {
					self.properties[(num_prop * prop_vert + gaussian_idx) as usize] =
						vert_gaussian_curvature[vert as usize];
				}
				if mean_idx >= 0 {
					self.properties[(num_prop * prop_vert + mean_idx) as usize] =
						vert_mean_curvature[vert as usize];
				}
			}
		});
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
