use crate::collider::{Recorder, SimpleRecorder};
use crate::common::{AABB, AABBOverlap, OpType};
use crate::disjoint_sets::DisjointSets;
use crate::meshboolimpl::MeshBoolImpl;
use crate::shared::Halfedge;
use crate::utils::permute;
use core::f64;
use nalgebra::{Point3, Vector2, Vector3, Vector4};
use std::collections::HashSet;
use std::mem;
use std::ops::DerefMut;

// These two functions (interpolate and intersect) are the only places where
// floating-point operations take place in the whole Boolean function. These
// are carefully designed to minimize rounding error and to eliminate it at edge
// cases to ensure consistency.

//given a line segment and x coordinate, find y and z coordinates
fn interpolate(p_l: Point3<f64>, p_r: Point3<f64>, x: f64) -> Vector2<f64> {
	let dx_l = x - p_l.x;
	let dx_r = x - p_r.x;
	debug_assert!(dx_l * dx_r <= 0.0, "Boolean manifold error: not in domain");

	let use_l = dx_l.abs() < dx_r.abs();
	let d_lr = p_r - p_l;
	let lambda = (if use_l { dx_l } else { dx_r }) / d_lr.x;
	if !lambda.is_finite() || !d_lr.y.is_finite() || !d_lr.z.is_finite() {
		return Vector2::new(p_l.y, p_l.z);
	}

	let mut yz = Vector2::default();
	yz[0] = lambda * d_lr.y + (if use_l { p_l.y } else { p_r.y });
	yz[1] = lambda * d_lr.z + (if use_l { p_l.z } else { p_r.z });
	return yz;
}

fn intersect(
	p_l: &Point3<f64>,
	p_r: &Point3<f64>,
	q_l: &Point3<f64>,
	q_r: &Point3<f64>,
) -> Vector4<f64> {
	let dyl = q_l.y - p_l.y;
	let dyr = q_r.y - p_r.y;
	debug_assert!(dyl * dyr <= 0.0, "Boolean manifold error: no intersection");
	let use_l = dyl.abs() < dyr.abs();
	let dx = p_r.x - p_l.x;
	let mut lambda = (if use_l { dyl } else { dyr }) / (dyl - dyr);
	if !lambda.is_finite() {
		lambda = 0.0;
	}
	let mut xyzz = Vector4::default();
	xyzz.x = lambda * dx + (if use_l { p_l.x } else { p_r.x });
	let pdy = p_r.y - p_l.y;
	let qdy = q_r.y - q_l.y;
	let use_p = pdy.abs() < qdy.abs();
	xyzz.y = lambda * (if use_p { pdy } else { qdy })
		+ (if use_l {
			if use_p { p_l.y } else { q_l.y }
		} else {
			if use_p { p_r.y } else { q_r.y }
		});
	xyzz.z = lambda * (p_r.z - p_l.z) + (if use_l { p_l.z } else { p_r.z });
	xyzz.w = lambda * (q_r.z - q_l.z) + (if use_l { q_l.z } else { q_r.z });
	xyzz
}

#[inline]
fn shadows(p: f64, q: f64, sym_pert: bool) -> bool {
	if p == q { sym_pert } else { p < q }
}

#[inline]
fn shadow01(
	p0: usize,
	q1: usize,
	vert_pos_p: &[Point3<f64>],
	vert_pos_q: &[Point3<f64>],
	halfedge_q: &[Halfedge],
	sym_pert_map: &[Vector3<bool>],
	reverse: bool,
) -> (i32, Vector2<f64>) {
	let p0_pos = vert_pos_p[p0];
	let q1s_vert_idx = halfedge_q[q1].start_vert as usize;
	let q1e_vert_idx = halfedge_q[q1].end_vert as usize;
	let q1s_pos = vert_pos_q[q1s_vert_idx];
	let q1e_pos = vert_pos_q[q1e_vert_idx];

	//1D vertex-edge intersection test
	let mut s01 = if reverse {
		shadows(q1s_pos.x, p0_pos.x, sym_pert_map[q1s_vert_idx].x) as i32
			- shadows(q1e_pos.x, p0_pos.x, sym_pert_map[q1e_vert_idx].x) as i32
	} else {
		shadows(p0_pos.x, q1e_pos.x, sym_pert_map[p0].x) as i32
			- shadows(p0_pos.x, q1s_pos.x, sym_pert_map[p0].x) as i32
	};

	let mut yz01_pos = Vector2::from_element(f64::NAN);

	if s01 != 0 {
		yz01_pos = interpolate(q1s_pos, q1e_pos, p0_pos.x);
		if reverse {
			let mut diff = q1s_pos - p0_pos;
			let start2 = diff.magnitude_squared();
			diff = q1e_pos - p0_pos;
			let end2 = diff.magnitude_squared();
			let yz01_sym_pert = if start2 < end2 {
				sym_pert_map[q1s_vert_idx].y
			} else {
				sym_pert_map[q1e_vert_idx].y
			};

			if !shadows(yz01_pos[0], p0_pos.y, yz01_sym_pert) {
				s01 = 0;
			}
		} else {
			if !shadows(p0_pos.y, yz01_pos[0], sym_pert_map[p0].y) {
				s01 = 0;
			}
		}
	}

	(s01, yz01_pos)
}

//2D edge-edge intersection test
struct Kernel11<'a> {
	vert_pos_p: &'a [Point3<f64>],
	vert_pos_q: &'a [Point3<f64>],
	halfedge_p: &'a [Halfedge],
	halfedge_q: &'a [Halfedge],
	sym_pert_map: &'a [Vector3<bool>],
}

impl<'a> Kernel11<'a> {
	fn call(&self, p1: usize, q1: usize) -> (i32, Vector4<f64>) {
		let xyzz11;
		let mut s11 = 0;

		// For pRL[k], qRL[k], k==0 is the left and k==1 is the right.
		let mut k = 0;
		let mut p_rl = [Point3::<f64>::default(); 2];
		let mut q_rl = [Point3::<f64>::default(); 2];
		// Either the left or right must shadow, but not both. This ensures the
		// intersection is between the left and right.
		let mut shadows_var = false;

		let p0 = [
			self.halfedge_p[p1].start_vert as usize,
			self.halfedge_p[p1].end_vert as usize,
		];

		for i in 0..p0.len() {
			let (s01, yz01) = shadow01(
				p0[i],
				q1,
				self.vert_pos_p,
				self.vert_pos_q,
				self.halfedge_q,
				self.sym_pert_map,
				false,
			);
			// If the value is NaN, then these do not overlap.
			if yz01[0].is_finite() {
				s11 += s01 * (if i == 0 { -1 } else { 1 });
				if k < 2 && (k == 0 || (s01 != 0) != shadows_var) {
					shadows_var = s01 != 0;
					p_rl[k] = self.vert_pos_p[p0[i]];
					q_rl[k] = Point3::new(p_rl[k].x, yz01.x, yz01.y);
					k += 1;
				}
			}
		}

		let q0 = [
			self.halfedge_q[q1].start_vert as usize,
			self.halfedge_q[q1].end_vert as usize,
		];

		for i in 0..q0.len() {
			let (s10, yz10) = shadow01(
				q0[i],
				p1,
				self.vert_pos_q,
				self.vert_pos_p,
				self.halfedge_p,
				self.sym_pert_map,
				true,
			);
			// If the value is NaN, then these do not overlap.
			if yz10[0].is_finite() {
				s11 += s10 * (if i == 0 { -1 } else { 1 });
				if k < 2 && (k == 0 || (s10 != 0) != shadows_var) {
					shadows_var = s10 != 0;
					q_rl[k] = self.vert_pos_q[q0[i]];
					p_rl[k] = Point3::new(q_rl[k].x, yz10.x, yz10.y);
					k += 1;
				}
			}
		}

		if s11 == 0
		//no intersection
		{
			xyzz11 = Vector4::from_element(f64::NAN);
		} else {
			debug_assert!(k == 2, "Boolean manifold error: s11");
			xyzz11 = intersect(&p_rl[0], &p_rl[1], &q_rl[0], &q_rl[1]);

			let p1s = self.halfedge_p[p1].start_vert as usize;
			let p1e = self.halfedge_p[p1].end_vert as usize;
			let mut diff = self.vert_pos_p[p1s] - xyzz11.xyz();
			let start2 = diff.coords.magnitude_squared();
			diff = self.vert_pos_p[p1e] - xyzz11.xyz();
			let end2 = diff.coords.magnitude_squared();
			let xyzz11_sym_pert = if start2 < end2 {
				self.sym_pert_map[p1s].z
			} else {
				self.sym_pert_map[p1e].z
			};

			if !shadows(xyzz11.z, xyzz11.w, xyzz11_sym_pert) {
				s11 = 0;
			}
		}

		(s11, xyzz11)
	}
}

//2D vertex-triangle intersection test
struct Kernel02<'a> {
	vert_pos_p: &'a [Point3<f64>],
	halfedge_q: &'a [Halfedge],
	vert_pos_q: &'a [Point3<f64>],
	sym_pert_map: &'a [Vector3<bool>],
	forward: bool,
}

impl<'a> Kernel02<'a> {
	fn call(&self, p0: usize, q2: usize) -> (i32, f64) {
		let mut s02 = 0;
		let z02;

		// For yzzLR[k], k==0 is the left and k==1 is the right.
		let mut k = 0;
		let mut yzz_rl = [Point3::<f64>::default(); 2];
		// Either the left or right must shadow, but not both. This ensures the
		// intersection is between the left and right.
		let mut shadows_var = false;
		let mut closest_vert_q = -1;
		let mut min_metric = f64::INFINITY;

		let pos_p = self.vert_pos_p[p0];
		for i in 0..3 {
			let q1 = 3 * q2 + i;
			let edge = self.halfedge_q[q1];
			let q1_f = if edge.is_forward() {
				q1
			} else {
				edge.paired_halfedge as usize
			};

			if !self.forward {
				let q_vert = self.halfedge_q[q1_f].start_vert;
				let diff = pos_p - self.vert_pos_q[q_vert as usize];
				let metric = diff.magnitude_squared();
				if metric < min_metric {
					min_metric = metric;
					closest_vert_q = q_vert;
				}
			}

			let syz01 = shadow01(
				p0,
				q1_f,
				self.vert_pos_p,
				self.vert_pos_q,
				self.halfedge_q,
				self.sym_pert_map,
				!self.forward,
			);
			let s01 = syz01.0;
			let yz01 = syz01.1;
			// If the value is NaN, then these do not overlap.
			if yz01[0].is_finite() {
				s02 += s01
					* (if self.forward == edge.is_forward() {
						-1
					} else {
						1
					});
				if k < 2 && (k == 0 || (s01 != 0) != shadows_var) {
					shadows_var = s01 != 0;
					yzz_rl[k] = Point3::new(yz01[0], yz01[1], yz01[1]);
					k += 1;
				}
			}
		}

		if s02 == 0
		//no intersection
		{
			z02 = f64::NAN;
		} else {
			debug_assert!(k == 2, "Boolean manifold error: s02");
			z02 = interpolate(yzz_rl[0], yzz_rl[1], pos_p.y)[1];
			if self.forward {
				if !shadows(pos_p.z, z02, self.sym_pert_map[p0].z) {
					s02 = 0;
				}
			} else {
				// DEBUG_ASSERT(closestVert != -1, topologyErr, "No closest vert");
				if !shadows(z02, pos_p.z, self.sym_pert_map[closest_vert_q as usize].z) {
					s02 = 0;
				}
			}
		}

		(s02, z02)
	}
}

//3D edge-triangle intersection test
struct Kernel12<'a> {
	halfedges_p: &'a [Halfedge],
	halfedges_q: &'a [Halfedge],
	vert_pos_p: &'a [Point3<f64>],
	forward: bool,
	k02: Kernel02<'a>,
	k11: Kernel11<'a>,
}

impl<'a> Kernel12<'a> {
	fn call(&self, p1: usize, q2: usize) -> (i32, Point3<f64>) {
		let mut x12 = 0;
		let mut v12 = Point3::new(f64::NAN, f64::NAN, f64::NAN);

		// For xzy_lr-[k], k==0 is the left and k==1 is the right.
		let mut k = 0;
		let mut xzy_lr0 = [Point3::<f64>::default(); 2];
		let mut xzy_lr1 = [Point3::<f64>::default(); 2];
		// Either the left or right must shadow, but not both. This ensures the
		// intersection is between the left and right.
		let mut shadows_var = false;

		let edge = self.halfedges_p[p1];

		for vert in [edge.start_vert, edge.end_vert] {
			let (s, z) = self.k02.call(vert as usize, q2);
			if z.is_finite() {
				x12 += s
					* (if (vert == edge.start_vert) == self.forward {
						1
					} else {
						-1
					});
				if k < 2 && (k == 0 || (s != 0) != shadows_var) {
					shadows_var = s != 0;
					xzy_lr0[k] = self.vert_pos_p[vert as usize];
					let switcheroo = xzy_lr0[k].deref_mut();
					mem::swap(&mut switcheroo.y, &mut switcheroo.z);
					xzy_lr1[k] = xzy_lr0[k];
					xzy_lr1[k][1] = z;
					k += 1;
				}
			}
		}

		for i in 0..3 {
			let q1 = 3 * q2 + i;
			let edge = self.halfedges_q[q1];
			let q1_f = if edge.is_forward() {
				q1
			} else {
				edge.paired_halfedge as usize
			};
			let (s, xyzz) = if self.forward {
				self.k11.call(p1, q1_f)
			} else {
				self.k11.call(q1_f, p1)
			};
			if xyzz[0].is_finite() {
				x12 -= s * (if edge.is_forward() { 1 } else { -1 });
				if k < 2 && (k == 0 || (s != 0) != shadows_var) {
					shadows_var = s != 0;
					xzy_lr0[k][0] = xyzz.x;
					xzy_lr0[k][1] = xyzz.z;
					xzy_lr0[k][2] = xyzz.y;
					xzy_lr1[k] = xzy_lr0[k];
					xzy_lr1[k][1] = xyzz.w;
					if !self.forward {
						mem::swap(&mut xzy_lr0[k][1], &mut xzy_lr1[k][1]);
					}
					k += 1;
				}
			}
		}

		if x12 == 0
		//no intersection
		{
			v12 = Point3::new(f64::NAN, f64::NAN, f64::NAN);
		} else {
			debug_assert!(k == 2, "Boolean manifold error: v12");
			let xzyy = intersect(&xzy_lr0[0], &xzy_lr0[1], &xzy_lr1[0], &xzy_lr1[1]);
			v12.x = xzyy[0];
			v12.y = xzyy[2];
			v12.z = xzyy[1];
		}

		(x12, v12)
	}
}

#[derive(Default)]
pub struct Kernel12Tmp {
	p1q2: Vec<[i32; 2]>,
	x12: Vec<i32>,
	v12: Vec<Point3<f64>>,
}

pub struct Kernel12Recorder<'a> {
	k12: &'a Kernel12<'a>,
	forward: bool,
	local_store: Kernel12Tmp,
}

impl<'a> Recorder for Kernel12Recorder<'a> {
	fn record(&mut self, query_idx: i32, leaf_idx: i32) {
		let tmp = &mut self.local_store;
		let (x12, v12) = self.k12.call(query_idx as usize, leaf_idx as usize);
		if v12[0].is_finite() {
			if self.forward {
				tmp.p1q2.push([query_idx, leaf_idx]);
			} else {
				tmp.p1q2.push([leaf_idx, query_idx]);
			}

			tmp.x12.push(x12);
			tmp.v12.push(v12);
		}
	}
}

fn intersect12(
	in_p: &MeshBoolImpl,
	in_q: &MeshBoolImpl,
	sym_pert_map: &[Vector3<bool>],
	forward: bool,
) -> (Vec<[i32; 2]>, Vec<i32>, Vec<Point3<f64>>) {
	// a: 1 (edge), b: 2 (face)
	let a = if forward { in_p } else { in_q };
	let b = if forward { in_q } else { in_p };

	let k02 = Kernel02 {
		vert_pos_p: &a.vert_pos,
		halfedge_q: &b.halfedge,
		vert_pos_q: &b.vert_pos,
		sym_pert_map,
		forward,
	};
	let k11 = Kernel11 {
		vert_pos_p: &in_p.vert_pos,
		vert_pos_q: &in_q.vert_pos,
		halfedge_p: &in_p.halfedge,
		halfedge_q: &in_q.halfedge,
		sym_pert_map,
	};

	let k12 = Kernel12 {
		halfedges_p: &a.halfedge,
		halfedges_q: &b.halfedge,
		vert_pos_p: &a.vert_pos,
		forward,
		k02,
		k11,
	};
	let mut recorder = Kernel12Recorder {
		k12: &k12,
		forward,
		local_store: Kernel12Tmp::default(),
	};
	let f = |i| {
		let i = i as usize;
		if a.halfedge[i].is_forward() {
			AABB::new(
				a.vert_pos[a.halfedge[i].start_vert as usize],
				a.vert_pos[a.halfedge[i].end_vert as usize],
			)
		} else {
			AABB::default()
		}
	};

	b.collider
		.collisions_from_fn::<false, _, _, Kernel12Recorder>(f, a.halfedge.len(), &mut recorder);

	let result = recorder.local_store;
	let mut p1q2 = result.p1q2;
	let mut x12 = result.x12;
	let mut v12 = result.v12;
	// sort p1q2 according to edges
	let mut i12: Vec<_> = (0..p1q2.len()).collect();

	let index = if forward { 0 } else { 1 };
	i12.sort_by_key(|&i| (p1q2[i][index], p1q2[i][1 - index]));
	permute(&mut p1q2, &i12);
	permute(&mut x12, &i12);
	permute(&mut v12, &i12);
	(p1q2, x12, v12)
}

//3D vertex-solid intersection test
fn winding03(
	in_p: &MeshBoolImpl,
	in_q: &MeshBoolImpl,
	p1q2: &[[i32; 2]],
	sym_pert_map: &[Vector3<bool>],
	forward: bool,
) -> Vec<i32> {
	let a = if forward { in_p } else { in_q };
	let b = if forward { in_q } else { in_p };
	let index = if forward { 0 } else { 1 };

	let u_a = DisjointSets::new(a.vert_pos.len() as u32);
	for edge in 0..a.halfedge.len() {
		let he = &a.halfedge[edge];
		let edge = edge as i32;
		if !he.is_forward() {
			continue;
		}
		// check if the edge is broken
		let it = p1q2.partition_point(|collision_pair| collision_pair[index] < edge);
		if it == p1q2.len() || p1q2[it][index] != edge {
			u_a.unite(he.start_vert as u32, he.end_vert as u32);
		}
	}

	// find components, the hope is the number of components should be small
	let mut components = HashSet::new();
	for v in 0..a.vert_pos.len() {
		components.insert(u_a.find(v as u32));
	}

	let verts: Vec<_> = components.into_iter().collect();

	let mut w03 = vec![0; a.num_vert()];
	let k02 = Kernel02 {
		vert_pos_p: &a.vert_pos,
		halfedge_q: &b.halfedge,
		vert_pos_q: &b.vert_pos,
		sym_pert_map,
		forward,
	};
	let mut recorderf = |query_idx: i32, leaf_idx: i32| {
		let (s02, z02) = k02.call(verts[query_idx as usize] as usize, leaf_idx as usize);
		if z02.is_finite() {
			w03[verts[query_idx as usize] as usize] += s02 * (if !forward { -1 } else { 1 });
		}
	};
	let mut recorder = SimpleRecorder::new(&mut recorderf);
	let f = |i| a.vert_pos[verts[i as usize] as usize];
	b.collider
		.collisions_from_fn::<false, _, _, SimpleRecorder<'_>>(f, verts.len(), &mut recorder);
	// flood fill
	for i in 0..w03.len() {
		let root = u_a.find(i as u32) as usize;
		if root == i {
			continue;
		}
		w03[i] = w03[root];
	}

	w03
}

#[derive(Debug)]
pub struct Boolean3<'a> {
	pub in_p: &'a MeshBoolImpl,
	pub in_q: &'a MeshBoolImpl,
	///Pairs of indices [edge_from_P, face_from_Q] that intersect
	pub p1q2: Vec<[i32; 2]>,
	///Pairs of indices [face_from_P, edge_from_Q] that intersect
	pub p2q1: Vec<[i32; 2]>,
	///Intersection status values for each P-edge crossing Q-face (corresponds to X12(eA, fB) in the dissertation).
	///Values are -1, 0, or +1 indicating crossing direction
	pub x12: Vec<i32>,
	///Intersection status values for each Q-edge crossing P-face (corresponds to X21(fA, eB) in the dissertation)
	///Values are -1, 0, or +1 indicating crossing direction
	pub x21: Vec<i32>,
	///Winding number for each vertex in P relative to solid Q (corresponds to X03(vA, B) in the dissertation).
	///Indicates if vertices of P are inside (1) or outside (0) of Q
	pub w03: Vec<i32>,
	///Winding number for each vertex in Q relative to solid P (corresponds to X30(A, vB) in the dissertation).
	///Indicates if vertices of Q are inside (1) or outside (0) of P
	pub w30: Vec<i32>,
	///The 3D intersection points where P-edges pierce Q-faces
	pub v12: Vec<Point3<f64>>,
	///The 3D intersection points where Q-faces are pierced by P-edges
	pub v21: Vec<Point3<f64>>,
	pub valid: bool,
}

impl<'a> Boolean3<'a> {
	pub fn new(in_p: &'a MeshBoolImpl, in_q: &'a MeshBoolImpl, op: OpType) -> Self {
		const INT_MAX_SZ: usize = i32::MAX as usize;

		if in_p.is_empty() || in_q.is_empty() || !in_p.bbox.does_overlap(&in_q.bbox) {
			//No overlap, early out
			return Boolean3 {
				in_p,
				in_q,
				p1q2: Vec::default(),
				p2q1: Vec::default(),
				x12: Vec::default(),
				x21: Vec::default(),
				w03: vec![0; in_p.num_vert()],
				w30: vec![0; in_q.num_vert()],
				v12: Vec::default(),
				v21: Vec::default(),
				valid: true,
			};
		}

		let sym_pert_map = in_p.get_perturbation_map(in_q, op);

		// Level 3
		// Build up the intersection of the edges and triangles, keeping only those
		// that intersect, and record the direction the edge is passing through the
		// triangle.
		let (p1q2, x12, v12) = intersect12(in_p, in_q, &sym_pert_map, true);
		let (p2q1, x21, v21) = intersect12(in_p, in_q, &sym_pert_map, false);

		if x12.len() > INT_MAX_SZ || x21.len() > INT_MAX_SZ {
			return Boolean3 {
				in_p,
				in_q,
				p1q2: Vec::default(),
				p2q1: Vec::default(),
				x12: Vec::default(),
				x21: Vec::default(),
				w03: Vec::default(),
				w30: Vec::default(),
				v12: Vec::default(),
				v21: Vec::default(),
				valid: false,
			};
		}

		// Compute winding numbers of all vertices using flood fill
		// Vertices on the same connected component have the same winding number
		let w03 = winding03(in_p, in_q, &p1q2, &sym_pert_map, true);
		let w30 = winding03(in_p, in_q, &p2q1, &sym_pert_map, false);

		Boolean3 {
			in_p,
			in_q,
			p1q2,
			p2q1,
			x12,
			x21,
			w03,
			w30,
			v12,
			v21,
			valid: true,
		}
	}
}
