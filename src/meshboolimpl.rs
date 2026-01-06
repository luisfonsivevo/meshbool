use crate::collider::Collider;
use crate::common::{AABB, sun_acos};
use crate::disjoint_sets::DisjointSets;
use crate::mesh_fixes::{FlipTris, transform_normal};
use crate::parallel::exclusive_scan_in_place;
use crate::shared::{Halfedge, TriRef, max_epsilon, next_halfedge, normal_transform};
use crate::utils::{atomic_add_i32, mat3, mat4, next3_i32, next3_usize};
use crate::vec::{vec_resize, vec_resize_nofill, vec_uninit};
use crate::{ManifoldError, MeshGL};
use nalgebra::{Matrix3x4, Point3, Vector3, Vector4};
use std::cmp::Ordering as CmpOrdering;
use std::collections::{BTreeMap, HashMap};
use std::f64;
use std::mem;
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering as AtomicOrdering};

#[derive(Clone)]
pub struct BaryIndices {
	pub tri: i32,
	pub start4: i32,
	pub end4: i32,
}

impl BaryIndices {
	pub fn new(tri: i32, start4: i32, end4: i32) -> Self {
		Self { tri, start4, end4 }
	}
}

#[derive(Copy, Clone)]
#[allow(unused)]
pub enum Shape {
	Tetrahedron,
	Cube,
	Octahedron,
}

pub static MESH_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

///@brief This library's internal representation of an oriented, 2-manifold,
///triangle mesh - a simple boundary-representation of a solid object. Use this
///class to store and operate on solids, and use MeshGL for input and output.
///
///In addition to storing geometric data, a Manifold can also store an arbitrary
///number of vertex properties. These could be anything, e.g. normals, UV
///coordinates, colors, etc, but this library is completely agnostic. All
///properties are merely float values indexed by channel number. It is up to the
///user to associate channel numbers with meaning.
///
///Manifold allows vertex properties to be shared for efficient storage, or to
///have multiple property verts associated with a single geometric vertex,
///allowing sudden property changes, e.g. at Boolean intersections, without
///sacrificing manifoldness.
///
///Manifolds also keep track of their relationships to their inputs, via
///OriginalIDs and the faceIDs and transforms accessible through MeshGL. This
///allows object-level properties to be re-associated with the output after many
///operations, particularly useful for materials. Since separate object's
///properties are not mixed, there is no requirement that channels have
///consistent meaning between different inputs.
#[derive(Clone, Debug)]
pub struct MeshBoolImpl {
	pub bbox: AABB,
	pub epsilon: f64,
	pub tolerance: f64,
	pub num_prop: i32,
	pub status: ManifoldError,
	pub vert_pos: Vec<Point3<f64>>,
	pub halfedge: Vec<Halfedge>,
	pub properties: Vec<f64>,
	// Note that vertNormal_ is not precise due to the use of an approximated acos
	// function
	pub vert_normal: Vec<Vector3<f64>>,
	pub face_normal: Vec<Vector3<f64>>,
	pub mesh_relation: MeshRelationD,
	pub collider: Collider,
}

#[derive(Clone, Debug)]
pub struct MeshRelationD {
	/// The originalID of this Manifold if it is an original; -1 otherwise.
	pub original_id: i32,
	pub mesh_id_transform: BTreeMap<i32, Relation>,
	pub tri_ref: Vec<TriRef>,
}

impl Default for MeshRelationD {
	fn default() -> Self {
		MeshRelationD {
			original_id: -1,
			mesh_id_transform: BTreeMap::default(),
			tri_ref: Vec::default(),
		}
	}
}

#[derive(Clone, Copy, Debug)]
pub struct Relation {
	pub original_id: i32,
	pub transform: Matrix3x4<f64>,
	pub back_side: bool,
}

impl Default for Relation {
	fn default() -> Self {
		Self {
			original_id: -1,
			transform: Matrix3x4::identity(),
			back_side: false,
		}
	}
}

impl MeshBoolImpl {
	pub fn is_empty(&self) -> bool {
		self.num_tri() == 0
	}

	pub fn num_vert(&self) -> usize {
		self.vert_pos.len()
	}

	pub fn num_edge(&self) -> usize {
		self.halfedge.len() / 2
	}

	pub fn num_tri(&self) -> usize {
		self.halfedge.len() / 3
	}

	pub fn num_prop(&self) -> usize {
		self.num_prop as usize
	}

	pub fn num_prop_vert(&self) -> usize {
		if self.num_prop() == 0 {
			self.num_vert()
		} else {
			self.properties.len() / self.num_prop()
		}
	}
}

const K_REMOVED_HALFEDGE: i32 = -2;

#[derive(Clone, Default)]
struct HalfedgePairData {
	large_vert: i32,
	tri: i32,
	edge_index: i32,
}

struct PrepHalfedges<'a, const USE_PROP: bool, F: FnMut(i32, i32, i32)> {
	halfedges: &'a mut Vec<Halfedge>,
	tri_prop: &'a Vec<Vector3<i32>>,
	tri_vert: &'a Vec<Vector3<i32>>,
	f: &'a mut F,
}

impl<'a, const USE_PROP: bool, F: FnMut(i32, i32, i32)> PrepHalfedges<'a, USE_PROP, F> {
	fn call(&mut self, tri: i32) {
		let props = self.tri_prop[tri as usize];
		for i in 0..3 {
			let j = next3_i32(i);
			let e = 3 * tri + i;
			let v0 = if USE_PROP {
				props[i as usize]
			} else {
				self.tri_vert[tri as usize][i as usize]
			};
			let v1 = if USE_PROP {
				props[j as usize]
			} else {
				self.tri_vert[tri as usize][j as usize]
			};
			debug_assert!(v0 != v1, "topological degeneracy");
			self.halfedges[e as usize] = Halfedge {
				start_vert: v0,
				end_vert: v1,
				paired_halfedge: -1,
				prop_vert: props[i as usize],
			};

			(self.f)(e, v0, v1);
		}
	}
}

impl MeshBoolImpl {
	pub fn from_meshgl(mesh_gl: &MeshGL) -> Self {
		let num_vert = mesh_gl.num_vert();
		let num_tri = mesh_gl.num_tri();

		let mut manifold = Self::default();

		if num_vert == 0 && num_tri == 0 {
			manifold.make_empty(ManifoldError::NoError);
			return manifold;
		}

		if num_vert < 4 || num_tri < 4 {
			manifold.make_empty(ManifoldError::NotManifold);
			return manifold;
		}

		if mesh_gl.num_prop < 3 {
			manifold.make_empty(ManifoldError::MissingPositionProperties);
			return manifold;
		}

		if mesh_gl.merge_from_vert.len() != mesh_gl.merge_to_vert.len() {
			manifold.make_empty(ManifoldError::MergeVectorsDifferentLengths);
			return manifold;
		}

		if !mesh_gl.run_transform.is_empty()
			&& 12 * mesh_gl.run_original_id.len() != mesh_gl.run_transform.len()
		{
			manifold.make_empty(ManifoldError::TransformWrongLength);
			return manifold;
		}

		if !mesh_gl.run_original_id.is_empty()
			&& !mesh_gl.run_index.is_empty()
			&& mesh_gl.run_original_id.len() + 1 != mesh_gl.run_index.len()
			&& mesh_gl.run_original_id.len() != mesh_gl.run_index.len()
		{
			manifold.make_empty(ManifoldError::RunIndexWrongLength);
			return manifold;
		}

		if !mesh_gl.face_id.is_empty() && mesh_gl.face_id.len() != mesh_gl.num_tri() {
			manifold.make_empty(ManifoldError::FaceIDWrongLength);
			return manifold;
		}

		if mesh_gl.vert_properties.iter().any(|v| !v.is_finite()) {
			manifold.make_empty(ManifoldError::NonFiniteVertex);
			return manifold;
		}

		if mesh_gl.run_transform.iter().any(|x| !x.is_finite()) {
			manifold.make_empty(ManifoldError::InvalidConstruction);
			return manifold;
		}

		// if (!manifold::all_of(meshGL.halfedgeTangent.begin(),
		//                       meshGL.halfedgeTangent.end(),
		//                       [](Precision x) { return std::isfinite(x); })) {
		//   MakeEmpty(Error::InvalidConstruction);
		//   return;
		// }

		let mut prop2vert: Vec<i32>;
		if !mesh_gl.merge_from_vert.is_empty() {
			prop2vert = (0..num_vert as i32).collect();
			for i in 0..mesh_gl.merge_from_vert.len() {
				let from = mesh_gl.merge_from_vert[i];
				let to = mesh_gl.merge_to_vert[i];
				if from as usize >= num_vert || to as usize >= num_vert {
					manifold.make_empty(ManifoldError::MergeIndexOutOfBounds);
					return manifold;
				}
				prop2vert[from as usize] = to as i32;
			}
		} else {
			prop2vert = vec![];
		}

		let num_prop = mesh_gl.num_prop - 3;
		manifold.num_prop = num_prop as i32;
		unsafe {
			vec_resize_nofill(
				&mut manifold.properties,
				mesh_gl.num_vert() * num_prop as usize,
			)
		};
		manifold.tolerance = mesh_gl.tolerance.into();
		// This will have unreferenced duplicate positions that will be removed by
		// Impl::remove_unreferenced_verts().
		unsafe { vec_resize_nofill(&mut manifold.vert_pos, mesh_gl.num_vert()) };

		for i in 0..mesh_gl.num_vert() {
			for j in [0, 1, 2] {
				manifold.vert_pos[i][j] =
					mesh_gl.vert_properties[mesh_gl.num_prop as usize * i + j].into();
			}
			for j in 0..num_prop {
				manifold.properties[i * num_prop as usize + j as usize] =
					mesh_gl.vert_properties[mesh_gl.num_prop as usize * i + 3 + j as usize].into();
			}
		}

		// halfedgeTangent_.resize_nofill(meshGL.halfedgeTangent.len() / 4);
		// for i in 0..halfedgeTangent_.len() {
		//   for j in [0, 1, 2, 3] {
		//     halfedgeTangent_[i][j] = meshGL.halfedgeTangent[4 * i + j];
		//   }
		// }

		let mut tri_ref: Vec<TriRef> = unsafe { vec_uninit(mesh_gl.num_tri()) };

		let mut run_index = mesh_gl.run_index.clone();
		let run_end = mesh_gl.tri_verts.len();
		if run_index.is_empty() {
			run_index = vec![0, run_end as u32];
		} else if run_index.len() == mesh_gl.run_original_id.len() {
			run_index.push(run_end as u32);
		} else if run_index.len() == 1 {
			run_index.push(run_end as u32);
		}

		let start_id = MeshBoolImpl::reserve_ids(1.max(mesh_gl.run_original_id.len()));
		let mut run_original_id = mesh_gl.run_original_id.clone();
		if run_original_id.is_empty() {
			run_original_id.push(start_id as u32);
		}
		for i in 0..run_original_id.len() {
			let mesh_id = start_id + i;
			let original_id = run_original_id[i];
			for tri in (run_index[i] / 3)..(run_index[i + 1] / 3) {
				let r = &mut tri_ref[tri as usize];
				r.mesh_id = mesh_id as i32;
				r.original_id = original_id as i32;
				r.face_id = if mesh_gl.face_id.is_empty() {
					-1
				} else {
					mesh_gl.face_id[tri as usize] as i32
				};
				r.coplanar_id = tri as i32;
			}

			if mesh_gl.run_transform.is_empty() {
				manifold.mesh_relation.mesh_id_transform.insert(
					mesh_id as i32,
					Relation {
						original_id: original_id as i32,
						..Relation::default() //in c++ the rest were uninitialized
					},
				);
			} else {
				let m = &mesh_gl.run_transform[12 * i..];
				manifold.mesh_relation.mesh_id_transform.insert(
					mesh_id as i32,
					Relation {
						original_id: original_id as i32,
						transform: [
							[m[0] as f64, m[1] as f64, m[2] as f64],
							[m[3] as f64, m[4] as f64, m[5] as f64],
							[m[6] as f64, m[7] as f64, m[8] as f64],
							[m[9] as f64, m[10] as f64, m[11] as f64],
						]
						.into(),
						..Relation::default()
					},
				);
			}
		}

		let mut tri_prop: Vec<Vector3<i32>> = Vec::with_capacity(num_tri);
		let mut tri_vert: Vec<Vector3<i32>> = vec![];
		let needs_prop_map = num_prop > 0 && !prop2vert.is_empty();
		if needs_prop_map {
			tri_vert.reserve(num_tri)
		}
		if tri_ref.len() > 0 {
			manifold.mesh_relation.tri_ref.reserve(num_tri);
		}
		for i in 0..num_tri {
			let mut tri_p: Vector3<i32> = Vector3::default();
			let mut tri_v: Vector3<i32> = Vector3::default();
			for j in [0, 1, 2] {
				let vert = mesh_gl.tri_verts[3 * i + j];
				if vert as usize >= num_vert {
					manifold.make_empty(ManifoldError::VertexOutOfBounds);
					return manifold;
				}
				tri_p[j] = vert as i32;
				tri_v[j] = if prop2vert.is_empty() {
					vert as i32
				} else {
					prop2vert[vert as usize]
				};
			}
			if tri_v[0] != tri_v[1] && tri_v[1] != tri_v[2] && tri_v[2] != tri_v[0] {
				if needs_prop_map {
					tri_prop.push(tri_p);
					tri_vert.push(tri_v);
				} else {
					tri_prop.push(tri_v);
				}
				if tri_ref.len() > 0 {
					manifold.mesh_relation.tri_ref.push(tri_ref[i]);
				}
			}
		}

		manifold.create_halfedges(tri_prop, tri_vert);
		if !manifold.is_manifold() {
			manifold.make_empty(ManifoldError::NotManifold);
			return manifold;
		}

		manifold.calculate_bbox();
		manifold.set_epsilon(-1.0f64, false); // TODO: if Precision == float

		// we need to split pinched verts before calculating vertex normals, because
		// the algorithm doesn't work with pinched verts
		manifold.cleanup_topology();
		manifold.calculate_normals();

		manifold.dedupe_prop_verts();
		manifold.mark_coplanar();

		manifold.remove_degenerates(None);
		manifold.remove_unreferenced_verts();
		manifold.finish();

		if !manifold.is_finite() {
			manifold.make_empty(ManifoldError::NonFiniteVertex);
			return manifold;
		}

		// A Manifold created from an input mesh is never an original - the input is
		// the original.
		manifold.mesh_relation.original_id = -1;

		manifold
	}

	pub fn from_shape(shape: Shape, m: Matrix3x4<f64>) -> Self {
		let (mut vert_pos, tri_verts) = match shape {
			Shape::Tetrahedron => (
				vec![
					Point3::<f64>::new(-1.0, -1.0, 1.0),
					Point3::<f64>::new(-1.0, 1.0, -1.0),
					Point3::<f64>::new(1.0, -1.0, -1.0),
					Point3::<f64>::new(1.0, 1.0, 1.0),
				],
				vec![
					Vector3::<i32>::new(2, 0, 1),
					Vector3::<i32>::new(0, 3, 1),
					Vector3::<i32>::new(2, 3, 0),
					Vector3::<i32>::new(3, 2, 1),
				],
			),
			Shape::Cube => (
				vec![
					Point3::<f64>::new(0.0, 0.0, 0.0),
					Point3::<f64>::new(0.0, 0.0, 1.0),
					Point3::<f64>::new(0.0, 1.0, 0.0),
					Point3::<f64>::new(0.0, 1.0, 1.0),
					Point3::<f64>::new(1.0, 0.0, 0.0),
					Point3::<f64>::new(1.0, 0.0, 1.0),
					Point3::<f64>::new(1.0, 1.0, 0.0),
					Point3::<f64>::new(1.0, 1.0, 1.0),
				],
				vec![
					Vector3::<i32>::new(1, 0, 4),
					Vector3::<i32>::new(2, 4, 0),
					Vector3::<i32>::new(1, 3, 0),
					Vector3::<i32>::new(3, 1, 5),
					Vector3::<i32>::new(3, 2, 0),
					Vector3::<i32>::new(3, 7, 2),
					Vector3::<i32>::new(5, 4, 6),
					Vector3::<i32>::new(5, 1, 4),
					Vector3::<i32>::new(6, 4, 2),
					Vector3::<i32>::new(7, 6, 2),
					Vector3::<i32>::new(7, 3, 5),
					Vector3::<i32>::new(7, 5, 6),
				],
			),
			Shape::Octahedron => (
				vec![
					Point3::<f64>::new(1.0, 0.0, 0.0),
					Point3::<f64>::new(-1.0, 0.0, 0.0),
					Point3::<f64>::new(0.0, 1.0, 0.0),
					Point3::<f64>::new(0.0, -1.0, 0.0),
					Point3::<f64>::new(0.0, 0.0, 1.0),
					Point3::<f64>::new(0.0, 0.0, -1.0),
				],
				vec![
					Vector3::<i32>::new(0, 2, 4),
					Vector3::<i32>::new(1, 5, 3),
					Vector3::<i32>::new(2, 1, 4),
					Vector3::<i32>::new(3, 5, 0),
					Vector3::<i32>::new(1, 3, 4),
					Vector3::<i32>::new(0, 5, 2),
					Vector3::<i32>::new(3, 0, 4),
					Vector3::<i32>::new(2, 5, 1),
				],
			),
		};

		for v in &mut vert_pos {
			v.coords = m * v.coords.push(1.0);
		}

		let mut meshbool_impl = Self {
			vert_pos,
			..MeshBoolImpl::default()
		};

		meshbool_impl.create_halfedges(tri_verts, Vec::new());
		meshbool_impl.finish();
		meshbool_impl.initialize_original(false);
		meshbool_impl.mark_coplanar();

		meshbool_impl
	}

	pub fn remove_unreferenced_verts(&mut self) {
		let num_vert = self.num_vert();
		let keep = vec![0; num_vert];
		for h in &self.halfedge {
			if h.start_vert >= 0 {
				let atomic_ref: &AtomicI32 =
					unsafe { std::mem::transmute(&keep[h.start_vert as usize]) };
				atomic_ref.store(1, AtomicOrdering::Relaxed);
			}
		}

		for v in 0..num_vert {
			if keep[v] == 0 {
				self.vert_pos[v] = Point3::new(f64::NAN, f64::NAN, f64::NAN);
			}
		}
	}

	pub fn reserve_ids(n: usize) -> usize {
		MESH_ID_COUNTER.fetch_add(n, AtomicOrdering::Relaxed)
	}

	pub fn initialize_original(&mut self, keep_face_id: bool) {
		let mesh_id = MeshBoolImpl::reserve_ids(1) as i32;
		self.mesh_relation.original_id = mesh_id;
		let num_tri = self.num_tri();
		let tri_ref = &mut self.mesh_relation.tri_ref;
		unsafe {
			vec_resize_nofill(tri_ref, num_tri);
		}
		for tri in 0..num_tri {
			tri_ref[tri] = TriRef {
				mesh_id,
				original_id: mesh_id,
				face_id: -1,
				coplanar_id: if keep_face_id {
					tri_ref[tri].coplanar_id
				} else {
					tri as i32
				},
			};

			self.mesh_relation.mesh_id_transform.clear();
			self.mesh_relation
				.mesh_id_transform
				.entry(mesh_id)
				.or_insert_with(|| Relation {
					original_id: mesh_id,
					..Relation::default() //in c++ the rest were uninitialized
				});
		}
	}

	pub fn mark_coplanar(&mut self) {
		let num_tri = self.num_tri();
		struct TriPriority {
			area2: f64,
			tri: i32,
		}
		let mut tri_priority = unsafe { vec_uninit(num_tri) };
		for tri in 0..num_tri {
			self.mesh_relation.tri_ref[tri].coplanar_id = -1;
			if self.halfedge[3 * tri].start_vert < 0 {
				tri_priority[tri] = TriPriority {
					area2: 0.0,
					tri: tri as i32,
				};
				continue;
			}

			let v = self.vert_pos[self.halfedge[3 * tri].start_vert as usize];
			tri_priority[tri] = TriPriority {
				area2: (self.vert_pos[self.halfedge[3 * tri].end_vert as usize] - v)
					.cross(&(self.vert_pos[self.halfedge[3 * tri + 1].end_vert as usize] - v))
					.magnitude_squared(),
				tri: tri as i32,
			};
		}

		tri_priority.sort_by(|a, b| b.area2.partial_cmp(&a.area2).unwrap_or(CmpOrdering::Equal));

		let mut interior_halfedges: Vec<i32> = Vec::default();
		for tp in &tri_priority {
			if self.mesh_relation.tri_ref[tp.tri as usize].coplanar_id >= 0 {
				continue;
			}

			self.mesh_relation.tri_ref[tp.tri as usize].coplanar_id = tp.tri;
			if self.halfedge[(3 * tp.tri) as usize].start_vert < 0 {
				continue;
			}
			let base = self.vert_pos[self.halfedge[(3 * tp.tri) as usize].start_vert as usize];
			let normal = self.face_normal[tp.tri as usize];
			vec_resize(&mut interior_halfedges, 3);
			interior_halfedges[0] = 3 * tp.tri;
			interior_halfedges[1] = 3 * tp.tri + 1;
			interior_halfedges[2] = 3 * tp.tri + 2;
			while !interior_halfedges.is_empty() {
				let h = next_halfedge(
					self.halfedge[*interior_halfedges.last().unwrap() as usize].paired_halfedge,
				);
				interior_halfedges.pop().unwrap();
				if self.mesh_relation.tri_ref[(h / 3) as usize].coplanar_id >= 0 {
					continue;
				}

				let v = self.vert_pos[self.halfedge[h as usize].end_vert as usize];
				if (v - base).dot(&normal).abs() < self.tolerance {
					self.mesh_relation.tri_ref[(h / 3) as usize].coplanar_id = tp.tri;

					if interior_halfedges.is_empty()
						|| h != self.halfedge[*interior_halfedges.last().unwrap() as usize]
							.paired_halfedge
					{
						interior_halfedges.push(h);
					} else {
						interior_halfedges.pop().unwrap();
					}

					let h_next = next_halfedge(h);
					interior_halfedges.push(h_next);
				}
			}
		}
	}

	///Create the halfedge_ data structure from a list of triangles. If the optional
	///prop2vert array is missing, it's assumed these triangles are are pointing to
	///both vert and propVert indices. If prop2vert is present, the triangles are
	///assumed to be pointing to propVert indices only. The prop2vert array is used
	///to map the propVert indices to vert indices.
	pub fn create_halfedges(&mut self, tri_prop: Vec<Vector3<i32>>, tri_vert: Vec<Vector3<i32>>) {
		let num_tri = tri_prop.len();
		let num_halfedge: i32 = (3 * num_tri) as i32;
		// drop the old value first to avoid copy
		self.halfedge.clear();
		unsafe {
			vec_resize_nofill(&mut self.halfedge, num_halfedge as usize);
		}

		let vert_count = self.vert_pos.len() as i32;

		//PrepHalfedges start
		let mut ids = {
			let ids = if vert_count < (1 << 18) {
				// For small vertex count, it is faster to just do sorting
				let mut edge: Vec<u64> = unsafe { vec_uninit(num_halfedge as usize) };
				let mut set_edge = |e: i32, v0: i32, v1: i32| {
					edge[e as usize] = (if v0 < v1 { 1 } else { 0 }) << 63
						| (v0.min(v1) as u64) << 32
						| (v0.max(v1) as u64);
				};

				if tri_vert.is_empty() {
					let mut job = PrepHalfedges::<true, _> {
						halfedges: &mut self.halfedge,
						tri_prop: &tri_prop,
						tri_vert: &tri_vert,
						f: &mut set_edge,
					};

					for i in 0..num_tri {
						let i = i as i32;
						job.call(i);
					}
				} else {
					let mut job = PrepHalfedges::<false, _> {
						halfedges: &mut self.halfedge,
						tri_prop: &tri_prop,
						tri_vert: &tri_vert,
						f: &mut set_edge,
					};

					for i in 0..num_tri {
						let i = i as i32;
						job.call(i);
					}
				}

				let mut ids: Vec<i32> = (0..num_halfedge).collect();
				ids.sort_by_key(|&i| edge[i as usize]);
				ids
			} else {
				// For larger vertex count, we separate the ids into slices for halfedges
				// with the same smaller vertex.
				// We first copy them there (as HalfedgePairData), and then do sorting
				// locally for each slice.
				// This helps with memory locality, and is faster for larger meshes.
				let mut entries = unsafe { vec_uninit(num_halfedge as usize) };
				let mut offsets: Vec<i32> = vec![0; (vert_count * 2) as usize];
				let mut set_offset = |_e: i32, v0: i32, v1: i32| {
					let offset = if v0 > v1 { 0 } else { vert_count };
					unsafe {
						atomic_add_i32(&mut offsets[(v0.min(v1) + offset) as usize], 1);
					}
				};

				if tri_vert.is_empty() {
					let mut job = PrepHalfedges::<true, _> {
						halfedges: &mut self.halfedge,
						tri_prop: &tri_prop,
						tri_vert: &tri_vert,
						f: &mut set_offset,
					};

					for i in 0..num_tri {
						let i = i as i32;
						job.call(i);
					}
				} else {
					let mut job = PrepHalfedges::<false, _> {
						halfedges: &mut self.halfedge,
						tri_prop: &tri_prop,
						tri_vert: &tri_vert,
						f: &mut set_offset,
					};

					for i in 0..num_tri {
						let i = i as i32;
						job.call(i);
					}
				}

				exclusive_scan_in_place(&mut offsets, 0);

				for tri in 0..num_tri {
					let tri = tri as i32;
					for i in 0..3 {
						let e = 3 * tri + i;
						let e_usize = e as usize;
						let v0 = self.halfedge[e_usize].start_vert;
						let v1 = self.halfedge[e_usize].end_vert;
						let offset = if v0 > v1 { 0 } else { vert_count as i32 };
						let start = v0.min(v1);
						let index =
							unsafe { atomic_add_i32(&mut offsets[(start + offset) as usize], 1) };
						entries[index as usize] = HalfedgePairData {
							large_vert: v0.max(v1),
							tri,
							edge_index: e,
						};
					}
				}

				let mut ids: Vec<i32> = unsafe { vec_uninit(num_halfedge as usize) };
				for v in 0..offsets.len() {
					let start = if v == 0 { 0 } else { offsets[v - 1] };
					let end = offsets[v];
					for i in start..end {
						ids[i as usize] = i;
					}

					ids[start as usize..end as usize].sort_unstable_by_key(|&i| {
						let entry = &entries[i as usize];
						(entry.large_vert, entry.tri)
					});

					for i in start..end {
						let i = i as usize;
						ids[i] = entries[ids[i] as usize].edge_index;
					}
				}

				ids
			};

			ids
		};

		//PrepHalfedges end

		// Mark opposed triangles for removal - this may strand unreferenced verts
		// which are removed later by self.remove_unreferenced_verts() and self.finish().
		let num_edge = num_halfedge / 2;
		let mut consecutive_start = 0;
		for i in 0..num_edge {
			let pair0 = ids[i as usize];
			let h0 = self.halfedge[pair0 as usize];
			let mut k = num_edge + consecutive_start;
			loop {
				let pair1 = ids[k as usize];
				let h1 = self.halfedge[pair1 as usize];
				if h0.start_vert != h1.end_vert || h0.end_vert != h1.start_vert {
					break;
				}
				if h1.paired_halfedge != K_REMOVED_HALFEDGE
					&& self.halfedge[next_halfedge(pair0) as usize].end_vert
						== self.halfedge[next_halfedge(pair1) as usize].end_vert
				{
					self.halfedge[pair0 as usize].paired_halfedge = K_REMOVED_HALFEDGE;
					self.halfedge[pair1 as usize].paired_halfedge = K_REMOVED_HALFEDGE;
					// Reorder so that remaining edges pair up
					if k != i + num_edge {
						ids.swap((i + num_edge) as usize, k as usize);
					}

					break;
				}

				k += 1;
				if k >= num_edge * 2 {
					break;
				}
			}

			if i + 1 == num_edge {
				continue;
			}
			let h1 = self.halfedge[ids[(i + 1) as usize] as usize];
			if h1.start_vert == h0.start_vert && h1.end_vert == h0.end_vert {
				continue;
			}

			consecutive_start = i + 1;
		}

		for i in 0..num_edge as usize {
			let i = i as i32;
			let pair0 = ids[i as usize];
			let pair1 = ids[(i + num_edge) as usize];
			let pair0_usize = pair0 as usize;
			let pair1_usize = pair1 as usize;

			if self.halfedge[pair0_usize].paired_halfedge != K_REMOVED_HALFEDGE {
				self.halfedge[pair0_usize].paired_halfedge = pair1;
				self.halfedge[pair1_usize].paired_halfedge = pair0;
			} else {
				let new_halfedge = Halfedge {
					start_vert: -1,
					end_vert: -1,
					paired_halfedge: -1,
					prop_vert: 0,
				};

				self.halfedge[pair0_usize] = new_halfedge;
				self.halfedge[pair1_usize] = new_halfedge;
			}
		}
	}

	///Does a full recalculation of the face bounding boxes, including updating
	///the collider, but does not resort the faces.
	fn update(&mut self) {
		self.calculate_bbox();
		let mut face_box = Vec::new();
		let mut face_morton = Vec::new();
		self.get_face_box_morton(&mut face_box, &mut face_morton);
		self.collider.update_boxes(&face_box);
	}

	pub fn make_empty(&mut self, status: ManifoldError) {
		self.bbox = AABB::default();
		self.vert_pos = Vec::default();
		self.halfedge = Vec::default();
		self.vert_normal = Vec::default();
		self.face_normal = Vec::default();
		self.mesh_relation = MeshRelationD::default();
		self.status = status;
	}

	pub fn warp(&mut self, warp_func: impl Fn(&mut Point3<f64>)) {
		self.warp_batch(|vecs| {
			vecs.iter_mut().for_each(|v| warp_func(v));
		});
	}

	pub fn warp_batch(&mut self, warp_func: impl Fn(&mut [Point3<f64>])) {
		warp_func(&mut self.vert_pos);
		self.calculate_bbox();
		if !self.is_finite() {
			self.make_empty(ManifoldError::NonFiniteVertex);
			return;
		}
		self.update();
		self.face_normal.clear(); // force recalculation of triNormal
		self.set_epsilon(-1.0, false);
		self.finish();
		self.mark_coplanar();
		self.mesh_relation.original_id = -1;
	}

	///Transform this Manifold in space. The first three columns form a 3x3 matrix
	///transform and the last is a translation vector. This operation can be
	///chained. Transforms are combined and applied lazily.
	///
	///@param m The affine transform matrix to apply to all the vertices.
	pub fn transform(&self, transform: &Matrix3x4<f64>) -> MeshBoolImpl {
		if *transform == Matrix3x4::identity() {
			return self.clone();
		}
		let mut result = MeshBoolImpl::default();
		if self.status != ManifoldError::NoError {
			result.status = self.status;
			return result;
		}
		if !transform.iter().fold(true, |acc, e| acc && e.is_finite()) {
			result.make_empty(ManifoldError::NonFiniteVertex);
			return result;
		}

		result.collider = self.collider.clone();
		result.mesh_relation = self.mesh_relation.clone();
		result.epsilon = self.epsilon;
		result.tolerance = self.tolerance;
		result.num_prop = self.num_prop;
		result.properties = self.properties.clone();
		result.bbox = self.bbox;
		result.halfedge = self.halfedge.clone();

		result.mesh_relation.original_id = -1;
		for m in &mut result.mesh_relation.mesh_id_transform {
			m.1.transform = transform * mat4(&m.1.transform);
		}

		vec_resize(&mut result.vert_pos, self.num_vert());
		vec_resize(&mut result.face_normal, self.face_normal.len());
		vec_resize(&mut result.vert_normal, self.vert_normal.len());
		for i in 0..self.vert_pos.len() {
			let v = &self.vert_pos[i];
			result.vert_pos[i] = (transform * Vector4::new(v.x, v.y, v.z, 1.0)).into();
		}

		let normal_transform = normal_transform(transform);
		for i in 0..self.face_normal.len() {
			result.face_normal[i] = transform_normal(normal_transform, self.face_normal[i]);
		}
		for i in 0..self.vert_normal.len() {
			result.vert_normal[i] = transform_normal(normal_transform, self.vert_normal[i]);
		}

		let invert = mat3(transform).determinant() < 0.0;
		if invert {
			for tri in 0..result.num_tri() {
				FlipTris {
					halfedge: &mut result.halfedge,
				}
				.call(tri);
			}
		}

		// This optimization does a cheap collider update if the transform is
		// axis-aligned.
		if !result.collider.transform(*transform) {
			result.update();
		}

		result.calculate_bbox();
		result.epsilon *= mat3(transform).svd(false, false).singular_values[0];
		result.set_epsilon(result.epsilon, false);
		result
	}

	pub fn set_epsilon(&mut self, min_epsilon: f64, use_single: bool) {
		self.epsilon = max_epsilon(min_epsilon, &self.bbox);
		let mut min_tol = self.epsilon;
		if use_single {
			min_tol = min_tol.max(f32::EPSILON as f64 * self.bbox.scale());
		}

		self.tolerance = self.tolerance.max(min_tol);
	}

	///If face normals are already present, this function uses them to compute
	///vertex normals (angle-weighted pseudo-normals); otherwise it also computes
	///the face normals. Face normals are only calculated when needed because
	///nearly degenerate faces will accrue rounding error, while the Boolean can
	///retain their original normal, which is more accurate and can help with
	///merging coplanar faces.
	///
	///If the face normals have been invalidated by an operation like Warp(),
	///ensure you do faceNormal_.resize(0) before calling this function to force
	///recalculation.
	pub fn calculate_normals(&mut self) {
		let num_vert = self.num_vert();
		vec_resize(&mut self.vert_normal, num_vert);

		let vert_halfedge_map: Vec<AtomicI32> = (0..self.num_vert())
			.map(|_| AtomicI32::new(i32::MAX))
			.collect();

		let atomic_min = |value, vert: i32| {
			if vert < 0 {
				return;
			}
			let mut old = i32::MAX;
			while let Err(actual) = vert_halfedge_map[vert as usize].compare_exchange(
				old,
				value,
				AtomicOrdering::SeqCst,
				AtomicOrdering::SeqCst,
			) {
				old = actual;
				if old < value {
					break;
				}
			}
		};

		if self.face_normal.len() != self.num_tri() {
			let num_tri = self.num_tri();
			vec_resize(&mut self.face_normal, num_tri);
			for face in 0..num_tri {
				let face = face as i32;
				let tri_normal = &mut self.face_normal[face as usize];
				if self.halfedge[(3 * face) as usize].start_vert < 0 {
					*tri_normal = Vector3::new(0.0, 0.0, 1.0);
					continue;
				}

				let mut tri_verts = Vector3::<i32>::default();
				for i in 0..3 {
					let v = self.halfedge[(3 * face + i) as usize].start_vert;
					tri_verts[i as usize] = v;
					atomic_min(3 * face + i, v);
				}

				let mut edge = [Vector3::<f64>::default(); 3];
				for i in 0..3 {
					let j = next3_usize(i);
					edge[i] = (self.vert_pos[tri_verts[j] as usize]
						- self.vert_pos[tri_verts[i] as usize])
						.normalize();
				}

				*tri_normal = edge[0].cross(&edge[1]).normalize();
				if tri_normal.x.is_nan() {
					*tri_normal = Vector3::new(0.0, 0.0, 1.0);
				}
			}
		} else {
			for i in 0..self.halfedge.len() {
				let i = i as i32;
				atomic_min(i, self.halfedge[i as usize].start_vert);
			}
		}

		for vert in 0..self.num_vert() {
			let first_edge = vert_halfedge_map[vert].load(AtomicOrdering::SeqCst);
			// not referenced
			if first_edge == i32::MAX {
				self.vert_normal[vert] = Vector3::from_element(0.0);
				continue;
			}

			let mut normal = Vector3::from_element(0.0);
			self.for_vert(first_edge, |edge| {
				let tri_verts = Vector3::<i32>::new(
					self.halfedge[edge as usize].start_vert,
					self.halfedge[edge as usize].end_vert,
					self.halfedge[next_halfedge(edge) as usize].end_vert,
				);
				let curr_edge = (self.vert_pos[tri_verts[1] as usize]
					- self.vert_pos[tri_verts[0] as usize])
					.normalize();
				let prev_edge = (self.vert_pos[tri_verts[0] as usize]
					- self.vert_pos[tri_verts[2] as usize])
					.normalize();

				// if it is not finite, this means that the triangle is degenerate, and we
				// should just exclude it from the normal calculation...
				if !curr_edge[0].is_finite() || !prev_edge[0].is_finite() {
					return;
				}
				let dot = -prev_edge.dot(&curr_edge);
				let phi = if dot >= 1.0 {
					0.0
				} else if dot <= -1.0 {
					f64::consts::PI
				} else {
					sun_acos(dot)
				};
				normal += phi * self.face_normal[(edge / 3) as usize];
			});

			self.vert_normal[vert] = normal.normalize();
		}
	}

	///Remaps all the contained meshIDs to new unique values to represent new
	///instances of these meshes.
	pub fn increment_mesh_ids(&mut self) {
		//in c++ this uses a custom hashtable class
		let mut mesh_id_old2new =
			HashMap::with_capacity(self.mesh_relation.mesh_id_transform.len() * 2);
		let old_transforms = mem::take(&mut self.mesh_relation.mesh_id_transform);
		let num_mesh_ids = old_transforms.len();
		let mut next_mesh_id = MeshBoolImpl::reserve_ids(num_mesh_ids) as i32;
		for pair in old_transforms {
			mesh_id_old2new.insert(pair.0, next_mesh_id);
			self.mesh_relation
				.mesh_id_transform
				.entry(next_mesh_id)
				.or_insert(pair.1);
			next_mesh_id += 1;
		}

		let num_tri = self.num_tri();
		for i in 0..num_tri {
			let tri_ref = &mut self.mesh_relation.tri_ref[i];
			tri_ref.mesh_id = *mesh_id_old2new.get(&tri_ref.mesh_id).unwrap_or(&0)
		}
	}

	#[inline]
	pub fn for_vert(&self, halfedge: i32, mut func: impl FnMut(i32)) {
		let mut current = halfedge;
		loop {
			current = next_halfedge(self.halfedge[current as usize].paired_halfedge);
			func(current);
			if current == halfedge {
				break;
			}
		}
	}

	#[inline]
	pub fn for_vert_mut(&mut self, halfedge: i32, mut func: impl FnMut(&mut Self, i32)) {
		let mut current = halfedge;
		loop {
			current = next_halfedge(self.halfedge[current as usize].paired_halfedge);
			func(self, current);
			if current == halfedge {
				break;
			}
		}
	}

	#[inline]
	pub fn for_vert_fun<T>(
		&self,
		halfedge: i32,
		mut transform: impl FnMut(i32) -> T,
		mut binary_op: impl FnMut(i32, &T, &mut T),
	) {
		let mut here: T = transform(halfedge);
		let mut current: i32 = halfedge;
		loop {
			let next_halfedge: i32 = next_halfedge(self.halfedge[current as usize].paired_halfedge);
			let mut next: T = transform(next_halfedge);
			binary_op(current, &here, &mut next);
			here = next;
			current = next_halfedge;
			if current == halfedge {
				break;
			}
		}
	}

	///Dereference duplicate property vertices if they are exactly floating-point
	///equal. These unreferenced properties are then removed by CompactProps.
	pub fn dedupe_prop_verts(&mut self) {
		let num_prop = self.num_prop();
		if num_prop == 0 {
			return;
		}

		let mut vert2vert: Vec<(i32, i32)> = vec![(-1, -1); self.halfedge.len()];
		for edge_idx in 0..self.halfedge.len() {
			let edge = self.halfedge[edge_idx];
			if edge.paired_halfedge < 0 {
				continue;
			}
			let edge_face = edge_idx / 3;
			let pair_face = edge.paired_halfedge / 3;

			if self.mesh_relation.tri_ref[edge_face].mesh_id
				!= self.mesh_relation.tri_ref[pair_face as usize].mesh_id
			{
				continue;
			}

			let prop0 = self.halfedge[edge_idx].prop_vert;
			let prop1 = self.halfedge[next_halfedge(edge.paired_halfedge) as usize].prop_vert;
			let mut prop_equal = true;
			for p in 0..num_prop {
				if self.properties[num_prop * prop0 as usize + p]
					!= self.properties[num_prop * prop1 as usize + p]
				{
					prop_equal = false;
					break;
				}
			}
			if prop_equal {
				vert2vert[edge_idx] = (prop0, prop1);
			}
		}

		let mut vert_labels: Vec<i32> = vec![];
		let num_prop_vert = self.num_prop_vert();

		fn get_labels(
			components: &mut Vec<i32>,
			edges: &Vec<(i32, i32)>,
			num_nodes: usize,
		) -> usize {
			let uf = DisjointSets::new(num_nodes as u32);
			for edge in edges {
				if edge.0 == -1 || edge.1 == -1 {
					continue;
				}
				uf.unite(edge.0 as u32, edge.1 as u32);
			}

			return uf.connected_components(components);
		}

		let num_labels = get_labels(&mut vert_labels, &vert2vert, num_prop_vert);

		let mut label2vert: Vec<i32> = vec![0; num_labels];
		for v in 0..num_prop_vert {
			label2vert[vert_labels[v] as usize] = v as i32;
		}
		for edge in self.halfedge.iter_mut() {
			edge.prop_vert = label2vert[vert_labels[edge.prop_vert as usize] as usize];
		}
	}
}

impl Default for MeshBoolImpl {
	fn default() -> Self {
		Self {
			bbox: AABB::default(),
			epsilon: -1.0,
			tolerance: -1.0,
			num_prop: 0,
			status: ManifoldError::NoError,
			vert_pos: Vec::default(),
			halfedge: Vec::default(),
			properties: Vec::default(),
			vert_normal: Vec::default(),
			face_normal: Vec::default(),
			mesh_relation: MeshRelationD::default(),
			collider: Collider::default(),
		}
	}
}
