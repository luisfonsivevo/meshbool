use crate::meshboolimpl::MeshBoolImpl;
use crate::polygon::{PolyVert, PolygonsIdx, SimplePolygonIdx, triangulate_idx};
use crate::shared::{Halfedge, TriRef, get_axis_aligned_projection};
use crate::utils::ccw;
use crate::vec::InsertSorted;
use nalgebra::{Matrix2x3, Matrix3x2, Point3, Vector3};
use std::mem;
use std::ops::DerefMut;

///Returns an assembled set of vertex index loops of the input list of
///Halfedges, where each vert must be referenced the same number of times as a
///startVert and endVert. If startHalfedgeIdx is given, instead of putting
///vertex indices into the returned polygons structure, it will use the halfedge
///indices instead.
fn assemble_halfedges(edges: &[Halfedge], start_halfedge_idx: i32) -> Vec<Vec<i32>> {
	//originally a c++ multimap. tuple is (k, v). sort by key, and if keys are equal, sort by insertion order within the key
	let mut vert_edge = Vec::new();
	for (i, edge) in edges.iter().enumerate() {
		vert_edge.insert_sorted_by_key((edge.start_vert, i as i32), |&(vert, _)| vert);
	}

	let mut polys = Vec::new();
	let mut start_edge = 0;
	let mut this_edge = start_edge;
	loop {
		if this_edge == start_edge {
			if vert_edge.is_empty() {
				break;
			}
			start_edge = vert_edge[0].1;
			this_edge = start_edge;
			polys.push(Vec::new());
		}

		polys
			.last_mut()
			.unwrap()
			.push(start_halfedge_idx + this_edge);
		let result =
			vert_edge.binary_search_by_key(&edges[this_edge as usize].end_vert, |&(vert, _)| vert);
		let result = vert_edge.remove(result.expect("non-manifold edge"));
		this_edge = result.1;
	}

	polys
}

///Add the vertex position projection to the indexed polygons.
fn project_polygons(
	polys: &[Vec<i32>],
	halfedge: &[Halfedge],
	vert_pos: &[Point3<f64>],
	projection: Matrix2x3<f64>,
) -> PolygonsIdx {
	let mut polygons = PolygonsIdx::new();
	for poly in polys {
		let mut polygon = SimplePolygonIdx::new();
		for &edge in poly {
			polygon.push(PolyVert {
				pos: (projection * vert_pos[halfedge[edge as usize].start_vert as usize]),
				idx: edge,
			});
		} //for vert

		polygons.push(polygon);
	} //for poly

	polygons
}

impl MeshBoolImpl {
	///Triangulates the faces. In this case, the halfedge_ vector is not yet a set
	///of triangles as required by this data structure, but is instead a set of
	///general faces with the input faceEdge vector having length of the number of
	///faces + 1. The values are indicies into the halfedge_ vector for the first
	///edge of each face, with the final value being the length of the halfedge_
	///vector itself. Upon return, halfedge_ has been lengthened and properly
	///represents the mesh as a set of triangles as usual. In this process the
	///faceNormal_ values are retained, repeated as necessary.
	pub fn face2tri(&mut self, face_edge: &[i32], halfedge_ref: &[TriRef], allow_convex: bool) {
		let general_triangulation = |face| {
			let normal = self.face_normal[face];
			let projection = get_axis_aligned_projection(normal);
			let polys = project_polygons(
				&assemble_halfedges(
					&self.halfedge[(face_edge[face] as usize)..(face_edge[face + 1] as usize)],
					face_edge[face],
				),
				&self.halfedge,
				&self.vert_pos,
				projection,
			);

			triangulate_idx(&polys, self.epsilon, allow_convex)
		};

		let mut tri_verts = Vec::with_capacity(face_edge.len());
		let mut tri_normal = Vec::with_capacity(face_edge.len());
		let mut tri_prop = Vec::with_capacity(face_edge.len());
		let mut tri_ref = Vec::with_capacity(face_edge.len());
		for face in 0..(face_edge.len() - 1) {
			self.process_face(
				face_edge,
				&halfedge_ref,
				general_triangulation,
				|_, tri, normal, r| {
					let mut verts = Vector3::default();
					let mut props = Vector3::default();
					for i in 0..3 {
						verts[i] = self.halfedge[tri[i] as usize].start_vert;
						props[i] = self.halfedge[tri[i] as usize].prop_vert;
					}

					tri_verts.push(verts);
					tri_prop.push(props);
					tri_normal.push(normal);
					tri_ref.push(r);
				},
				face,
			);
		}

		self.face_normal = tri_normal;
		self.create_halfedges(tri_prop, tri_verts);
		self.mesh_relation.tri_ref = tri_ref;
	}

	fn process_face(
		&self,
		face_edge: &[i32],
		halfedge_ref: &[TriRef],
		mut general: impl FnMut(usize) -> Vec<Vector3<i32>>,
		mut add_tri: impl FnMut(usize, Vector3<i32>, Vector3<f64>, TriRef),
		face: usize,
	) {
		let first_edge_i32 = face_edge[face];
		let first_edge = first_edge_i32 as usize;
		let last_edge = face_edge[face + 1];
		let num_edge = last_edge - first_edge_i32;
		if num_edge == 0 {
			return;
		}
		debug_assert!(num_edge >= 3, "face has less than three edges.");
		let normal = self.face_normal[face];

		if num_edge == 3
		//single triangle
		{
			let mut tri_edge = Vector3::new(first_edge_i32, first_edge_i32 + 1, first_edge_i32 + 2);
			let mut tri = Vector3::new(
				self.halfedge[first_edge].start_vert,
				self.halfedge[first_edge + 1].start_vert,
				self.halfedge[first_edge + 2].start_vert,
			);
			let mut ends = Vector3::new(
				self.halfedge[first_edge].end_vert,
				self.halfedge[first_edge + 1].end_vert,
				self.halfedge[first_edge + 2].end_vert,
			);

			if ends[0] == tri[2] {
				let switcheroo = tri_edge.deref_mut();
				mem::swap(&mut switcheroo.y, &mut switcheroo.z);
				let switcheroo = tri.deref_mut();
				mem::swap(&mut switcheroo.y, &mut switcheroo.z);
				let switcheroo = ends.deref_mut();
				mem::swap(&mut switcheroo.y, &mut switcheroo.z);
			}

			debug_assert!(
				ends[0] == tri[1] && ends[1] == tri[2] && ends[2] == tri[0],
				"These 3 edges do not form a triangle!"
			);

			add_tri(face, tri_edge, normal, halfedge_ref[first_edge]);
		} else if num_edge == 4
		//pair of triangles
		{
			let projection = get_axis_aligned_projection(normal);
			let tri_ccw = |tri: Vector3<i32>| {
				ccw(
					projection * self.vert_pos[self.halfedge[tri[0] as usize].start_vert as usize],
					projection * self.vert_pos[self.halfedge[tri[1] as usize].start_vert as usize],
					projection * self.vert_pos[self.halfedge[tri[2] as usize].start_vert as usize],
					self.epsilon,
				) >= 0
			};

			let quad = &assemble_halfedges(
				&self.halfedge[face_edge[face] as usize..face_edge[face + 1] as usize],
				face_edge[face],
			)[0];

			let tris = [
				Matrix3x2::<i32>::new(quad[0], quad[0], quad[1], quad[2], quad[2], quad[3]),
				Matrix3x2::<i32>::new(quad[1], quad[0], quad[2], quad[1], quad[3], quad[3]),
			];

			let mut choice = 0;
			if !(tri_ccw(tris[0].column(0).into()) && tri_ccw(tris[0].column(1).into())) {
				choice = 1;
			} else if tri_ccw(tris[1].column(0).into()) && tri_ccw(tris[1].column(1).into()) {
				let diag0 = self.vert_pos[self.halfedge[quad[0] as usize].start_vert as usize]
					- self.vert_pos[self.halfedge[quad[2] as usize].start_vert as usize];
				let diag1 = self.vert_pos[self.halfedge[quad[1] as usize].start_vert as usize]
					- self.vert_pos[self.halfedge[quad[3] as usize].start_vert as usize];

				if diag0.magnitude_squared() > diag1.magnitude_squared() {
					choice = 1;
				}
			}

			for tri in tris[choice].column_iter() {
				add_tri(face, tri.into(), normal, halfedge_ref[first_edge]);
			}
		} else {
			for tri in general(face) {
				add_tri(face, tri, normal, halfedge_ref[first_edge]);
			}
		}
	}
}
