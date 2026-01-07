use nalgebra::{Vector3, Vector4};

use crate::meshboolimpl::MeshBoolImpl;
use crate::shared::{TriRef, next_halfedge, safe_normalize};

///Get the angle between two unit-vectors.
/*fn angle_between(a: Vector3<f64>, b: Vector3<f64>) -> f64 {
	let dot: f64 = a.dot(&b);
	if dot >= 1.0 {
		0.0
	} else {
		if dot <= -1.0 {
			core::f64::consts::PI
		} else {
			dot.acos()
		}
	}
}*/

///Calculate a tangent vector in the form of a weighted cubic Bezier taking as
///input the desired tangent direction (length doesn't matter) and the edge
///vector to the neighboring vertex. In a symmetric situation where the tangents
///at each end are mirror images of each other, this will result in a circular
///arc.
fn circular_tangent(tangent: Vector3<f64>, edge_vec: Vector3<f64>) -> Vector4<f64> {
	let dir: Vector3<f64> = safe_normalize(tangent);

	let weight: f64 = 0.5_f64.max(dir.dot(&safe_normalize(edge_vec)));
	// Quadratic weighted bezier for circular interpolation
	let bz2: Vector4<f64> = (dir * 0.5 * edge_vec.norm()).push(weight);
	// Equivalent cubic weighted bezier
	let bz3: Vector4<f64> = Vector4::new(0.0, 0.0, 0.0, 1.0).lerp(&bz2, 2.0 / 3.0);
	// Convert from homogeneous form to geometric form
	return (bz3.xyz() / bz3.w).push(bz3.w);
}

impl MeshBoolImpl {
	///Returns a circular tangent for the requested halfedge, orthogonal to the
	///given normal vector, and avoiding folding.
	pub fn tangent_from_normal(&self, normal: &Vector3<f64>, halfedge: i32) -> Vector4<f64> {
		let edge = self.halfedge[halfedge as usize];
		let edge_vec: Vector3<f64> =
			self.vert_pos[edge.end_vert as usize] - self.vert_pos[edge.start_vert as usize];
		let edge_normal: Vector3<f64> = self.face_normal[halfedge as usize / 3]
			+ self.face_normal[edge.paired_halfedge as usize / 3];
		let dir: Vector3<f64> = edge_normal.cross(&edge_vec).cross(&normal);
		return circular_tangent(dir, edge_vec);
	}

	///Returns true if this halfedge should be marked as the interior of a quad, as
	///defined by its two triangles referring to the same face, and those triangles
	///having no further face neighbors beyond.
	pub fn is_inside_quad(&self, halfedge: i32) -> bool {
		// if self.halfedge_tangent.len() > 0 {
		//   return self.halfedge_tangent[halfedge as usize].w < 0;
		// }
		let tri: i32 = halfedge / 3;
		let tref: TriRef = self.mesh_relation.tri_ref[tri as usize];
		let pair: i32 = self.halfedge[halfedge as usize].paired_halfedge;
		let pair_tri: i32 = pair / 3;
		let pair_ref: TriRef = self.mesh_relation.tri_ref[pair_tri as usize];
		if !tref.same_face(&pair_ref) {
			return false;
		}

		let same_face = |halfedge: i32, tref: &TriRef| {
			tref.same_face(
				&self.mesh_relation.tri_ref
					[self.halfedge[halfedge as usize].paired_halfedge as usize / 3],
			)
		};

		let mut neighbor: i32 = next_halfedge(halfedge);
		if same_face(neighbor, &tref) {
			return false;
		}
		neighbor = next_halfedge(neighbor);
		if same_face(neighbor, &tref) {
			return false;
		}
		neighbor = next_halfedge(pair);
		if same_face(neighbor, &pair_ref) {
			return false;
		}
		neighbor = next_halfedge(neighbor);
		if same_face(neighbor, &pair_ref) {
			return false;
		}
		return true;
	}

	///Returns true if this halfedge is an interior of a quad, as defined by its
	///halfedge tangent having negative weight.
	pub fn is_marked_inside_quad(&self, _halfedge: i32) -> bool {
		// if !self.halfedge_tangent.is_empty() {
		// 	return self.halfedge_tangent[halfedge as usize].w < 0;
		// }
		return false;
	}

	///Find faces containing at least 3 triangles - these will not have
	///interpolated normals - all their vert normals must match their face normal.
	pub fn flat_faces(&self) -> Vec<bool> {
		let num_tri: i32 = self.num_tri() as i32;
		let mut tri_is_flat_face: Vec<bool> = vec![false; num_tri as usize];
		(0..num_tri).into_iter().for_each(|tri| {
			let tref = &self.mesh_relation.tri_ref[tri as usize];
			let mut face_neighbors: i32 = 0;
			let mut face_tris: Vector3<i32> = Vector3::new(-1, -1, -1);
			for j in [0, 1, 2] {
				let neighbor_tri: i32 = self.halfedge[(3 * tri + j) as usize].paired_halfedge / 3;
				let j_ref = &self.mesh_relation.tri_ref[neighbor_tri as usize];
				if j_ref.same_face(tref) {
					face_neighbors += 1;
					face_tris[j as usize] = neighbor_tri;
				}
			}
			if face_neighbors > 1 {
				tri_is_flat_face[tri as usize] = true;
				for j in [0, 1, 2] {
					if face_tris[j] >= 0 {
						tri_is_flat_face[face_tris[j as usize] as usize] = true;
					}
				}
			}
		});
		return tri_is_flat_face;
	}

	///Returns a vector of length numVert that has a tri that is part of a
	///neighboring flat face if there is only one flat face. If there are none it
	///gets -1, and if there are more than one it gets -2.
	pub fn vert_flat_face(&self, flat_faces: &[bool]) -> Vec<i32> {
		let mut vert_flat_face: Vec<i32> = vec![-1; self.num_vert()];
		let mut vert_ref: Vec<TriRef> = vec![
			TriRef {
				mesh_id: -1,
				original_id: -1,
				face_id: -1,
				coplanar_id: -1,
			};
			self.num_vert()
		];
		for tri in 0..self.num_tri() {
			if flat_faces[tri] {
				for j in [0, 1, 2] {
					let vert: i32 = self.halfedge[(3 * tri + j) as usize].start_vert;
					if vert_ref[vert as usize].same_face(&self.mesh_relation.tri_ref[tri as usize])
					{
						continue;
					}
					vert_ref[vert as usize] = self.mesh_relation.tri_ref[tri as usize];
					vert_flat_face[vert as usize] = if vert_flat_face[vert as usize] == -1 {
						tri as i32
					} else {
						-2
					};
				}
			}
		}
		return vert_flat_face;
	}

	////Instead of calculating the internal shared normals like CalculateNormals
	////does, this method fills in vertex properties, unshared across edges that
	////are bent more than minSharpAngle.
	/*pub fn set_normals(&mut self, normal_idx: i32, min_sharp_angle: f64) {
		if self.is_empty() {
			return;
		}
		if normal_idx < 0 {
			return;
		}

		let old_num_prop: i32 = self.num_prop() as i32;

		let tri_is_flat_face: Vec<bool> = self.flat_faces();
		let vert_flat_face: Vec<i32> = self.vert_flat_face(&tri_is_flat_face);
		let mut vert_num_sharp: Vec<i32> = vec![0; self.num_vert()];
		for e in 0..self.halfedge.len() {
			if !self.halfedge[e].is_forward() {
				continue;
			}
			let pair: i32 = self.halfedge[e].paired_halfedge;
			let tri1: i32 = (e / 3) as i32;
			let tri2: i32 = pair / 3;
			let dihedral: f64 = self.face_normal[tri1 as usize]
				.dot(&self.face_normal[tri2 as usize])
				.acos()
				.to_degrees();
			if dihedral > min_sharp_angle {
				vert_num_sharp[self.halfedge[e].start_vert as usize] += 1;
				vert_num_sharp[self.halfedge[e].end_vert as usize] += 1;
			} else {
				let face_split = tri_is_flat_face[tri1 as usize] != tri_is_flat_face[tri2 as usize]
					|| (tri_is_flat_face[tri1 as usize]
						&& tri_is_flat_face[tri2 as usize]
						&& !self.mesh_relation.tri_ref[tri1 as usize]
							.same_face(&self.mesh_relation.tri_ref[tri2 as usize]));
				if vert_flat_face[self.halfedge[e].start_vert as usize] == -2 && face_split {
					vert_num_sharp[self.halfedge[e].start_vert as usize] += 1;
				}
				if vert_flat_face[self.halfedge[e].end_vert as usize] == -2 && face_split {
					vert_num_sharp[self.halfedge[e].end_vert as usize] += 1;
				}
			}
		}

		let num_prop: i32 = old_num_prop.max(normal_idx + 3);
		let mut old_properties: Vec<f64> = vec![0.0; num_prop as usize * self.num_prop_vert()];
		core::mem::swap(&mut self.properties, &mut old_properties);
		self.num_prop = num_prop;

		let mut old_halfedge_prop: Vec<i32> = vec![0; self.halfedge.len()];
		(0..self.halfedge.len()).for_each(|i| {
			old_halfedge_prop[i] = self.halfedge[i].prop_vert;
			self.halfedge[i].prop_vert = -1;
		});

		let num_edge: i32 = self.halfedge.len() as i32;
		for start_edge in 0..num_edge {
			if self.halfedge[start_edge as usize].prop_vert >= 0 {
				continue;
			}
			let vert: i32 = self.halfedge[start_edge as usize].start_vert;

			if vert_num_sharp[vert as usize] < 2 {
				// vertex has single normal
				let normal: Vector3<f64> = if vert_flat_face[vert as usize] >= 0 {
					self.face_normal[vert_flat_face[vert as usize] as usize]
				} else {
					self.vert_normal[vert as usize]
				};
				let mut last_prop: i32 = -1;
				self.for_vert_mut(start_edge, |self_mut, current| {
					let prop: i32 = old_halfedge_prop[current as usize];
					self_mut.halfedge[current as usize].prop_vert = prop;
					if prop == last_prop {
						return;
					}
					last_prop = prop;
					// update property vertex
					let start = &old_properties[(prop * old_num_prop) as usize..];
					self_mut.properties
						[(prop * num_prop) as usize..(prop * num_prop + old_num_prop) as usize]
						.copy_from_slice(&start[..old_num_prop as usize]);
					for i in [0, 1, 2] {
						self_mut.properties[(prop * num_prop + normal_idx + i) as usize] =
							normal[i as usize];
					}
				});
			} else {
				// vertex has multiple normals
				let center_pos: Vector3<f64> = self.vert_pos[vert as usize].coords;
				// Length degree
				let mut group: Vec<i32> = vec![];
				// Length number of normals
				let mut normals: Vec<Vector3<f64>> = vec![];
				let mut current: i32 = start_edge;
				let mut prev_face: i32 = current / 3;

				loop {
					// find a sharp edge to start on
					let next: i32 = next_halfedge(self.halfedge[current as usize].paired_halfedge);
					let face: i32 = next / 3;

					let dihedral: f64 = self.face_normal[face as usize]
						.dot(&self.face_normal[prev_face as usize])
						.acos()
						.to_degrees();
					if dihedral > min_sharp_angle
						|| tri_is_flat_face[face as usize] != tri_is_flat_face[prev_face as usize]
						|| (tri_is_flat_face[face as usize]
							&& tri_is_flat_face[prev_face as usize]
							&& !self.mesh_relation.tri_ref[face as usize]
								.same_face(&self.mesh_relation.tri_ref[prev_face as usize]))
					{
						break;
					}
					current = next;
					prev_face = face;
					if current == start_edge {
						break;
					}
				}

				let end_edge: i32 = current;

				struct FaceEdge {
					face: i32,
					edge_vec: Vector3<f64>,
				}

				// calculate pseudo-normals between each sharp edge
				self.for_vert_fun::<FaceEdge>(
					end_edge,
					|current| {
						if self.is_inside_quad(current) {
							return FaceEdge {
								face: current / 3,
								edge_vec: Vector3::repeat(core::f64::NAN),
							};
						}
						let vert: i32 = self.halfedge[current as usize].end_vert;
						let mut pos: Vector3<f64> = self.vert_pos[vert as usize].coords;
						if vert_num_sharp[vert as usize] < 2 {
							// opposite vert has fixed normal
							let normal: Vector3<f64> = if vert_flat_face[vert as usize] >= 0 {
								self.face_normal[vert_flat_face[vert as usize] as usize]
							} else {
								self.vert_normal[vert as usize]
							};

							// Flair out the normal we're calculating to give the edge a
							// more constant curvature to meet the opposite normal. Achieve
							// this by pointing the tangent toward the opposite bezier
							// control point instead of the vert itself.
							pos += self
								.tangent_from_normal(
									&normal,
									self.halfedge[current as usize].paired_halfedge,
								)
								.xyz();
						}
						return FaceEdge {
							face: current / 3,
							edge_vec: safe_normalize(pos - center_pos),
						};
					},
					|_, here: &FaceEdge, next: &mut FaceEdge| {
						let dihedral: f64 = self.face_normal[here.face as usize]
							.dot(&self.face_normal[next.face as usize])
							.acos()
							.to_degrees();
						if dihedral > min_sharp_angle
							|| tri_is_flat_face[here.face as usize]
								!= tri_is_flat_face[next.face as usize]
							|| (tri_is_flat_face[here.face as usize]
								&& tri_is_flat_face[next.face as usize]
								&& !self.mesh_relation.tri_ref[here.face as usize]
									.same_face(&self.mesh_relation.tri_ref[next.face as usize]))
						{
							normals.push(Vector3::repeat(0.0_f64));
						}
						group.push(normals.len() as i32 - 1);
						if next.edge_vec.x.is_finite() {
							*normals.last_mut().unwrap() +=
								safe_normalize(next.edge_vec.cross(&here.edge_vec))
									* angle_between(here.edge_vec, next.edge_vec);
						} else {
							next.edge_vec = here.edge_vec;
						}
					},
				);

				for normal in normals.iter_mut() {
					*normal = safe_normalize(normal.clone());
				}

				let mut last_group: i32 = 0;
				let mut last_prop: i32 = -1;
				let mut new_prop: i32 = -1;
				let mut idx: i32 = 0;
				self.for_vert_mut(end_edge, |self_mut, current1| {
					let prop: i32 = old_halfedge_prop[current1 as usize];
					let start = &mut old_properties[(prop * old_num_prop) as usize..];

					if group[idx as usize] != last_group
						&& group[idx as usize] != 0
						&& prop == last_prop
					{
						// split property vertex, duplicating but with an updated normal
						last_group = group[idx as usize];
						new_prop = self_mut.num_prop_vert() as i32;
						self_mut
							.properties
							.resize(self_mut.properties.len() + num_prop as usize, 0.0);
						self_mut.properties[(new_prop * num_prop) as usize
							..(new_prop * num_prop + old_num_prop) as usize]
							.copy_from_slice(&start[..old_num_prop as usize]);
						for i in [0, 1, 2] {
							self_mut.properties[(new_prop * num_prop + normal_idx + i) as usize] =
								normals[group[idx as usize] as usize][i as usize];
						}
					} else if prop != last_prop {
						// update property vertex
						last_prop = prop;
						new_prop = prop;
						self_mut.properties
							[(prop * num_prop) as usize..(prop * num_prop + old_num_prop) as usize]
							.copy_from_slice(&start[..old_num_prop as usize]);
						for i in [0, 1, 2] {
							self_mut.properties[(prop * num_prop + normal_idx + i) as usize] =
								normals[group[idx as usize] as usize][i as usize];
						}
					}

					// point to updated property vertex
					self_mut.halfedge[current1 as usize].prop_vert = new_prop;
					idx += 1;
				});
			}
		}
	}*/
}
