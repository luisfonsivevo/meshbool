use crate::boolean3::Boolean3;
pub use crate::common::OpType;
pub use crate::meshboolimpl::MeshBoolImpl;
use crate::meshboolimpl::Relation;
use crate::shared::normal_transform;
use nalgebra::{Matrix3, Matrix3x4, Point3, UnitQuaternion, Vector2, Vector3};
use std::ops::{Add, AddAssign, BitXor, BitXorAssign, Sub, SubAssign};

mod boolean3;
mod boolean_result;
mod collider;
mod common;
mod constructors;
mod disjoint_sets;
mod edge_op;
mod face_op;
mod mesh_fixes;
pub mod meshboolimpl;
mod parallel;
mod polygon;
mod properties;
mod shared;
mod sort;
mod tree2d;
mod utils;
mod vec;

#[test]
fn test() {
	use nalgebra::Vector3;

	//just make sure it don't crash at the sight of the simplest shapes
	let cube1 = MeshBool::cube(Vector3::new(1.0, 1.0, 1.0), true);
	let cube2 = MeshBool::cube(Vector3::new(1.0, 1.0, 1.0), false);

	let union = &cube1 + &cube2;
	println!("{:?}", union.get_mesh_gl(0));

	let difference = &cube1 - &cube2;
	println!("{:?}", difference.get_mesh_gl(0));

	let intersection = &cube1 ^ &cube2;
	println!("{:?}", intersection.get_mesh_gl(0));
}

///@brief Mesh input/output suitable for pushing directly into graphics
///libraries.
///
///The core (non-optional) parts of MeshGL are the triVerts indices buffer and
///the vertProperties interleaved vertex buffer, which follow the conventions of
///OpenGL (and other graphic libraries') buffers and are therefore generally
///easy to map directly to other applications' data structures.
///
///The triVerts vector has a stride of 3 and specifies triangles as
///vertex indices. For triVerts = [2, 4, 5, 3, 1, 6, ...], the triangles are [2,
///4, 5], [3, 1, 6], etc. and likewise the halfedges are [2, 4], [4, 5], [5, 2],
///[3, 1], [1, 6], [6, 3], etc.
///
///The triVerts indices should form a manifold mesh: each of the 3 halfedges of
///each triangle should have exactly one paired halfedge in the list, defined as
///having the first index of one equal to the second index of the other and
///vice-versa. However, this is not always possible - consider e.g. a cube with
///normal-vector properties. Shared vertices would turn the cube into a ball by
///interpolating normals - the common solution is to duplicate each corner
///vertex into 3, each with the same position, but different normals
///corresponding to each face. This is exactly what should be done in MeshGL,
///however we request two additional vectors in this case: mergeFromVert and
///mergeToVert. Each vertex mergeFromVert[i] is merged into vertex
///mergeToVert[i], avoiding unreliable floating-point comparisons to recover the
///manifold topology. These merges are simply a union, so which is from and to
///doesn't matter.
///
///If you don't have merge vectors, you can create them with the Merge() method,
///however this will fail if the mesh is not already manifold within the set
///tolerance. For maximum reliability, always store the merge vectors with the
///mesh, e.g. using the EXT_mesh_manifold extension in glTF.
///
///You can have any number of arbitrary floating-point properties per vertex,
///and they will all be interpolated as necessary during operations. It is up to
///you to keep track of which channel represents what type of data. A few of
///Manifold's methods allow you to specify the channel where normals data
///starts, in order to update it automatically for transforms and such. This
///will be easier if your meshes all use the same channels for properties, but
///this is not a requirement. Operations between meshes with different numbers
///of peroperties will simply use the larger numProp and pad the smaller one
///with zeroes.
///
///On output, the triangles are sorted into runs (runIndex, runOriginalID,
///runTransform) that correspond to different mesh inputs. Other 3D libraries
///may refer to these runs as primitives of a mesh (as in glTF) or draw calls,
///as they often represent different materials on different parts of the mesh.
///It is generally a good idea to maintain a map of OriginalIDs to materials to
///make it easy to reapply them after a set of Boolean operations. These runs
///can also be used as input, and thus also ensure a lossless roundtrip of data
///through MeshGL.
///
///As an example, with runIndex = [0, 6, 18, 21] and runOriginalID = [1, 3, 3],
///there are 7 triangles, where the first two are from the input mesh with ID 1,
///the next 4 are from an input mesh with ID 3, and the last triangle is from a
///different copy (instance) of the input mesh with ID 3. These two instances
///can be distinguished by their different runTransform matrices.
///
///You can reconstruct polygonal faces by assembling all the triangles that are
///from the same run and share the same faceID. These faces will be planar
///within the output tolerance.
///
///The halfedgeTangent vector is used to specify the weighted tangent vectors of
///each halfedge for the purpose of using the Refine methods to create a
///smoothly-interpolated surface. They can also be output when calculated
///automatically by the Smooth functions.
///
///MeshGL is an alias for the standard single-precision version. Use MeshGL64 to
///output the full double precision that Manifold uses internally.
#[derive(Debug)]
pub struct MeshGL {
	/// Number of properties per vertex, always >= 3.
	pub num_prop: u32,
	/// Flat, GL-style interleaved list of all vertex properties: propVal =
	/// vertProperties[vert * numProp + propIdx]. The first three properties are
	/// always the position x, y, z. The stride of the array is numProp.
	pub vert_properties: Vec<f32>,
	/// The vertex indices of the three triangle corners in CCW (from the outside)
	/// order, for each triangle.
	pub tri_verts: Vec<u32>,
	/// Optional: A list of only the vertex indicies that need to be merged to
	/// reconstruct the manifold.
	pub merge_from_vert: Vec<u32>,
	/// Optional: The same length as mergeFromVert, and the corresponding value
	/// contains the vertex to merge with. It will have an identical position, but
	/// the other properties may differ.
	pub merge_to_vert: Vec<u32>,
	/// Optional: Indicates runs of triangles that correspond to a particular
	/// input mesh instance. The runs encompass all of triVerts and are sorted
	/// by runOriginalID. Run i begins at triVerts[runIndex[i]] and ends at
	/// triVerts[runIndex[i+1]]. All runIndex values are divisible by 3. Returned
	/// runIndex will always be 1 longer than runOriginalID, but same length is
	/// also allowed as input: triVerts.size() will be automatically appended in
	/// this case.
	pub run_index: Vec<u32>,
	/// Optional: The OriginalID of the mesh this triangle run came from. This ID
	/// is ideal for reapplying materials to the output mesh. Multiple runs may
	/// have the same ID, e.g. representing different copies of the same input
	/// mesh. If you create an input MeshGL that you want to be able to reference
	/// as one or more originals, be sure to set unique values from ReserveIDs().
	pub run_original_id: Vec<u32>,
	/// Optional: For each run, a 3x4 transform is stored representing how the
	/// corresponding original mesh was transformed to create this triangle run.
	/// This matrix is stored in column-major order and the length of the overall
	/// vector is 12 * runOriginalID.size().
	pub run_transform: Vec<f32>,
	/// Optional: Length NumTri, contains the source face ID this triangle comes
	/// from. Simplification will maintain all edges between triangles with
	/// different faceIDs. Input faceIDs will be maintained to the outputs, but if
	/// none are given, they will be filled in with Manifold's coplanar face
	/// calculation based on mesh tolerance.
	pub face_id: Vec<u32>,
	/// Tolerance for mesh simplification. When creating a Manifold, the tolerance
	/// used will be the maximum of this and a baseline tolerance from the size of
	/// the bounding box. Any edge shorter than tolerance may be collapsed.
	/// Tolerance may be enlarged when floating point error accumulates.
	pub tolerance: f32,
}

impl Default for MeshGL {
	fn default() -> Self {
		Self {
			num_prop: 3,
			tolerance: 0.0,
			vert_properties: Vec::default(),
			tri_verts: Vec::default(),
			merge_from_vert: Vec::default(),
			merge_to_vert: Vec::default(),
			run_index: Vec::default(),
			run_original_id: Vec::default(),
			run_transform: Vec::default(),
			face_id: Vec::default(),
		}
	}
}

impl MeshGL {
	pub fn num_vert(&self) -> usize {
		self.vert_properties.len() / self.num_prop as usize
	}

	pub fn num_tri(&self) -> usize {
		self.tri_verts.len() / 3
	}
}

#[derive(Default)]
pub struct MeshBool {
	meshbool_impl: MeshBoolImpl,
}

impl MeshBool {
	fn invalid() -> Self {
		let mut meshbool = Self::default();
		meshbool.meshbool_impl.status = ManifoldError::InvalidConstruction;
		meshbool
	}

	///Return a copy of the manifold simplified to the given tolerance, but with its
	///actual tolerance value unchanged. If the tolerance is not given or is less
	///than the current tolerance, the current tolerance is used for simplification.
	///The result will contain a subset of the original verts and all surfaces will
	///have moved by less than tolerance.
	pub fn simplify(&self, tolerance: Option<f64>) -> Self {
		let mut meshbool_impl = self.meshbool_impl.clone();
		let old_tolerance = meshbool_impl.tolerance;
		let tolerance = tolerance.unwrap_or(old_tolerance);
		if tolerance > old_tolerance {
			meshbool_impl.tolerance = tolerance;
			meshbool_impl.mark_coplanar();
		}

		meshbool_impl.simplify_topology(0);
		meshbool_impl.finish();
		meshbool_impl.tolerance = tolerance;
		Self::from(meshbool_impl)
	}

	///This removes all relations (originalID, faceID, transform) to ancestor meshes
	///and this new Manifold is marked an original. It also recreates faces
	///- these don't get joined at boundaries where originalID changes, so the
	///reset may allow triangles of flat faces to be further collapsed with
	///Simplify().
	pub fn as_original(&self) -> Self {
		let old_impl = &self.meshbool_impl;
		if old_impl.status != ManifoldError::NoError {
			let mut new_impl = MeshBoolImpl::default();
			new_impl.status = old_impl.status;
			return Self::from(new_impl);
		}

		let mut new_impl = self.meshbool_impl.clone();
		new_impl.initialize_original(false);
		new_impl.mark_coplanar();
		new_impl.initialize_original(true);
		Self::from(new_impl)
	}

	///Move this Manifold in space. This operation can be chained. Transforms are
	///combined and applied lazily.
	///
	///@param v The vector to add to every vertex.
	pub fn translate(&self, v: Point3<f64>) -> Self {
		let mut transform = Matrix3x4::<f64>::identity();
		*transform.column_mut(3) = *v;
		Self::from(self.meshbool_impl.transform(&transform))
	}

	///Scale this Manifold in space. This operation can be chained. Transforms are
	///combined and applied lazily.
	///
	///@param v The vector to multiply every vertex by per component.
	pub fn scale(&self, v: Vector3<f64>) -> Self {
		let mut transform = Matrix3x4::<f64>::identity();
		for i in 0..3 {
			transform[(i, i)] = v[i];
		}

		Self::from(self.meshbool_impl.transform(&transform))
	}

	///Applies an Euler angle rotation to the manifold, first about the X axis, then
	///Y, then Z, in degrees. We use degrees so that we can minimize rounding error,
	///and eliminate it completely for any multiples of 90 degrees. Additionally,
	///more efficient code paths are used to update the manifold when the transforms
	///only rotate by multiples of 90 degrees. This operation can be chained.
	///Transforms are combined and applied lazily.
	///
	///@param xDegrees First rotation, degrees about the X-axis.
	///@param yDegrees Second rotation, degrees about the Y-axis.
	///@param zDegrees Third rotation, degrees about the Z-axis.
	pub fn rotate(&self, x_degrees: f64, y_degrees: f64, z_degrees: f64) -> Self {
		let transform = UnitQuaternion::from_euler_angles(
			x_degrees.to_radians(),
			y_degrees.to_radians(),
			z_degrees.to_radians(),
		)
		.to_homogeneous()
		.fixed_view::<3, 4>(0, 0)
		.into_owned();

		Self::from(self.meshbool_impl.transform(&transform))
	}

	///Transform this Manifold in space. The first three columns form a 3x3 matrix
	///transform and the last is a translation vector. This operation can be
	///chained. Transforms are combined and applied lazily.
	///
	///@param m The affine transform matrix to apply to all the vertices.
	pub fn transform(&self, m: &Matrix3x4<f64>) -> Self {
		Self::from(self.meshbool_impl.transform(&m))
	}

	///	The central operation of this library: the Boolean combines two manifolds
	///	into another by calculating their intersections and removing the unused
	///	portions.
	///	[&epsilon;-valid](https://github.com/elalish/manifold/wiki/Manifold-Library#definition-of-%CE%B5-valid)
	///	inputs will produce &epsilon;-valid output. &epsilon;-invalid input may fail
	///	triangulation.
	///
	///	These operations are optimized to produce nearly-instant results if either
	///	input is empty or their bounding boxes do not overlap.
	///
	///	@param second The other Manifold.
	///	@param op The type of operation to perform.
	pub fn boolean(&self, other: &Self, op: OpType) -> Self {
		Self::from(Boolean3::new(&self.meshbool_impl, &other.meshbool_impl, op).result(op))
	}

	///The most complete output of this library, returning a MeshGL that is designed
	///to easily push into a renderer, including all interleaved vertex properties
	///that may have been input. It also includes relations to all the input meshes
	///that form a part of this result and the transforms applied to each.
	///
	///@param normalIdx If the original MeshGL inputs that formed this manifold had
	///properties corresponding to normal vectors, you can specify the first of the
	///three consecutive property channels forming the (x, y, z) normals, which will
	///cause this output MeshGL to automatically update these normals according to
	///the applied transforms and front/back side. normalIdx + 3 must be <=
	///numProp, and all original MeshGLs must use the same channels for their
	///normals.
	pub fn get_mesh_gl_internal(meshbool_impl: &MeshBoolImpl, normal_idx: i32) -> MeshGL {
		let num_prop = meshbool_impl.num_prop();
		let num_vert = meshbool_impl.num_prop_vert();
		let num_tri = meshbool_impl.num_tri();

		let is_original = meshbool_impl.mesh_relation.original_id >= 0;
		let update_normals = !is_original && normal_idx >= 0;

		let out_num_prop: u32 = 3 + num_prop as u32;
		let tolerance = meshbool_impl
			.tolerance
			.max((f32::EPSILON as f64) * meshbool_impl.bbox.scale()) as f32;

		let mut tri_verts: Vec<u32> = vec![0; 3 * num_tri];

		// Sort the triangles into runs
		let mut face_id: Vec<u32> = vec![0; num_tri];
		let mut tri_new2old: Vec<_> = (0..num_tri).map(|i| i as i32).collect();
		let tri_ref = &meshbool_impl.mesh_relation.tri_ref;
		// Don't sort originals - keep them in order
		if !is_original {
			tri_new2old
				.sort_by_key(|&i| (tri_ref[i as usize].original_id, tri_ref[i as usize].mesh_id));
		}

		let mut run_index: Vec<u32> = Vec::new();
		let mut run_original_id: Vec<u32> = Vec::new();
		let mut run_transform: Vec<f32> = Vec::new();

		let mut run_normal_transform: Vec<Matrix3<f64>> = Vec::new();
		let mut add_run = |tri, rel: Relation| {
			run_index.push((3 * tri) as u32);
			run_original_id.push(rel.original_id as u32);
			if update_normals {
				run_normal_transform.push(
					normal_transform(&rel.transform) * (if rel.back_side { -1.0 } else { 1.0 }),
				);
			}

			if !is_original {
				for col in 0..4 {
					for row in 0..3 {
						run_transform.push(rel.transform[(row, col)] as f32)
					}
				}
			}
		};

		let mut mesh_id_transform = meshbool_impl.mesh_relation.mesh_id_transform.clone();
		let mut last_id = -1;
		for tri in 0..num_tri {
			let old_tri = tri_new2old[tri] as usize;
			let r#ref = tri_ref[old_tri];
			let mesh_id = r#ref.mesh_id;

			face_id[tri] = (if r#ref.face_id >= 0 {
				r#ref.face_id
			} else {
				r#ref.coplanar_id
			}) as u32;
			for i in 0..3 {
				tri_verts[3 * tri + i] = meshbool_impl.halfedge[3 * old_tri + i].start_vert as u32;
			}

			if mesh_id != last_id {
				let it = mesh_id_transform.remove(&mesh_id);
				let rel = it.unwrap_or_default();
				add_run(tri, rel);
				last_id = mesh_id;
			}
		}

		// Add runs for originals that did not contribute any faces to the output
		for pair in mesh_id_transform {
			add_run(num_tri, pair.1);
		}

		run_index.push((3 * num_tri) as u32);

		// Early return for no props
		if num_prop == 0 {
			let mut vert_properties: Vec<f32> = vec![0.0; 3 * num_vert];
			for i in 0..num_vert {
				let v = meshbool_impl.vert_pos[i];
				vert_properties[3 * i] = v.x as f32;
				vert_properties[3 * i + 1] = v.y as f32;
				vert_properties[3 * i + 2] = v.z as f32;
			}

			return MeshGL {
				num_prop: out_num_prop,
				vert_properties,
				tri_verts,
				merge_from_vert: Vec::default(),
				merge_to_vert: Vec::default(),
				run_index,
				run_original_id,
				run_transform,
				face_id,
				tolerance,
			};
		}

		// Duplicate verts with different props
		let mut vert2idx: Vec<i32> = vec![-1; meshbool_impl.num_vert()];
		let mut vert_prop_pair: Vec<Vec<Vector2<i32>>> = vec![Vec::new(); meshbool_impl.num_vert()];
		let mut vert_properties: Vec<f32> = Vec::with_capacity(num_vert * (out_num_prop as usize));

		let mut merge_from_vert: Vec<u32> = Vec::new();
		let mut merge_to_vert: Vec<u32> = Vec::new();

		for run in 0..run_original_id.len() {
			for tri in (run_index[run] / 3)..run_index[run + 1] / 3 {
				let tri = tri as usize;
				for i in 0..3 {
					let prop =
						meshbool_impl.halfedge[3 * (tri_new2old[tri] as usize) + i].prop_vert;
					let vert = tri_verts[3 * tri + i] as usize;

					let bin = &mut vert_prop_pair[vert];
					let mut b_found = false;
					for b in bin.iter() {
						if b.x == prop {
							b_found = true;
							tri_verts[3 * tri + i] = b.y as u32;
							break;
						}
					}

					if b_found {
						continue;
					}
					let idx = vert_properties.len() / (out_num_prop as usize);
					tri_verts[3 * tri + i] = idx as u32;
					bin.push(Vector2::new(prop, idx as i32));

					for p in 0..3 {
						vert_properties.push(meshbool_impl.vert_pos[vert][p] as f32);
					}
					for p in 0..num_prop {
						vert_properties
							.push(meshbool_impl.properties[(prop as usize) * num_prop + p] as f32);
					}

					if update_normals {
						let mut normal = Vector3::<f64>::default();
						let start = vert_properties.len() - (out_num_prop as usize);
						for i in 0..3 {
							normal[i] =
								vert_properties[start + 3 + (normal_idx as usize) + i] as f64;
						}

						normal = (run_normal_transform[run] * normal).normalize();
						for i in 0..3 {
							vert_properties[start + 3 + (normal_idx as usize) + i] =
								normal[i] as f32;
						}
					}

					if vert2idx[vert] == -1 {
						vert2idx[vert] = idx as i32;
					} else {
						merge_from_vert.push(idx as u32);
						merge_to_vert.push(vert2idx[vert] as u32);
					}
				}
			}
		}

		MeshGL {
			num_prop: out_num_prop,
			vert_properties,
			tri_verts,
			merge_from_vert,
			merge_to_vert,
			run_index,
			run_original_id,
			run_transform,
			face_id,
			tolerance,
		}
	}

	pub fn get_mesh_gl(&self, normal_idx: i32) -> MeshGL {
		Self::get_mesh_gl_internal(&self.meshbool_impl, normal_idx)
	}

	pub fn from_meshgl(mesh_gl: MeshGL) -> Self {
		Self::from(MeshBoolImpl::from_meshgl(mesh_gl))
	}
}

impl From<MeshBoolImpl> for MeshBool {
	fn from(value: MeshBoolImpl) -> Self {
		Self {
			meshbool_impl: value,
		}
	}
}

impl Add for &MeshBool {
	type Output = MeshBool;
	fn add(self, rhs: Self) -> Self::Output {
		self.boolean(rhs, OpType::Add)
	}
}

impl AddAssign<&Self> for MeshBool {
	fn add_assign(&mut self, rhs: &Self) {
		*self = self.boolean(rhs, OpType::Add);
	}
}

impl Sub for &MeshBool {
	type Output = MeshBool;
	fn sub(self, rhs: Self) -> Self::Output {
		self.boolean(rhs, OpType::Subtract)
	}
}

impl SubAssign<&Self> for MeshBool {
	fn sub_assign(&mut self, rhs: &Self) {
		*self = self.boolean(rhs, OpType::Subtract);
	}
}

impl BitXor for &MeshBool {
	type Output = MeshBool;
	fn bitxor(self, rhs: Self) -> Self::Output {
		self.boolean(rhs, OpType::Intersect)
	}
}

impl BitXorAssign<&Self> for MeshBool {
	fn bitxor_assign(&mut self, rhs: &Self) {
		*self = self.boolean(rhs, OpType::Intersect);
	}
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ManifoldError {
	NoError,
	NonFiniteVertex,
	InvalidConstruction,
	ResultTooLarge,
	NotManifold,
	MissingPositionProperties,
	MergeVectorsDifferentLengths,
	TransformWrongLength,
	RunIndexWrongLength,
	FaceIDWrongLength,
	MergeIndexOutOfBounds,
	VertexOutOfBounds,
}
