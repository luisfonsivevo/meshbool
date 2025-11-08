use std::ffi::c_int;

use meshbool::MeshBool as Manifold;
use meshbool::MeshGL;
//AABB, ManifoldError,, OpType

type MeshGL64 = MeshGL;

// Polygons

// extern "C" fn manifold_simple_polygon(
// 	mem: *mut ::std::os::raw::c_void,
// 	ps: *mut ManifoldVec2,
// 	length: usize,
// ) -> *mut ManifoldSimplePolygon {
// }
//
// extern "C" fn manifold_polygons(
// 	mem: *mut ::std::os::raw::c_void,
// 	ps: *mut *mut ManifoldSimplePolygon,
// 	length: usize,
// ) -> *mut ManifoldPolygons {
// }

// extern "C" fn manifold_simple_polygon_length(p: *mut ManifoldSimplePolygon) -> usize {
// }
//
// extern "C" fn manifold_polygons_length(ps: *mut ManifoldPolygons) -> usize {}
//
// extern "C" fn manifold_polygons_simple_length(ps: *mut ManifoldPolygons, idx: usize) -> usize {}
//
// extern "C" fn manifold_simple_polygon_get_point(
// 	p: *mut ManifoldSimplePolygon,
// 	idx: usize,
// ) -> ManifoldVec2 {
// }
//
// extern "C" fn manifold_polygons_get_simple(
// 	mem: *mut ::std::os::raw::c_void,
// 	ps: *mut ManifoldPolygons,
// 	idx: usize,
// ) -> *mut ManifoldSimplePolygon {
// }
//
// extern "C" fn manifold_polygons_get_point(
// 	ps: *mut ManifoldPolygons,
// 	simple_idx: usize,
// 	pt_idx: usize,
// ) -> ManifoldVec2 {
// }

// Mesh Construction

extern "C" fn manifold_meshgl(
	mem: *mut ::std::os::raw::c_void,
	vert_props: *mut f32,
	n_verts: usize,
	n_props: usize,
	tri_verts: *mut u32,
	n_tris: usize,
) -> *mut MeshGL {
	let mem = mem as *mut MeshGL;
	unsafe {
		core::ptr::write(mem, MeshGL::default());
		let m = mem.as_mut().expect("Invalid MeshGL pointer");
		m.num_prop = n_props as u32;
		m.vert_properties = core::slice::from_raw_parts(vert_props, n_verts * n_props).to_vec();
		m.tri_verts = core::slice::from_raw_parts(tri_verts, n_tris * 3).to_vec();
	}
	mem
}

extern "C" fn manifold_meshgl_w_tangents(
	mem: *mut ::std::os::raw::c_void,
	vert_props: *mut f32,
	n_verts: usize,
	n_props: usize,
	tri_verts: *mut u32,
	n_tris: usize,
	halfedge_tangent: *mut f32,
) -> *mut MeshGL {
	let mem = mem as *mut MeshGL;
	unsafe {
		core::ptr::write(mem, MeshGL::default());
		let m = mem.as_mut().expect("Invalid MeshGL pointer");
		m.num_prop = n_props as u32;
		m.vert_properties = core::slice::from_raw_parts(vert_props, n_verts * n_props).to_vec();
		m.tri_verts = core::slice::from_raw_parts(tri_verts, n_tris * 3).to_vec();
		// TODO: add halfedge_tangent
		_ = halfedge_tangent;
	}
	mem
}

extern "C" fn manifold_get_meshgl(
	mem: *mut ::std::os::raw::c_void,
	m: *mut Manifold,
) -> *mut MeshGL {
	let mem = mem as *mut MeshGL;
	unsafe {
		let m = m.as_ref().expect("Invalid Manifold pointer");
		// TODO: Make sure this is correct
		core::ptr::write(mem, m.get_mesh_gl(0));
	}
	mem
}

extern "C" fn manifold_meshgl_copy(
	mem: *mut ::std::os::raw::c_void,
	m: *mut MeshGL,
) -> *mut MeshGL {
	let mem = mem as *mut MeshGL;
	unsafe {
		core::ptr::copy(m, mem, 1);
	}
	mem
}

// extern "C" fn manifold_meshgl_merge(
// 	mem: *mut ::std::os::raw::c_void,
// 	m: *mut MeshGL,
// ) -> *mut MeshGL {
// }

extern "C" fn manifold_meshgl64(
	mem: *mut ::std::os::raw::c_void,
	vert_props: *mut f64,
	n_verts: usize,
	n_props: usize,
	tri_verts: *mut u64,
	n_tris: usize,
) -> *mut MeshGL64 {
	let mem = mem as *mut MeshGL;
	unsafe {
		core::ptr::write(mem, MeshGL::default());
		let m = mem.as_mut().expect("Invalid MeshGL pointer");
		m.num_prop = n_props as u32;
		m.vert_properties = core::slice::from_raw_parts(vert_props, n_verts * n_props)
			.iter()
			.map(|v| *v as f32)
			.collect::<Vec<f32>>();
		m.tri_verts = core::slice::from_raw_parts(tri_verts, n_tris * 3)
			.iter()
			.map(|v| *v as u32)
			.collect::<Vec<u32>>();
	}
	mem
}

extern "C" fn manifold_meshgl64_w_tangents(
	mem: *mut ::std::os::raw::c_void,
	vert_props: *mut f64,
	n_verts: usize,
	n_props: usize,
	tri_verts: *mut u64,
	n_tris: usize,
	halfedge_tangent: *mut f64,
) -> *mut MeshGL64 {
	let mem = mem as *mut MeshGL;
	unsafe {
		core::ptr::write(mem, MeshGL::default());
		let m = mem.as_mut().expect("Invalid MeshGL pointer");
		m.num_prop = n_props as u32;
		m.vert_properties = core::slice::from_raw_parts(vert_props, n_verts * n_props)
			.iter()
			.map(|v| *v as f32)
			.collect::<Vec<f32>>();
		m.tri_verts = core::slice::from_raw_parts(tri_verts, n_tris * 3)
			.iter()
			.map(|v| *v as u32)
			.collect::<Vec<u32>>();
		// TODO: add halfedge_tangent
		_ = halfedge_tangent;
	}
	mem
}

extern "C" fn manifold_get_meshgl64(
	mem: *mut ::std::os::raw::c_void,
	m: *mut Manifold,
) -> *mut MeshGL64 {
	let mem = mem as *mut MeshGL64;
	unsafe {
		let m = m.as_ref().expect("Invalid Manifold pointer");
		// TODO: Make sure this is correct
		core::ptr::write(mem, m.get_mesh_gl(0));
	}
	mem
}

extern "C" fn manifold_meshgl64_copy(
	mem: *mut ::std::os::raw::c_void,
	m: *mut MeshGL64,
) -> *mut MeshGL64 {
	let mem = mem as *mut MeshGL64;
	unsafe {
		core::ptr::copy(m, mem, 1);
	}
	mem
}

// extern "C" fn manifold_meshgl64_merge(
// 	mem: *mut ::std::os::raw::c_void,
// 	m: *mut MeshGL64,
// ) -> *mut MeshGL64 {
// }

// SDF
// By default, the execution policy (sequential or parallel) of
// manifold_level_set will be chosen automatically depending on the size of the
// job and whether Manifold has been compiled with a PAR backend. If you are
// using these bindings from a language that has a runtime lock preventing the
// parallel execution of closures, then you should use manifold_level_set_seq to
// force sequential execution.

// extern "C" fn manifold_level_set(
// 	mem: *mut ::std::os::raw::c_void,
// 	sdf: ManifoldSdf,
// 	bounds: *mut ManifoldBox,
// 	edge_length: f64,
// 	level: f64,
// 	tolerance: f64,
// 	ctx: *mut ::std::os::raw::c_void,
// ) -> *mut Manifold {
// }
//
// extern "C" fn manifold_level_set_seq(
// 	mem: *mut ::std::os::raw::c_void,
// 	sdf: ManifoldSdf,
// 	bounds: *mut ManifoldBox,
// 	edge_length: f64,
// 	level: f64,
// 	tolerance: f64,
// 	ctx: *mut ::std::os::raw::c_void,
// ) -> *mut Manifold {
// }

// Manifold Vectors

// extern "C" fn manifold_manifold_empty_vec(mem: *mut ::std::os::raw::c_void) -> *mut ManifoldVec {}
//
// extern "C" fn manifold_manifold_vec(
// 	mem: *mut ::std::os::raw::c_void,
// 	sz: usize,
// ) -> *mut ManifoldVec {
// }

// extern "C" fn manifold_manifold_vec_reserve(ms: *mut ManifoldVec, sz: usize) {}
//
// extern "C" fn manifold_manifold_vec_length(ms: *mut ManifoldVec) -> usize {}
//
// extern "C" fn manifold_manifold_vec_get(
// 	mem: *mut ::std::os::raw::c_void,
// 	ms: *mut ManifoldVec,
// 	idx: usize,
// ) -> *mut Manifold {
// }
//
// extern "C" fn manifold_manifold_vec_set(ms: *mut ManifoldVec, idx: usize, m: *mut Manifold) {}
//
// extern "C" fn manifold_manifold_vec_push_back(ms: *mut ManifoldVec, m: *mut Manifold) {}

// Manifold Booleans

// extern "C" fn manifold_boolean(
// 	mem: *mut ::std::os::raw::c_void,
// 	a: *mut Manifold,
// 	b: *mut Manifold,
// 	op: ManifoldOpType,
// ) -> *mut Manifold {
// 	let mem = mem as *mut Manifold;
// 	unsafe {
// 		let a = a.as_ref().expect("Invalid Manifold pointer");
// 		let b = b.as_ref().expect("Invalid Manifold pointer");
//
// 		core::ptr::write(mem, a.boolean(other, op));
// 	}
// 	mem
// }

// extern "C" fn manifold_batch_boolean(
// 	mem: *mut ::std::os::raw::c_void,
// 	ms: *mut ManifoldVec,
// 	op: ManifoldOpType,
// ) -> *mut Manifold {
// }

extern "C" fn manifold_union(
	mem: *mut ::std::os::raw::c_void,
	a: *mut Manifold,
	b: *mut Manifold,
) -> *mut Manifold {
	let mem = mem as *mut Manifold;
	unsafe {
		let a = a.as_ref().expect("Invalid Manifold pointer");
		let b = b.as_ref().expect("Invalid Manifold pointer");

		core::ptr::write(mem, a + b);
	}
	mem
}

extern "C" fn manifold_difference(
	mem: *mut ::std::os::raw::c_void,
	a: *mut Manifold,
	b: *mut Manifold,
) -> *mut Manifold {
	let mem = mem as *mut Manifold;
	unsafe {
		let a = a.as_ref().expect("Invalid Manifold pointer");
		let b = b.as_ref().expect("Invalid Manifold pointer");

		core::ptr::write(mem, a - b);
	}
	mem
}

extern "C" fn manifold_intersection(
	mem: *mut ::std::os::raw::c_void,
	a: *mut Manifold,
	b: *mut Manifold,
) -> *mut Manifold {
	let mem = mem as *mut Manifold;
	unsafe {
		let a = a.as_ref().expect("Invalid Manifold pointer");
		let b = b.as_ref().expect("Invalid Manifold pointer");

		core::ptr::write(mem, a ^ b);
	}
	mem
}

// extern "C" fn manifold_split(
// 	mem_first: *mut ::std::os::raw::c_void,
// 	mem_second: *mut ::std::os::raw::c_void,
// 	a: *mut Manifold,
// 	b: *mut Manifold,
// ) -> ManifoldPair {
// }

// extern "C" fn manifold_split_by_plane(
// 	mem_first: *mut ::std::os::raw::c_void,
// 	mem_second: *mut ::std::os::raw::c_void,
// 	m: *mut Manifold,
// 	normal_x: f64,
// 	normal_y: f64,
// 	normal_z: f64,
// 	offset: f64,
// ) -> ManifoldPair {
// }

// extern "C" fn manifold_trim_by_plane(
// 	mem: *mut ::std::os::raw::c_void,
// 	m: *mut Manifold,
// 	normal_x: f64,
// 	normal_y: f64,
// 	normal_z: f64,
// 	offset: f64,
// ) -> *mut Manifold {
// }

// 3D to 2D

// extern "C" fn manifold_slice(
// 	mem: *mut ::std::os::raw::c_void,
// 	m: *mut Manifold,
// 	height: f64,
// ) -> *mut ManifoldPolygons {
// }

// extern "C" fn manifold_project(
// 	mem: *mut ::std::os::raw::c_void,
// 	m: *mut Manifold,
// ) -> *mut ManifoldPolygons {
// }

// Convex Hulls

// extern "C" fn manifold_hull(mem: *mut ::std::os::raw::c_void, m: *mut Manifold) -> *mut Manifold {}

// extern "C" fn manifold_batch_hull(
// 	mem: *mut ::std::os::raw::c_void,
// 	ms: *mut ManifoldVec,
// ) -> *mut Manifold {
// }

// extern "C" fn manifold_hull_pts(
// 	mem: *mut ::std::os::raw::c_void,
// 	ps: *mut ManifoldVec3,
// 	length: usize,
// ) -> *mut Manifold {
// }

// Manifold Transformations

extern "C" fn manifold_translate(
	mem: *mut ::std::os::raw::c_void,
	m: *mut Manifold,
	x: f64,
	y: f64,
	z: f64,
) -> *mut Manifold {
	let mem = mem as *mut Manifold;
	unsafe {
		let m = m.as_ref().expect("Invalid Manifold pointer");

		core::ptr::write(mem, m.translate([x, y, z].into()));
	}
	mem
}

extern "C" fn manifold_rotate(
	mem: *mut ::std::os::raw::c_void,
	m: *mut Manifold,
	x: f64,
	y: f64,
	z: f64,
) -> *mut Manifold {
	let mem = mem as *mut Manifold;
	unsafe {
		let m = m.as_ref().expect("Invalid Manifold pointer");

		core::ptr::write(mem, m.rotate(x, y, z));
	}
	mem
}

extern "C" fn manifold_scale(
	mem: *mut ::std::os::raw::c_void,
	m: *mut Manifold,
	x: f64,
	y: f64,
	z: f64,
) -> *mut Manifold {
	let mem = mem as *mut Manifold;
	unsafe {
		let m = m.as_ref().expect("Invalid Manifold pointer");

		core::ptr::write(mem, m.scale([x, y, z].into()));
	}
	mem
}

extern "C" fn manifold_transform(
	mem: *mut ::std::os::raw::c_void,
	m: *mut Manifold,
	x1: f64,
	y1: f64,
	z1: f64,
	x2: f64,
	y2: f64,
	z2: f64,
	x3: f64,
	y3: f64,
	z3: f64,
	x4: f64,
	y4: f64,
	z4: f64,
) -> *mut Manifold {
	let mem = mem as *mut Manifold;
	unsafe {
		let m = m.as_ref().expect("Invalid Manifold pointer");

		// core::ptr::write(
		// 	mem,
		// 	m.transform([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3], [x4, y4, z4]].into()),
		// );
		todo!();
	}
	mem
}

// extern "C" fn manifold_mirror(
// 	mem: *mut ::std::os::raw::c_void,
// 	m: *mut Manifold,
// 	nx: f64,
// 	ny: f64,
// 	nz: f64,
// ) -> *mut Manifold {
// 	let mem = mem as *mut Manifold;
// 	unsafe {
// 		let m = m.as_ref().expect("Invalid Manifold pointer");
//
// 		core::ptr::write(mem, m.mirror());
// 	}
// 	mem
// }

// extern "C" fn manifold_warp(
// 	mem: *mut ::std::os::raw::c_void,
// 	m: *mut Manifold,
// 	fun: ::std::option::Option<
// 		extern "C" fn(
// 			arg1: f64,
// 			arg2: f64,
// 			arg3: f64,
// 			arg4: *mut ::std::os::raw::c_void,
// 		) -> ManifoldVec3,
// 	>,
// 	ctx: *mut ::std::os::raw::c_void,
// ) -> *mut Manifold {
// }

// extern "C" fn manifold_smooth_by_normals(
// 	mem: *mut ::std::os::raw::c_void,
// 	m: *mut Manifold,
// 	normalIdx: ::std::os::raw::c_int,
// ) -> *mut Manifold {
// }
//
// extern "C" fn manifold_smooth_out(
// 	mem: *mut ::std::os::raw::c_void,
// 	m: *mut Manifold,
// 	minSharpAngle: f64,
// 	minSmoothness: f64,
// ) -> *mut Manifold {
// }

// extern "C" fn manifold_refine(
// 	mem: *mut ::std::os::raw::c_void,
// 	m: *mut Manifold,
// 	refine: ::std::os::raw::c_int,
// ) -> *mut Manifold {
// }
//
// extern "C" fn manifold_refine_to_length(
// 	mem: *mut ::std::os::raw::c_void,
// 	m: *mut Manifold,
// 	length: f64,
// ) -> *mut Manifold {
// }
//
// extern "C" fn manifold_refine_to_tolerance(
// 	mem: *mut ::std::os::raw::c_void,
// 	m: *mut Manifold,
// 	tolerance: f64,
// ) -> *mut Manifold {
// }

// Manifold Shapes / Constructors

extern "C" fn manifold_empty(mem: *mut ::std::os::raw::c_void) -> *mut Manifold {
	let mem = mem as *mut Manifold;
	unsafe {
		core::ptr::write(mem, Manifold::default());
	}
	mem
}

extern "C" fn manifold_copy(mem: *mut ::std::os::raw::c_void, m: *mut Manifold) -> *mut Manifold {
	let mem = mem as *mut Manifold;
	unsafe {
		core::ptr::copy(m, mem, 1);
	}
	mem
}

// extern "C" fn manifold_tetrahedron(mem: *mut ::std::os::raw::c_void) -> *mut Manifold {
// 	let mem = mem as *mut Manifold;
// 	unsafe {
// 		core::ptr::write(mem, Manifold::from_meshgl(mesh));
// 	}
// 	mem
// }

extern "C" fn manifold_cube(
	mem: *mut ::std::os::raw::c_void,
	x: f64,
	y: f64,
	z: f64,
	center: ::std::os::raw::c_int,
) -> *mut Manifold {
	let mem = mem as *mut Manifold;
	unsafe {
		core::ptr::write(mem, Manifold::cube([x, y, z].into(), center == 1));
	}
	mem
}

extern "C" fn manifold_cylinder(
	mem: *mut ::std::os::raw::c_void,
	height: f64,
	radius_low: f64,
	radius_high: f64,
	circular_segments: ::std::os::raw::c_int,
	center: ::std::os::raw::c_int,
) -> *mut Manifold {
	let mem = mem as *mut Manifold;
	unsafe {
		core::ptr::write(
			mem,
			Manifold::cylinder(
				height,
				radius_low,
				radius_high,
				circular_segments as u32,
				center == 1,
			),
		);
	}
	mem
}

// extern "C" fn manifold_sphere(
// 	mem: *mut ::std::os::raw::c_void,
// 	radius: f64,
// 	circular_segments: ::std::os::raw::c_int,
// ) -> *mut Manifold {
// }

extern "C" fn manifold_of_meshgl(
	mem: *mut ::std::os::raw::c_void,
	mesh: *mut MeshGL,
) -> *mut Manifold {
	let mem = mem as *mut Manifold;
	unsafe {
		let mesh = mesh.as_ref().expect("Invalid MeshGL pointer");

		core::ptr::write(mem, Manifold::from_meshgl(mesh));
	}
	mem
}

extern "C" fn manifold_of_meshgl64(
	mem: *mut ::std::os::raw::c_void,
	mesh: *mut MeshGL64,
) -> *mut Manifold {
	let mem = mem as *mut Manifold;
	unsafe {
		let mesh = mesh.as_ref().expect("Invalid MeshGL64 pointer");

		core::ptr::write(mem, Manifold::from_meshgl(mesh));
	}
	mem
}

// extern "C" fn manifold_smooth(
// 	mem: *mut ::std::os::raw::c_void,
// 	mesh: *mut MeshGL,
// 	half_edges: *mut usize,
// 	smoothness: *mut f64,
// 	n_idxs: usize,
// ) -> *mut Manifold {
// }

// extern "C" fn manifold_smooth64(
// 	mem: *mut ::std::os::raw::c_void,
// 	mesh: *mut MeshGL64,
// 	half_edges: *mut usize,
// 	smoothness: *mut f64,
// 	n_idxs: usize,
// ) -> *mut Manifold {
// }

// extern "C" fn manifold_extrude(
// 	mem: *mut ::std::os::raw::c_void,
// 	cs: *mut ManifoldPolygons,
// 	height: f64,
// 	slices: ::std::os::raw::c_int,
// 	twist_degrees: f64,
// 	scale_x: f64,
// 	scale_y: f64,
// ) -> *mut Manifold {
// }

// extern "C" fn manifold_revolve(
// 	mem: *mut ::std::os::raw::c_void,
// 	cs: *mut ManifoldPolygons,
// 	circular_segments: ::std::os::raw::c_int,
// 	revolve_degrees: f64,
// ) -> *mut Manifold {
// }

// extern "C" fn manifold_compose(
// 	mem: *mut ::std::os::raw::c_void,
// 	ms: *mut ManifoldVec,
// ) -> *mut Manifold {
// }

// extern "C" fn manifold_decompose(
// 	mem: *mut ::std::os::raw::c_void,
// 	m: *mut Manifold,
// ) -> *mut ManifoldVec {
// }

extern "C" fn manifold_as_original(
	mem: *mut ::std::os::raw::c_void,
	m: *mut Manifold,
) -> *mut Manifold {
	let mem = mem as *mut Manifold;
	unsafe {
		let m = m.as_ref().expect("Invalid Manifold pointer");

		core::ptr::write(mem, m.as_original());
	}
	mem
}

// Manifold Info

extern "C" fn manifold_is_empty(m: *mut Manifold) -> std::os::raw::c_int {
	unsafe {
		let m = m.as_ref().expect("Invalid pointer to Manifold");
		m.is_empty().into()
	}
}

// extern "C" fn manifold_status(m: *mut Manifold) -> ManifoldError {
// 	unsafe {
// 		let m = m.as_ref().expect("Invalid pointer to Manifold");
// 		m.status()
// 	}
// }

extern "C" fn manifold_num_vert(m: *mut Manifold) -> usize {
	unsafe {
		let m = m.as_ref().expect("Invalid pointer to Manifold");
		m.num_vert()
	}
}

extern "C" fn manifold_num_edge(m: *mut Manifold) -> usize {
	unsafe {
		let m = m.as_ref().expect("Invalid pointer to Manifold");
		m.num_edge()
	}
}

extern "C" fn manifold_num_tri(m: *mut Manifold) -> usize {
	unsafe {
		let m = m.as_ref().expect("Invalid pointer to Manifold");
		m.num_tri()
	}
}

extern "C" fn manifold_num_prop(m: *mut Manifold) -> usize {
	unsafe {
		let m = m.as_ref().expect("Invalid pointer to Manifold");
		m.num_prop()
	}
}

// extern "C" fn manifold_bounding_box(
// 	mem: *mut ::std::os::raw::c_void,
// 	m: *mut Manifold,
// ) -> *mut ManifoldBox {
// 	unsafe {
// 		let m = m.as_ref().expect("Invalid pointer to Manifold");
// 		m.bounding_box()
// 	}
// }

extern "C" fn manifold_epsilon(m: *mut Manifold) -> f64 {
	unsafe {
		let m = m.as_ref().expect("Invalid pointer to Manifold");
		m.get_epsilon()
	}
}

// extern "C" fn manifold_genus(m: *mut Manifold) -> ::std::os::raw::c_int {
// 	unsafe {
// 		let m = m.as_ref().expect("Invalid pointer to Manifold");
// 		m.num_prop() as c_int
// 	}
// }

// extern "C" fn manifold_surface_area(m: *mut Manifold) -> f64 {
// 	unsafe {
// 		let m = m.as_ref().expect("Invalid pointer to Manifold");
// 		m.sur()
// 	}
// }

// extern "C" fn manifold_volume(m: *mut Manifold) -> f64 {
// 	unsafe {
// 		let m = m.as_ref().expect("Invalid pointer to Manifold");
// 		m.num_prop()
// 	}
// }

// extern "C" fn manifold_get_circular_segments(radius: f64) -> std::os::raw::c_int {
// 	// unsafe {
// 	// 	let m = m.as_ref().expect("Invalid pointer to Manifold");
// 	// 	m.num_prop()
// 	// }
// }

// extern "C" fn manifold_original_id(m: *mut Manifold) -> std::os::raw::c_int {
// 	unsafe {
// 		let m = m.as_ref().expect("Invalid pointer to Manifold");
// 		m.num_prop()
// 	}
// }

// extern "C" fn manifold_reserve_ids(n: u32) -> u32 {}

// unsafe extern "C" fn manifold_set_properties(
// 	mem: *mut ::std::os::raw::c_void,
// 	m: *mut Manifold,
// 	num_prop: ::std::os::raw::c_int,
// 	fun: ::std::option::Option<
// 		extern "C" fn(
// 			new_prop: *mut f64,
// 			position: ManifoldVec3,
// 			old_prop: *const f64,
// 			ctx: *mut ::std::os::raw::c_void,
// 		),
// 	>,
// 	ctx: *mut ::std::os::raw::c_void,
// ) -> *mut Manifold {
// }

// extern "C" fn manifold_calculate_curvature(
// 	mem: *mut ::std::os::raw::c_void,
// 	m: *mut Manifold,
// 	gaussian_idx: ::std::os::raw::c_int,
// 	mean_idx: ::std::os::raw::c_int,
// ) -> *mut Manifold {
// }
//
// extern "C" fn manifold_min_gap(m: *mut Manifold, other: *mut Manifold, searchLength: f64) -> f64 {}
//
// extern "C" fn manifold_calculate_normals(
// 	mem: *mut ::std::os::raw::c_void,
// 	m: *mut Manifold,
// 	normal_idx: ::std::os::raw::c_int,
// 	min_sharp_angle: f64,
// ) -> *mut Manifold {
// }
//
// // CrossSection Shapes/Constructors
//
// extern "C" fn manifold_cross_section_empty(
// 	mem: *mut ::std::os::raw::c_void,
// ) -> *mut ManifoldCrossSection {
// }
//
// extern "C" fn manifold_cross_section_copy(
// 	mem: *mut ::std::os::raw::c_void,
// 	cs: *mut ManifoldCrossSection,
// ) -> *mut ManifoldCrossSection {
// }
//
// extern "C" fn manifold_cross_section_of_simple_polygon(
// 	mem: *mut ::std::os::raw::c_void,
// 	p: *mut ManifoldSimplePolygon,
// 	fr: ManifoldFillRule,
// ) -> *mut ManifoldCrossSection {
// }
//
// extern "C" fn manifold_cross_section_of_polygons(
// 	mem: *mut ::std::os::raw::c_void,
// 	p: *mut ManifoldPolygons,
// 	fr: ManifoldFillRule,
// ) -> *mut ManifoldCrossSection {
// }
//
// extern "C" fn manifold_cross_section_square(
// 	mem: *mut ::std::os::raw::c_void,
// 	x: f64,
// 	y: f64,
// 	center: ::std::os::raw::c_int,
// ) -> *mut ManifoldCrossSection {
// }
//
// extern "C" fn manifold_cross_section_circle(
// 	mem: *mut ::std::os::raw::c_void,
// 	radius: f64,
// 	circular_segments: ::std::os::raw::c_int,
// ) -> *mut ManifoldCrossSection {
// }
//
// extern "C" fn manifold_cross_section_compose(
// 	mem: *mut ::std::os::raw::c_void,
// 	csv: *mut ManifoldCrossSectionVec,
// ) -> *mut ManifoldCrossSection {
// }
//
// extern "C" fn manifold_cross_section_decompose(
// 	mem: *mut ::std::os::raw::c_void,
// 	cs: *mut ManifoldCrossSection,
// ) -> *mut ManifoldCrossSectionVec {
// }

// CrossSection Vectors

// extern "C" fn manifold_cross_section_empty_vec(
// 	mem: *mut ::std::os::raw::c_void,
// ) -> *mut ManifoldCrossSectionVec {
// }
//
// extern "C" fn manifold_cross_section_vec(
// 	mem: *mut ::std::os::raw::c_void,
// 	sz: usize,
// ) -> *mut ManifoldCrossSectionVec {
// }
//
// extern "C" fn manifold_cross_section_vec_reserve(csv: *mut ManifoldCrossSectionVec, sz: usize) {}
//
// extern "C" fn manifold_cross_section_vec_length(csv: *mut ManifoldCrossSectionVec) -> usize {}
//
// extern "C" fn manifold_cross_section_vec_get(
// 	mem: *mut ::std::os::raw::c_void,
// 	csv: *mut ManifoldCrossSectionVec,
// 	idx: usize,
// ) -> *mut ManifoldCrossSection {
// }
//
// extern "C" fn manifold_cross_section_vec_set(
// 	csv: *mut ManifoldCrossSectionVec,
// 	idx: usize,
// 	cs: *mut ManifoldCrossSection,
// ) {
// }
//
// extern "C" fn manifold_cross_section_vec_push_back(
// 	csv: *mut ManifoldCrossSectionVec,
// 	cs: *mut ManifoldCrossSection,
// ) {
// }

// CrossSection Booleans

// extern "C" fn manifold_cross_section_boolean(
// 	mem: *mut ::std::os::raw::c_void,
// 	a: *mut ManifoldCrossSection,
// 	b: *mut ManifoldCrossSection,
// 	op: ManifoldOpType,
// ) -> *mut ManifoldCrossSection {
// }
//
// extern "C" fn manifold_cross_section_batch_boolean(
// 	mem: *mut ::std::os::raw::c_void,
// 	csv: *mut ManifoldCrossSectionVec,
// 	op: ManifoldOpType,
// ) -> *mut ManifoldCrossSection {
// }
//
// extern "C" fn manifold_cross_section_union(
// 	mem: *mut ::std::os::raw::c_void,
// 	a: *mut ManifoldCrossSection,
// 	b: *mut ManifoldCrossSection,
// ) -> *mut ManifoldCrossSection {
// }
//
// extern "C" fn manifold_cross_section_difference(
// 	mem: *mut ::std::os::raw::c_void,
// 	a: *mut ManifoldCrossSection,
// 	b: *mut ManifoldCrossSection,
// ) -> *mut ManifoldCrossSection {
// }
//
// extern "C" fn manifold_cross_section_intersection(
// 	mem: *mut ::std::os::raw::c_void,
// 	a: *mut ManifoldCrossSection,
// 	b: *mut ManifoldCrossSection,
// ) -> *mut ManifoldCrossSection {
// }

// CrossSection Convex Hulls

// extern "C" fn manifold_cross_section_hull(
// 	mem: *mut ::std::os::raw::c_void,
// 	cs: *mut ManifoldCrossSection,
// ) -> *mut ManifoldCrossSection {
// }
//
// extern "C" fn manifold_cross_section_batch_hull(
// 	mem: *mut ::std::os::raw::c_void,
// 	css: *mut ManifoldCrossSectionVec,
// ) -> *mut ManifoldCrossSection {
// }
//
// extern "C" fn manifold_cross_section_hull_simple_polygon(
// 	mem: *mut ::std::os::raw::c_void,
// 	ps: *mut ManifoldSimplePolygon,
// ) -> *mut ManifoldCrossSection {
// }
//
// extern "C" fn manifold_cross_section_hull_polygons(
// 	mem: *mut ::std::os::raw::c_void,
// 	ps: *mut ManifoldPolygons,
// ) -> *mut ManifoldCrossSection {
// }

// CrossSection Transformation

// extern "C" fn manifold_cross_section_translate(
// 	mem: *mut ::std::os::raw::c_void,
// 	cs: *mut ManifoldCrossSection,
// 	x: f64,
// 	y: f64,
// ) -> *mut ManifoldCrossSection {
// }
//
// extern "C" fn manifold_cross_section_rotate(
// 	mem: *mut ::std::os::raw::c_void,
// 	cs: *mut ManifoldCrossSection,
// 	deg: f64,
// ) -> *mut ManifoldCrossSection {
// }
//
// extern "C" fn manifold_cross_section_scale(
// 	mem: *mut ::std::os::raw::c_void,
// 	cs: *mut ManifoldCrossSection,
// 	x: f64,
// 	y: f64,
// ) -> *mut ManifoldCrossSection {
// }
//
// extern "C" fn manifold_cross_section_mirror(
// 	mem: *mut ::std::os::raw::c_void,
// 	cs: *mut ManifoldCrossSection,
// 	ax_x: f64,
// 	ax_y: f64,
// ) -> *mut ManifoldCrossSection {
// }
//
// extern "C" fn manifold_cross_section_transform(
// 	mem: *mut ::std::os::raw::c_void,
// 	cs: *mut ManifoldCrossSection,
// 	x1: f64,
// 	y1: f64,
// 	x2: f64,
// 	y2: f64,
// 	x3: f64,
// 	y3: f64,
// ) -> *mut ManifoldCrossSection {
// }
//
// unsafe extern "C" fn manifold_cross_section_warp(
// 	mem: *mut ::std::os::raw::c_void,
// 	cs: *mut ManifoldCrossSection,
// 	fun: ::std::option::Option<extern "C" fn(arg1: f64, arg2: f64) -> ManifoldVec2>,
// ) -> *mut ManifoldCrossSection {
// }
//
// unsafe extern "C" fn manifold_cross_section_warp_context(
// 	mem: *mut ::std::os::raw::c_void,
// 	cs: *mut ManifoldCrossSection,
// 	fun: ::std::option::Option<
// 		extern "C" fn(arg1: f64, arg2: f64, arg3: *mut ::std::os::raw::c_void) -> ManifoldVec2,
// 	>,
// 	ctx: *mut ::std::os::raw::c_void,
// ) -> *mut ManifoldCrossSection {
// }
//
// extern "C" fn manifold_cross_section_simplify(
// 	mem: *mut ::std::os::raw::c_void,
// 	cs: *mut ManifoldCrossSection,
// 	epsilon: f64,
// ) -> *mut ManifoldCrossSection {
// }
//
// extern "C" fn manifold_cross_section_offset(
// 	mem: *mut ::std::os::raw::c_void,
// 	cs: *mut ManifoldCrossSection,
// 	delta: f64,
// 	jt: ManifoldJoinType,
// 	miter_limit: f64,
// 	circular_segments: ::std::os::raw::c_int,
// ) -> *mut ManifoldCrossSection {
// }

// CrossSection Info

// extern "C" fn manifold_cross_section_area(cs: *mut ManifoldCrossSection) -> f64 {}
//
// extern "C" fn manifold_cross_section_num_vert(cs: *mut ManifoldCrossSection) -> usize {}
//
// extern "C" fn manifold_cross_section_num_contour(cs: *mut ManifoldCrossSection) -> usize {}

// extern "C" fn manifold_cross_section_is_empty(
// 	cs: *mut ManifoldCrossSection,
// ) -> ::std::os::raw::c_int {
// }
//
// extern "C" fn manifold_cross_section_bounds(
// 	mem: *mut ::std::os::raw::c_void,
// 	cs: *mut ManifoldCrossSection,
// ) -> *mut ManifoldRect {
// }
//
// extern "C" fn manifold_cross_section_to_polygons(
// 	mem: *mut ::std::os::raw::c_void,
// 	cs: *mut ManifoldCrossSection,
// ) -> *mut ManifoldPolygons {
// }

// Rectangle

// extern "C" fn manifold_rect(
// 	mem: *mut ::std::os::raw::c_void,
// 	x1: f64,
// 	y1: f64,
// 	x2: f64,
// 	y2: f64,
// ) -> *mut ManifoldRect {
// }
//
// extern "C" fn manifold_rect_min(r: *mut ManifoldRect) -> ManifoldVec2 {}
//
// extern "C" fn manifold_rect_max(r: *mut ManifoldRect) -> ManifoldVec2 {}
//
// extern "C" fn manifold_rect_dimensions(r: *mut ManifoldRect) -> ManifoldVec2 {}
//
// extern "C" fn manifold_rect_center(r: *mut ManifoldRect) -> ManifoldVec2 {}
//
// extern "C" fn manifold_rect_scale(r: *mut ManifoldRect) -> f64 {}
//
// extern "C" fn manifold_rect_contains_pt(
// 	r: *mut ManifoldRect,
// 	x: f64,
// 	y: f64,
// ) -> ::std::os::raw::c_int {
// }
//
// extern "C" fn manifold_rect_contains_rect(
// 	a: *mut ManifoldRect,
// 	b: *mut ManifoldRect,
// ) -> ::std::os::raw::c_int {
// }
//
// extern "C" fn manifold_rect_include_pt(r: *mut ManifoldRect, x: f64, y: f64) {}
//
// extern "C" fn manifold_rect_union(
// 	mem: *mut ::std::os::raw::c_void,
// 	a: *mut ManifoldRect,
// 	b: *mut ManifoldRect,
// ) -> *mut ManifoldRect {
// }
//
// extern "C" fn manifold_rect_transform(
// 	mem: *mut ::std::os::raw::c_void,
// 	r: *mut ManifoldRect,
// 	x1: f64,
// 	y1: f64,
// 	x2: f64,
// 	y2: f64,
// 	x3: f64,
// 	y3: f64,
// ) -> *mut ManifoldRect {
// }
//
// extern "C" fn manifold_rect_translate(
// 	mem: *mut ::std::os::raw::c_void,
// 	r: *mut ManifoldRect,
// 	x: f64,
// 	y: f64,
// ) -> *mut ManifoldRect {
// }
//
// extern "C" fn manifold_rect_mul(
// 	mem: *mut ::std::os::raw::c_void,
// 	r: *mut ManifoldRect,
// 	x: f64,
// 	y: f64,
// ) -> *mut ManifoldRect {
// }
//
// extern "C" fn manifold_rect_does_overlap_rect(
// 	a: *mut ManifoldRect,
// 	r: *mut ManifoldRect,
// ) -> ::std::os::raw::c_int {
// }
//
// extern "C" fn manifold_rect_is_empty(r: *mut ManifoldRect) -> ::std::os::raw::c_int {}
//
// extern "C" fn manifold_rect_is_finite(r: *mut ManifoldRect) -> ::std::os::raw::c_int {}

// Bounding Box

// extern "C" fn manifold_box(
// 	mem: *mut ::std::os::raw::c_void,
// 	x1: f64,
// 	y1: f64,
// 	z1: f64,
// 	x2: f64,
// 	y2: f64,
// 	z2: f64,
// ) -> *mut ManifoldBox {
// }
//
// extern "C" fn manifold_box_min(b: *mut ManifoldBox) -> ManifoldVec3 {}
//
// extern "C" fn manifold_box_max(b: *mut ManifoldBox) -> ManifoldVec3 {}
//
// extern "C" fn manifold_box_dimensions(b: *mut ManifoldBox) -> ManifoldVec3 {}
//
// extern "C" fn manifold_box_center(b: *mut ManifoldBox) -> ManifoldVec3 {}
//
// extern "C" fn manifold_box_scale(b: *mut ManifoldBox) -> f64 {}
//
// extern "C" fn manifold_box_contains_pt(
// 	b: *mut ManifoldBox,
// 	x: f64,
// 	y: f64,
// 	z: f64,
// ) -> ::std::os::raw::c_int {
// }
//
// extern "C" fn manifold_box_contains_box(
// 	a: *mut ManifoldBox,
// 	b: *mut ManifoldBox,
// ) -> ::std::os::raw::c_int {
// }
//
// extern "C" fn manifold_box_include_pt(b: *mut ManifoldBox, x: f64, y: f64, z: f64) {}
//
// extern "C" fn manifold_box_union(
// 	mem: *mut ::std::os::raw::c_void,
// 	a: *mut ManifoldBox,
// 	b: *mut ManifoldBox,
// ) -> *mut ManifoldBox {
// }
//
// extern "C" fn manifold_box_transform(
// 	mem: *mut ::std::os::raw::c_void,
// 	b: *mut ManifoldBox,
// 	x1: f64,
// 	y1: f64,
// 	z1: f64,
// 	x2: f64,
// 	y2: f64,
// 	z2: f64,
// 	x3: f64,
// 	y3: f64,
// 	z3: f64,
// 	x4: f64,
// 	y4: f64,
// 	z4: f64,
// ) -> *mut ManifoldBox {
// }
//
// extern "C" fn manifold_box_translate(
// 	mem: *mut ::std::os::raw::c_void,
// 	b: *mut ManifoldBox,
// 	x: f64,
// 	y: f64,
// 	z: f64,
// ) -> *mut ManifoldBox {
// }
//
// extern "C" fn manifold_box_mul(
// 	mem: *mut ::std::os::raw::c_void,
// 	b: *mut ManifoldBox,
// 	x: f64,
// 	y: f64,
// 	z: f64,
// ) -> *mut ManifoldBox {
// }
//
// extern "C" fn manifold_box_does_overlap_pt(
// 	b: *mut ManifoldBox,
// 	x: f64,
// 	y: f64,
// 	z: f64,
// ) -> ::std::os::raw::c_int {
// }
//
// extern "C" fn manifold_box_does_overlap_box(
// 	a: *mut ManifoldBox,
// 	b: *mut ManifoldBox,
// ) -> ::std::os::raw::c_int {
// }
//
// extern "C" fn manifold_box_is_finite(b: *mut ManifoldBox) -> ::std::os::raw::c_int {}

// Static Quality Globals

// extern "C" fn manifold_set_min_circular_angle(degrees: f64) {}
//
// extern "C" fn manifold_set_min_circular_edge_length(length: f64) {}
//
// extern "C" fn manifold_set_circular_segments(number: ::std::os::raw::c_int) {}
//
// extern "C" fn manifold_reset_to_circular_defaults() {}

// Manifold Mesh Extraction

extern "C" fn manifold_meshgl_num_prop(m: *mut MeshGL) -> std::os::raw::c_int {
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL pointer");
		m.num_prop as c_int
	}
}

extern "C" fn manifold_meshgl_num_vert(m: *mut MeshGL) -> std::os::raw::c_int {
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL pointer");
		m.num_vert() as c_int
	}
}

extern "C" fn manifold_meshgl_num_tri(m: *mut MeshGL) -> std::os::raw::c_int {
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL pointer");
		m.num_tri() as c_int
	}
}

extern "C" fn manifold_meshgl_vert_properties_length(m: *mut MeshGL) -> usize {
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL pointer");
		m.vert_properties.len()
	}
}

extern "C" fn manifold_meshgl_tri_length(m: *mut MeshGL) -> usize {
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL pointer");
		m.tri_verts.len()
	}
}

extern "C" fn manifold_meshgl_merge_length(m: *mut MeshGL) -> usize {
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL pointer");
		m.merge_from_vert.len()
	}
}

extern "C" fn manifold_meshgl_run_index_length(m: *mut MeshGL) -> usize {
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL pointer");
		m.run_index.len()
	}
}

extern "C" fn manifold_meshgl_run_original_id_length(m: *mut MeshGL) -> usize {
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL pointer");
		m.run_original_id.len()
	}
}

extern "C" fn manifold_meshgl_run_transform_length(m: *mut MeshGL) -> usize {
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL pointer");
		m.run_transform.len()
	}
}

extern "C" fn manifold_meshgl_face_id_length(m: *mut MeshGL) -> usize {
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL pointer");
		m.face_id.len()
	}
}

// extern "C" fn manifold_meshgl_tangent_length(m: *mut MeshGL) -> usize {
// 	unsafe {
// 		let m = m.as_ref().expect("Invalid MeshGL pointer");
// 	}
// }

extern "C" fn manifold_meshgl_vert_properties(
	mem: *mut ::std::os::raw::c_void,
	m: *mut MeshGL,
) -> *mut f32 {
	let mem = mem as *mut f32;
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL pointer");
		core::ptr::copy(m.vert_properties.as_ptr(), mem, m.vert_properties.len());
	}
	mem
}

extern "C" fn manifold_meshgl_tri_verts(
	mem: *mut ::std::os::raw::c_void,
	m: *mut MeshGL,
) -> *mut u32 {
	let mem = mem as *mut u32;
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL pointer");
		core::ptr::copy(m.tri_verts.as_ptr(), mem, m.tri_verts.len());
	}
	mem
}

extern "C" fn manifold_meshgl_merge_from_vert(
	mem: *mut ::std::os::raw::c_void,
	m: *mut MeshGL,
) -> *mut u32 {
	let mem = mem as *mut u32;
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL pointer");
		core::ptr::copy(m.merge_from_vert.as_ptr(), mem, m.merge_from_vert.len());
	}
	mem
}

extern "C" fn manifold_meshgl_merge_to_vert(
	mem: *mut ::std::os::raw::c_void,
	m: *mut MeshGL,
) -> *mut u32 {
	let mem = mem as *mut u32;
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL pointer");
		core::ptr::copy(m.merge_to_vert.as_ptr(), mem, m.merge_to_vert.len());
	}
	mem
}

extern "C" fn manifold_meshgl_run_index(
	mem: *mut ::std::os::raw::c_void,
	m: *mut MeshGL,
) -> *mut u32 {
	let mem = mem as *mut u32;
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL pointer");
		core::ptr::copy(m.run_index.as_ptr(), mem, m.run_index.len());
	}
	mem
}

extern "C" fn manifold_meshgl_run_original_id(
	mem: *mut ::std::os::raw::c_void,
	m: *mut MeshGL,
) -> *mut u32 {
	let mem = mem as *mut u32;
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL pointer");
		core::ptr::copy(m.run_original_id.as_ptr(), mem, m.run_original_id.len());
	}
	mem
}

extern "C" fn manifold_meshgl_run_transform(
	mem: *mut ::std::os::raw::c_void,
	m: *mut MeshGL,
) -> *mut f32 {
	let mem = mem as *mut f32;
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL pointer");
		core::ptr::copy(m.run_transform.as_ptr(), mem, m.run_transform.len());
	}
	mem
}

extern "C" fn manifold_meshgl_face_id(
	mem: *mut ::std::os::raw::c_void,
	m: *mut MeshGL,
) -> *mut u32 {
	let mem = mem as *mut u32;
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL pointer");
		core::ptr::copy(m.face_id.as_ptr(), mem, m.face_id.len());
	}
	mem
}

// extern "C" fn manifold_meshgl_halfedge_tangent(
// 	mem: *mut ::std::os::raw::c_void,
// 	m: *mut MeshGL,
// ) -> *mut f32 {
// }

extern "C" fn manifold_meshgl64_num_prop(m: *mut MeshGL64) -> usize {
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL64 pointer");
		m.num_prop as usize
	}
}

extern "C" fn manifold_meshgl64_num_vert(m: *mut MeshGL64) -> usize {
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL64 pointer");
		m.num_vert()
	}
}

extern "C" fn manifold_meshgl64_num_tri(m: *mut MeshGL64) -> usize {
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL64 pointer");
		m.num_tri()
	}
}

extern "C" fn manifold_meshgl64_vert_properties_length(m: *mut MeshGL64) -> usize {
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL64 pointer");
		m.vert_properties.len()
	}
}

extern "C" fn manifold_meshgl64_tri_length(m: *mut MeshGL64) -> usize {
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL64 pointer");
		m.tri_verts.len()
	}
}

extern "C" fn manifold_meshgl64_merge_length(m: *mut MeshGL64) -> usize {
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL64 pointer");
		m.merge_from_vert.len()
	}
}

extern "C" fn manifold_meshgl64_run_index_length(m: *mut MeshGL64) -> usize {
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL64 pointer");
		m.run_index.len()
	}
}

extern "C" fn manifold_meshgl64_run_original_id_length(m: *mut MeshGL64) -> usize {
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL64 pointer");
		m.run_original_id.len()
	}
}

extern "C" fn manifold_meshgl64_run_transform_length(m: *mut MeshGL64) -> usize {
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL64 pointer");
		m.run_transform.len()
	}
}

extern "C" fn manifold_meshgl64_face_id_length(m: *mut MeshGL64) -> usize {
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL64 pointer");
		m.face_id.len()
	}
}

// extern "C" fn manifold_meshgl64_tangent_length(m: *mut MeshGL64) -> usize {
// 	unsafe {
// 		let m = m.as_ref().expect("Invalid MeshGL64 pointer");
// 		m.num_prop()
// 	}
// }

extern "C" fn manifold_meshgl64_vert_properties(
	mem: *mut ::std::os::raw::c_void,
	m: *mut MeshGL64,
) -> *mut f64 {
	let mem = mem as *mut f64;
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL64 pointer");
		core::ptr::copy(
			m.vert_properties
				.iter()
				.map(|f| *f as f64)
				.collect::<Vec<f64>>()
				.as_ptr(),
			mem,
			m.vert_properties.len(),
		);
	}
	mem
}

extern "C" fn manifold_meshgl64_tri_verts(
	mem: *mut ::std::os::raw::c_void,
	m: *mut MeshGL64,
) -> *mut u64 {
	let mem = mem as *mut u64;
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL64 pointer");
		core::ptr::copy(
			m.tri_verts
				.iter()
				.map(|f| *f as u64)
				.collect::<Vec<u64>>()
				.as_ptr(),
			mem,
			m.tri_verts.len(),
		);
	}
	mem
}

extern "C" fn manifold_meshgl64_merge_from_vert(
	mem: *mut ::std::os::raw::c_void,
	m: *mut MeshGL64,
) -> *mut u64 {
	let mem = mem as *mut u64;
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL64 pointer");
		core::ptr::copy(
			m.merge_from_vert
				.iter()
				.map(|f| *f as u64)
				.collect::<Vec<u64>>()
				.as_ptr(),
			mem,
			m.merge_from_vert.len(),
		);
	}
	mem
}

extern "C" fn manifold_meshgl64_merge_to_vert(
	mem: *mut ::std::os::raw::c_void,
	m: *mut MeshGL64,
) -> *mut u64 {
	let mem = mem as *mut u64;
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL64 pointer");
		core::ptr::copy(
			m.merge_to_vert
				.iter()
				.map(|f| *f as u64)
				.collect::<Vec<u64>>()
				.as_ptr(),
			mem,
			m.merge_to_vert.len(),
		);
	}
	mem
}

extern "C" fn manifold_meshgl64_run_index(
	mem: *mut ::std::os::raw::c_void,
	m: *mut MeshGL64,
) -> *mut u64 {
	let mem = mem as *mut u64;
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL64 pointer");
		core::ptr::copy(
			m.run_index
				.iter()
				.map(|f| *f as u64)
				.collect::<Vec<u64>>()
				.as_ptr(),
			mem,
			m.run_index.len(),
		);
	}
	mem
}

extern "C" fn manifold_meshgl64_run_original_id(
	mem: *mut ::std::os::raw::c_void,
	m: *mut MeshGL64,
) -> *mut u32 {
	let mem = mem as *mut u32;
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL64 pointer");
		core::ptr::copy(m.run_original_id.as_ptr(), mem, m.run_original_id.len());
	}
	mem
}

extern "C" fn manifold_meshgl64_run_transform(
	mem: *mut ::std::os::raw::c_void,
	m: *mut MeshGL64,
) -> *mut f64 {
	let mem = mem as *mut f64;
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL64 pointer");
		core::ptr::copy(
			m.run_transform
				.iter()
				.map(|f| *f as f64)
				.collect::<Vec<f64>>()
				.as_ptr(),
			mem,
			m.run_transform.len(),
		);
	}
	mem
}

extern "C" fn manifold_meshgl64_face_id(
	mem: *mut ::std::os::raw::c_void,
	m: *mut MeshGL64,
) -> *mut u64 {
	let mem = mem as *mut u64;
	unsafe {
		let m = m.as_ref().expect("Invalid MeshGL64 pointer");
		core::ptr::copy(
			m.face_id
				.iter()
				.map(|f| *f as u64)
				.collect::<Vec<u64>>()
				.as_ptr(),
			mem,
			m.face_id.len(),
		);
	}
	mem
}

// extern "C" fn manifold_meshgl64_halfedge_tangent(
// 	mem: *mut ::std::os::raw::c_void,
// 	m: *mut MeshGL64,
// ) -> *mut f64 {
// }

// Triangulation

// extern "C" fn manifold_triangulate(
// 	mem: *mut ::std::os::raw::c_void,
// 	ps: *mut ManifoldPolygons,
// 	epsilon: f64,
// ) -> *mut ManifoldTriangulation {
// }

// extern "C" fn manifold_triangulation_num_tri(m: *mut ManifoldTriangulation) -> usize {}

// extern "C" fn manifold_triangulation_tri_verts(
// 	mem: *mut ::std::os::raw::c_void,
// 	m: *mut ManifoldTriangulation,
// ) -> *mut ::std::os::raw::c_int {
// }

// memory size

extern "C" fn manifold_manifold_size() -> usize {
	core::mem::size_of::<Manifold>()
}

// extern "C" fn manifold_manifold_vec_size() -> usize {}
//
// extern "C" fn manifold_cross_section_size() -> usize {}
//
// extern "C" fn manifold_cross_section_vec_size() -> usize {}
//
// extern "C" fn manifold_simple_polygon_size() -> usize {}
//
// extern "C" fn manifold_polygons_size() -> usize {}
//
// extern "C" fn manifold_manifold_pair_size() -> usize {}

extern "C" fn manifold_meshgl_size() -> usize {
	core::mem::size_of::<MeshGL>()
}

extern "C" fn manifold_meshgl64_size() -> usize {
	core::mem::size_of::<MeshGL64>()
}

// extern "C" fn manifold_box_size() -> usize {}
//
// extern "C" fn manifold_rect_size() -> usize {}
//
// extern "C" fn manifold_curvature_size() -> usize {}
//
// extern "C" fn manifold_triangulation_size() -> usize {}

// allocation

extern "C" fn manifold_alloc_manifold() -> *mut Manifold {
	Box::into_raw(Box::new(Manifold::default()))
}

// extern "C" fn manifold_alloc_manifold_vec() -> *mut ManifoldVec {}
//
// extern "C" fn manifold_alloc_cross_section() -> *mut ManifoldCrossSection {}
//
// extern "C" fn manifold_alloc_cross_section_vec() -> *mut ManifoldCrossSectionVec {}
//
// extern "C" fn manifold_alloc_simple_polygon() -> *mut ManifoldSimplePolygon {}
//
// extern "C" fn manifold_alloc_polygons() -> *mut ManifoldPolygons {}

extern "C" fn manifold_alloc_meshgl() -> *mut MeshGL {
	Box::into_raw(Box::new(MeshGL::default()))
}

extern "C" fn manifold_alloc_meshgl64() -> *mut MeshGL64 {
	Box::into_raw(Box::new(MeshGL64::default()))
}

// extern "C" fn manifold_alloc_box() -> *mut ManifoldBox {}
//
// extern "C" fn manifold_alloc_rect() -> *mut ManifoldRect {}

// extern "C" fn manifold_alloc_triangulation() -> *mut ManifoldTriangulation {}

// destruction

extern "C" fn manifold_destruct_manifold(m: *mut Manifold) {
	unsafe {
		core::ptr::drop_in_place(m);
	}
}

// extern "C" fn manifold_destruct_manifold_vec(ms: *mut ManifoldVec) {}
//
// extern "C" fn manifold_destruct_cross_section(m: *mut ManifoldCrossSection) {}
//
// extern "C" fn manifold_destruct_cross_section_vec(csv: *mut ManifoldCrossSectionVec) {}
//
// extern "C" fn manifold_destruct_simple_polygon(p: *mut ManifoldSimplePolygon) {}
//
// extern "C" fn manifold_destruct_polygons(p: *mut ManifoldPolygons) {}
//
extern "C" fn manifold_destruct_meshgl(m: *mut MeshGL) {
	unsafe {
		core::ptr::drop_in_place(m);
	}
}

extern "C" fn manifold_destruct_meshgl64(m: *mut MeshGL64) {
	unsafe {
		core::ptr::drop_in_place(m);
	}
}

// extern "C" fn manifold_destruct_box(b: *mut ManifoldBox) {}
//
// extern "C" fn manifold_destruct_rect(b: *mut ManifoldRect) {}
//
// extern "C" fn manifold_destruct_triangulation(M: *mut ManifoldTriangulation) {}

// pointer free + destruction

extern "C" fn manifold_delete_manifold(m: *mut Manifold) {
	unsafe {
		drop(Box::from_raw(m));
	}
}

// extern "C" fn manifold_delete_manifold_vec(ms: *mut ManifoldVec) {}
//
// extern "C" fn manifold_delete_cross_section(cs: *mut ManifoldCrossSection) {}
//
// extern "C" fn manifold_delete_cross_section_vec(csv: *mut ManifoldCrossSectionVec) {}
//
// extern "C" fn manifold_delete_simple_polygon(p: *mut ManifoldSimplePolygon) {}
//
// extern "C" fn manifold_delete_polygons(p: *mut ManifoldPolygons) {}

extern "C" fn manifold_delete_meshgl(m: *mut MeshGL) {
	unsafe {
		drop(Box::from_raw(m));
	}
}

extern "C" fn manifold_delete_meshgl64(m: *mut MeshGL64) {
	unsafe {
		drop(Box::from_raw(m));
	}
}

// extern "C" fn manifold_delete_box(b: *mut ManifoldBox) {}
//
// extern "C" fn manifold_delete_rect(b: *mut ManifoldRect) {}
//
// extern "C" fn manifold_delete_triangulation(m: *mut ManifoldTriangulation) {}

// MeshIO / Export

// #ifdef MANIFOLD_EXPORT
// ManifoldMaterial* manifold_material(void* mem);
// void manifold_material_set_roughness(ManifoldMaterial* mat, double roughness);
// void manifold_material_set_metalness(ManifoldMaterial* mat, double metalness);
// void manifold_material_set_color(ManifoldMaterial* mat, ManifoldVec3 color);
// ManifoldExportOptions* manifold_export_options(void* mem);
// void manifold_export_options_set_faceted(ManifoldExportOptions* options,
//                                          int faceted);
// void manifold_export_options_set_material(ManifoldExportOptions* options,
//                                           ManifoldMaterial* mat);
// void manifold_export_meshgl(const char* filename, MeshGL* mesh,
//                             ManifoldExportOptions* options);
// MeshGL* manifold_import_meshgl(void* mem, const char* filename,
//                                        int force_cleanup);
//
// size_t manifold_material_size();
// size_t manifold_export_options_size();
//
// void manifold_destruct_material(ManifoldMaterial* m);
// void manifold_destruct_export_options(ManifoldExportOptions* options);
//
// void manifold_delete_material(ManifoldMaterial* m);
// void manifold_delete_export_options(ManifoldExportOptions* options);
