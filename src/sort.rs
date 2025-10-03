use std::mem;
use nalgebra::Point3;
use crate::r#impl::Impl;
use crate::common::AABB;
use crate::parallel::{inclusive_scan, scatter};
use crate::utils::permute;
use crate::vec::{vec_resize, vec_resize_nofill, vec_uninit};
use crate::ManifoldError;
use crate::collider::Collider;

const K_NO_CODE: u32 = 0xFFFFFFFF;

fn morton_code(position: Point3<f64>, bbox: AABB) -> u32
{
	// Unreferenced vertices are marked NaN, and this will sort them to the end
	// (the Morton code only uses the first 30 of 32 bits).
	if position.x.is_nan()
	{
		K_NO_CODE
	}
	else
	{
		Collider::morton_code(position, bbox)
	}
}

impl Impl
{
	///Once halfedge_ has been filled in, this function can be called to create the
	///rest of the internal data structures. This function also removes the verts
	///and halfedges flagged for removal (NaN verts and -1 halfedges).
	pub(crate) fn finish(&mut self)
	{
		if self.halfedge.len() == 0 { return; }
		
		self.calculate_bbox();
		self.set_epsilon(self.epsilon, false);
		if !self.bbox.is_finite()
		{
			// Decimated out of existence - early out.
			self.make_empty(ManifoldError::NoError);
			return;
		}
		
		self.sort_verts();
		let mut face_box: Vec<AABB> = Vec::default();
		let mut face_morton: Vec<u32> = Vec::default();
		self.get_face_box_morton(&mut face_box, &mut face_morton);
		self.sort_faces(&mut face_box, &mut face_morton);
		if self.halfedge.len() == 0 { return; }
		self.compact_props();
		
		debug_assert!(self.halfedge.len() % 6 == 0,
				"Not an even number of faces after sorting faces!");
		debug_assert!(self.mesh_relation.tri_ref.len() == self.num_tri() ||
				self.mesh_relation.tri_ref.len() == 0,
				"Mesh Relation doesn't fit!");
		debug_assert!(self.face_normal.len() == self.num_tri() || self.face_normal.len() == 0,
				"face_normal size = {}, num_tri = {}", self.face_normal.len(), self.num_tri());
		
		self.calculate_normals();
		self.collider = Collider::new(&face_box, &face_morton);
	}
	
	///Sorts the vertices according to their Morton code.
	fn sort_verts(&mut self)
	{
		let num_vert = self.num_vert();
		let mut vert_morton: Vec<u32> = unsafe { vec_uninit(num_vert) };
		for vert in 0..num_vert
		{
			println!("{:?}", self.bbox);
			vert_morton[vert] = morton_code(self.vert_pos[vert], self.bbox);
		}
		
		let mut vert_new2old: Vec<_> = (0..num_vert as i32).collect();
		vert_new2old.sort_by_key(|&i| vert_morton[i as usize]);
		
		self.reindex_verts(&vert_new2old, num_vert);
		
		// Verts were flagged for removal with NaNs and assigned kNoCode to sort
		// them to the end, which allows them to be removed.
		let new_num_vert = vert_new2old.partition_point(|&vert|
				vert_morton[vert as usize] < K_NO_CODE);
		
		vec_resize(&mut vert_new2old, new_num_vert);
		permute(&mut self.vert_pos, &vert_new2old);
		
		if self.vert_normal.len() == num_vert
		{
			permute(&mut self.vert_normal, &vert_new2old);
		}
	}
	
	///Updates the halfedges to point to new vert indices based on a mapping,
	///vertNew2Old. This may be a subset, so the total number of original verts is
	///also given.
	fn reindex_verts(&mut self, vert_new2old: &[i32], old_num_vert: usize)
	{
		let mut vert_old2new: Vec<i32> = unsafe { vec_uninit(old_num_vert) };
		scatter(0..self.num_vert() as i32, vert_new2old, &mut vert_old2new);
		let has_prop = self.num_prop() > 0;
		for edge in &mut self.halfedge
		{
			if edge.start_vert < 0 { continue; }
			edge.start_vert = vert_old2new[edge.start_vert as usize];
			edge.end_vert = vert_old2new[edge.end_vert as usize];
			if !has_prop
			{
				edge.prop_vert = edge.start_vert;
			}
		}
	}
	
	fn compact_props(&mut self)
	{
		if self.num_prop == 0 { return; }
		
		let num_prop = self.num_prop();
		let num_verts = self.properties.len() / num_prop;
		let mut keep = vec![0; num_verts];
		
		for h in &self.halfedge
		{
			keep[h.prop_vert as usize] = 1;
		}
		
		let mut prop_old2new = vec![0_i32; num_verts + 1];
		inclusive_scan(keep.iter().cloned(), &mut prop_old2new);
		
		let old_prop = self.properties.clone();
		let num_verts_new = prop_old2new[num_verts];
		unsafe { vec_resize_nofill(&mut self.properties, num_prop * (num_verts_new as usize)); }
		for old_idx in 0..num_verts
		{
			if keep[old_idx] == 0 { continue; }
			for p in 0..num_prop
			{
				self.properties[prop_old2new[old_idx] as usize * num_prop + p] =
						old_prop[old_idx * num_prop + p];
			}
		}
		
		for edge in &mut self.halfedge
		{
			edge.prop_vert = prop_old2new[edge.prop_vert as usize];
		}
	}
	
	///Fills the faceBox and faceMorton input with the bounding boxes and Morton
	///codes of the faces, respectively. The Morton code is based on the center of
	///the bounding box.
	pub(crate) fn get_face_box_morton(&self, face_box: &mut Vec<AABB>, face_morton: &mut Vec<u32>)
	{
		// faceBox should be initialized
		vec_resize(face_box, self.num_tri());
		unsafe { vec_resize_nofill(face_morton, self.num_tri()); }
		for face in 0..self.num_tri()
		{
			// Removed tris are marked by all halfedges having pairedHalfedge
			// = -1, and this will sort them to the end (the Morton code only
			// uses the first 30 of 32 bits).
			if self.halfedge[(3 * face) as usize].paired_halfedge < 0
			{
				face_morton[face] = K_NO_CODE;
				continue;
			}
			
			let mut center = Point3::<f64>::new(0.0, 0.0, 0.0);
			
			for i in 0..3
			{
				let pos = self.vert_pos[self.halfedge[(3 * face + i) as usize].start_vert as usize];
				center += pos.coords;
				face_box[face].union_point(pos);
			}
			
			center /= 3.;
			
			face_morton[face] = morton_code(center, self.bbox);
		}
	}
	
	///Sorts the faces of this manifold according to their input Morton code. The
	///bounding box and Morton code arrays are also sorted accordingly.
	fn sort_faces(&mut self, face_box: &mut Vec<AABB>, face_morton: &mut Vec<u32>)
	{
		let mut face_new2old: Vec<_> = (0..self.num_tri() as i32).collect();
		face_new2old.sort_by_key(|&i| face_morton[i as usize]);
		
		// Tris were flagged for removal with pairedHalfedge = -1 and assigned kNoCode
		// to sort them to the end, which allows them to be removed.
		let new_num_tri = face_new2old.partition_point(|&face|
				face_morton[face as usize] < K_NO_CODE);
		
		vec_resize(&mut face_new2old, new_num_tri);
		
		permute(face_morton, &face_new2old);
		permute(face_box, &face_new2old);
		self.gather_faces(&face_new2old);
	}
	
	///Creates the halfedge_ vector for this manifold by copying a set of faces from
	///another manifold, given by oldHalfedge. Input faceNew2Old defines the old
	///faces to gather into this.
	fn gather_faces(&mut self, face_new2old: &[i32])
	{
		let num_tri = face_new2old.len();
		if self.mesh_relation.tri_ref.len() == self.num_tri()
		{
			permute(&mut self.mesh_relation.tri_ref, face_new2old);
		}
		
		if self.face_normal.len() == self.num_tri()
		{
			permute(&mut self.face_normal, face_new2old);
		}
		
		let mut old_halfedge = unsafe { vec_uninit(3 * num_tri) };
		mem::swap(&mut old_halfedge, &mut self.halfedge);
		
		let mut face_old2new = unsafe { vec_uninit(old_halfedge.len() / 3) };
		scatter(0..num_tri as i32, face_new2old, &mut face_old2new);
		
		for new_face in 0..num_tri
		{
			let new_face = new_face as i32;
			let old_face = face_new2old[new_face as usize];
			for i in 0..3
			{
				let old_edge = 3 * old_face + i;
				let mut edge = old_halfedge[old_edge as usize];
				let paired_face = edge.paired_halfedge / 3;
				let offset = edge.paired_halfedge - 3 * paired_face;
				edge.paired_halfedge = 3 * face_old2new[paired_face as usize] + offset;
				let new_edge = 3 * new_face + i;
				self.halfedge[new_edge as usize] = edge;
			}
		}
		
	}
}
