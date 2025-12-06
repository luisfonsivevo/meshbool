use crate::meshboolimpl::MeshBoolImpl;

impl MeshBoolImpl {
	pub fn is_marked_inside_quad(&self, _halfedge: i32) -> bool {
		// if !self.halfedge_tangent.is_empty() {
		// 	return self.halfedge_tangent[halfedge as usize].w < 0;
		// }
		return false;
	}
}
