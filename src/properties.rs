use nalgebra::Point3;

use crate::r#impl::Impl;
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

impl Impl {
	/**
		* Returns true if this manifold is in fact an oriented even manifold and all of
		* the data structures are consistent.
		*/
	pub(crate) fn is_manifold(&self) -> bool {
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
	pub(crate) fn is_2_manifold(&self) -> bool {
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

	pub(crate) fn calculate_bbox(&mut self) {
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
}
