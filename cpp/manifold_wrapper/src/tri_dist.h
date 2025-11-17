#pragma once

#include "meshbool/common.h"
#include "meshbool/meshbool.h"

inline auto DistanceTriangleTriangleSquared(
    const std::array<manifold::vec3, 3>& p,
    const std::array<manifold::vec3, 3>& q) {
  using RustVec3 = rust::nalgebra::Vector3<double>;

  auto rust_p = rust::std::vec::Vec<RustVec3>::with_capacity(3);
  for (const auto& t : p) {
    rust_p.push(RustVec3::new_(t.x, t.y, t.z));
  }

  auto rust_q = rust::std::vec::Vec<RustVec3>::with_capacity(3);
  for (const auto& t : q) {
    rust_q.push(RustVec3::new_(t.x, t.y, t.z));
  }

  return rust::crate::tri_dis::distance_triangle_triangle_squared(
      rust_p.as_slice(), rust_q.as_slice());
}
