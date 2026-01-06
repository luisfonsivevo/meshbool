use core::f64;
use nalgebra::{Matrix3x4, Point2, Point3, Vector2, Vector3};
use std::cmp::Ordering;

pub type SimplePolygon = Vec<Point2<f64>>;
pub type Polygons = Vec<SimplePolygon>;

//struct was originally named "Box", causing name conflict with the built in rust type
///Axis-aligned 3D box, primarily for bounding.
#[derive(Clone, Copy, Debug)]
pub struct AABB {
	pub min: Point3<f64>,
	pub max: Point3<f64>,
}

impl Default for AABB {
	///Default constructor is an infinite box that contains all space.
	fn default() -> Self {
		Self {
			min: Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
			max: Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
		}
	}
}

impl AABB {
	///Creates a box that contains the two given points.
	pub fn new(p1: Point3<f64>, p2: Point3<f64>) -> Self {
		Self {
			min: p1.inf(&p2),
			max: p1.sup(&p2),
		}
	}

	///Returns the dimensions of the Box.
	pub fn size(&self) -> Vector3<f64> {
		self.max - self.min
	}

	///Returns the absolute-largest coordinate value of any contained
	///point.
	pub fn scale(&self) -> f64 {
		self.min.coords.abs().sup(&self.max.coords.abs()).max()
	}

	///Expand this box to include the given point.
	pub fn union_point(&mut self, p: Point3<f64>) {
		self.min = self.min.inf(&p);
		self.max = self.max.sup(&p);
	}

	///Expand this box to include the given box.
	pub fn union_aabb(&self, other: &Self) -> Self {
		Self {
			min: self.min.inf(&other.min),
			max: self.max.sup(&other.max),
		}
	}

	///Transform the given box by the given axis-aligned affine transform.
	///
	///Ensure the transform passed in is axis-aligned (rotations are all
	///multiples of 90 degrees), or else the resulting bounding box will no longer
	///bound properly.
	pub fn transform(&self, transform: Matrix3x4<f64>) -> Self {
		let mut out = Self::default();
		let min_t = Point3::from(transform * self.min.coords.push(1.0));
		let max_t = Point3::from(transform * self.max.coords.push(1.0));
		out.min = min_t.inf(&max_t);
		out.max = min_t.sup(&max_t);
		out
	}

	///Does this box have finite bounds?
	pub fn is_finite(&self) -> bool {
		self.min.iter().all(|x| x.is_finite()) && self.max.iter().all(|x| x.is_finite())
	}
}

pub trait AABBOverlap<T> {
	fn does_overlap(&self, other: &T) -> bool;
}

impl AABBOverlap<AABB> for AABB {
	///Does this box overlap the one given (including equality)?
	fn does_overlap(&self, other: &AABB) -> bool {
		self.min.x <= other.max.x
			&& self.min.y <= other.max.y
			&& self.min.z <= other.max.z
			&& self.max.x >= other.min.x
			&& self.max.y >= other.min.y
			&& self.max.z >= other.min.z
	}
}

impl AABBOverlap<Point3<f64>> for AABB {
	///Does the given point project within the XY extent of this box
	///(including equality)?
	fn does_overlap(&self, p: &Point3<f64>) -> bool {
		// projected in z
		p.x <= self.max.x && p.x >= self.min.x && p.y <= self.max.y && p.y >= self.min.y
	}
}

#[derive(Clone, Copy, Debug)]
pub struct Rect {
	pub min: Point2<f64>,
	pub max: Point2<f64>,
}

impl Default for Rect {
	///Default constructor is an infinite box that contains all space.
	fn default() -> Self {
		Self {
			min: Point2::new(f64::INFINITY, f64::INFINITY),
			max: Point2::new(f64::NEG_INFINITY, f64::NEG_INFINITY),
		}
	}
}

impl Rect {
	pub fn new(a: Point2<f64>, b: Point2<f64>) -> Rect {
		Rect {
			min: a.inf(&b),
			max: a.sup(&b),
		}
	}

	///Return the dimensions of the rectangle.
	pub fn size(&self) -> Vector2<f64> {
		self.max - self.min
	}

	///Returns the absolute-largest coordinate value of any contained
	///point.
	pub fn scale(&self) -> f64 {
		self.min.coords.abs().sup(&self.max.coords.abs()).max()
	}

	///Does this rectangle contain (includes on border) the given point?
	pub fn contains(&self, p: &Point2<f64>) -> bool {
		p.x >= self.min.x && p.y >= self.min.y && p.x <= self.max.x && p.y <= self.max.y
	}

	///Does this rectangle overlap the one given (including equality)?
	pub fn does_overlap(&self, rect: &Rect) -> bool {
		self.min.x <= rect.max.x
			&& self.min.y <= rect.max.y
			&& self.max.x >= rect.min.x
			&& self.max.y >= rect.min.y
	}

	///Expand this rectangle (in place) to include the given point.
	pub fn union(&mut self, p: Point2<f64>) {
		self.min = self.min.inf(&p);
		self.max = self.max.sup(&p);
	}
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum OpType {
	Add,
	Subtract,
	Intersect,
}

pub struct Quality;
impl Quality {
	const DEFAULT_SEGMENTS: u32 = 0;
	const DEFAULT_ANGLE: f64 = 10.0;
	const DEFAULT_LENGTH: f64 = 1.0;

	const CIRCULAR_SEGMENTS: u32 = Self::DEFAULT_SEGMENTS;
	const CIRCULAR_ANGLE: f64 = Self::DEFAULT_ANGLE;
	const CIRCULAR_EDGE_LENGTH: f64 = Self::DEFAULT_LENGTH;

	///Determine the result of the SetMinCircularAngle(),
	///SetMinCircularEdgeLength(), and SetCircularSegments() defaults.
	///
	///@param radius For a given radius of circle, determine how many default
	///segments there will be.
	pub fn get_circular_segments(radius: f64) -> u32 {
		if Self::CIRCULAR_SEGMENTS > 0 {
			return Self::CIRCULAR_SEGMENTS;
		}
		let n_seg_a = (360.0 / Self::CIRCULAR_ANGLE) as u32;
		let n_seg_l = (2.0 * radius * f64::consts::PI / Self::CIRCULAR_EDGE_LENGTH) as u32;
		let mut n_seg = n_seg_a.min(n_seg_l) + 3;
		n_seg -= n_seg % 4;
		n_seg.max(4)
	}
}

//disgusting cursed reimplementation of c++ implicit number type coercion
pub trait LossyFrom<T: Copy> {
	fn lossy_from(other: T) -> Self;
}

//impl lossyfrom instead!
pub trait LossyInto<T: Copy> {
	fn lossy_into(self) -> T;
}

impl<T, U> LossyInto<U> for T
where
	T: Copy,
	U: Copy + LossyFrom<T>,
{
	fn lossy_into(self) -> U {
		U::lossy_from(self)
	}
}

//lossy_from!([from, from, from], to)
macro_rules! lossy_from {
	([ $( $f:ty ),* ], $t:ty) => {
		$(
			impl LossyFrom<$f> for $t {
				fn lossy_from(other: $f) -> Self {
					other as Self
				}
			}
		)*
	};
}

lossy_from!([i32, u32, usize], usize);
lossy_from!([usize], u32);
lossy_from!([f64], f64);
lossy_from!([f64], f32);

pub struct OrderedF64(pub f64);

impl Ord for OrderedF64 {
	fn cmp(&self, other: &Self) -> Ordering {
		self.0.total_cmp(&other.0)
	}
}

impl PartialOrd for OrderedF64 {
	fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
		Some(self.cmp(other))
	}
}

impl Eq for OrderedF64 {}

impl PartialEq for OrderedF64 {
	fn eq(&self, other: &Self) -> bool {
		self.0.total_cmp(&other.0) == Ordering::Equal
	}
}

///Returns arc cosine of 𝑥.
///
///@return value in range [0,M_PI]
///@return NAN if 𝑥 ∈ {NAN,+INFINITY,-INFINITY}
///@return NAN if 𝑥 ∉ [-1,1]
pub fn sun_acos(x: f64) -> f64 {
	/*
	 * Origin of acos function: FreeBSD /usr/src/lib/msun/src/e_acos.c
	 * Changed the use of union to memcpy to avoid undefined behavior.
	 * ====================================================
	 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
	 *
	 * Developed at SunSoft, a Sun Microsystems, Inc. business.
	 * Permission to use, copy, modify, and distribute this
	 * software is freely granted, provided that this notice
	 * is preserved.
	 * ====================================================
	 */
	const PIO2_HI: f64 = 1.57079632679489655800e+00; /* 0x3FF921FB, 0x54442D18 */
	const PIO2_LO: f64 = 6.12323399573676603587e-17; /* 0x3C91A626, 0x33145C07 */
	const P_S0: f64 = 1.66666666666666657415e-01; /* 0x3FC55555, 0x55555555 */
	const P_S1: f64 = -3.25565818622400915405e-01; /* 0xBFD4D612, 0x03EB6F7D */
	const P_S2: f64 = 2.01212532134862925881e-01; /* 0x3FC9C155, 0x0E884455 */
	const P_S3: f64 = -4.00555345006794114027e-02; /* 0xBFA48228, 0xB5688F3B */
	const P_S4: f64 = 7.91534994289814532176e-04; /* 0x3F49EFE0, 0x7501B288 */
	const P_S5: f64 = 3.47933107596021167570e-05; /* 0x3F023DE1, 0x0DFDF709 */
	const Q_S1: f64 = -2.40339491173441421878e+00; /* 0xC0033A27, 0x1C8A2D4B */
	const Q_S2: f64 = 2.02094576023350569471e+00; /* 0x40002AE5, 0x9C598AC8 */
	const Q_S3: f64 = -6.88283971605453293030e-01; /* 0xBFE6066C, 0x1B8D0159 */
	const Q_S4: f64 = 7.70381505559019352791e-02; /* 0x3FB3B8C5, 0xB12E9282 */
	let r = |z| {
		let p = z * (P_S0 + z * (P_S1 + z * (P_S2 + z * (P_S3 + z * (P_S4 + z * P_S5)))));
		let q = 1.0 + z * (Q_S1 + z * (Q_S2 + z * (Q_S3 + z * Q_S4)));
		p / q
	};
	//double z, w, s, c, df;
	//uint64_t xx;
	//uint32_t hx, lx, ix;
	let mut xx = x.to_bits();
	let hx = (xx >> 32) as u32;
	let ix = hx & 0x7fffffff;
	/* |x| >= 1 or nan */
	if ix >= 0x3ff00000 {
		let lx = xx as u32;
		if ((ix - 0x3ff00000) | lx) == 0 {
			/* acos(1)=0, acos(-1)=pi */
			if (hx >> 31) != 0 {
				return 2.0 * PIO2_HI + 2.0_f64.powi(-120);
			}
			return 0.0;
		}
		return f64::NAN;
	}
	/* |x| < 0.5 */
	if ix < 0x3fe00000 {
		if ix <= 0x3c600000
		/* |x| < 2**-57 */
		{
			return PIO2_HI + 2.0_f64.powi(-120);
		}

		return PIO2_HI - (x - (PIO2_LO - x * r(x * x)));
	}
	/* x < -0.5 */
	if (hx >> 31) != 0 {
		let z = (1.0 + x) * 0.5;
		let s = z.sqrt();
		let w = r(z) * s - PIO2_LO;
		return 2.0 * (PIO2_HI - (s + w));
	}
	/* x > 0.5 */
	let z = (1.0 - x) * 0.5;
	let s = z.sqrt();
	xx = s.to_bits();
	xx &= 0xffffffff00000000;
	let df = f64::from_bits(xx);
	let c = (z - df * df) / (s + df);
	let w = r(z) * s + c;
	2.0 * (df + w)
}
