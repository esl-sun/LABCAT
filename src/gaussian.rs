use ndarray::ArrayView1;
use num_traits::real::Real;
use simba::scalar::RealField;

use crate::{
    dtype,
    kernel::{Bandwidth, Kernel, PDF},
};

#[derive(Clone, Debug)]
pub struct SphericalGaussian<T>
where
    T: dtype,
{
    dim: usize,
    h: T,
}

impl<T> Kernel<T> for SphericalGaussian<T>
where
    T: dtype + RealField,
{
    fn new(dim: usize) -> Self {
        SphericalGaussian { dim, h: T::one() }
    }

    fn k(&self, p: &[T], q: &[T]) -> T {
        let p: ArrayView1<'_, T> = p.into();
        let q: ArrayView1<'_, T> = q.into();

        #[cfg(debug_assertions)]
        if p.shape() != q.shape() {
            panic!("p and q should have the same shape!");
        }

        let dif = &p - &q;
        let two = T::one() + T::one();
        let exponent = T::neg(T::one() / (two)) * dif.dot(&dif) / self.h; // -0.5 * ...
        let norm_factor = T::one()
            / Real::sqrt(Real::powi(
                two * RealField::pi() * self.h,
                self.dim
                    .try_into()
                    .expect("Converting usize to i32 should not fail!"),
            ));

        norm_factor * Real::exp(exponent)
    }
}

impl<T> Bandwidth<T> for SphericalGaussian<T>
where
    T: dtype,
{
    fn h(&self) -> &T {
        &self.h
    }

    fn update_h(&mut self, new_h: &T) {
        if new_h <= &T::zero() {
            panic!(
                "New bandwidth for spherical gaussian ({:?}) must be non-zero and positive!",
                new_h,
            );
        }

        self.h = *new_h
    }
}

impl<T> PDF for SphericalGaussian<T> where T: dtype {}
