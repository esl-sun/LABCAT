use ndarray::ArrayView1;
use num_traits::real::Real;
use simba::scalar::RealField;

use crate::{
    dtype,
    kernel::{Bandwidth, BaseKernel, PDF},
    utils::DtypeUtils,
};

#[derive(Clone, Debug)]
pub struct SphericalGaussian<T>
where
    T: dtype,
{
    dim: usize,
    h: T,
}

impl<T> BaseKernel<T> for SphericalGaussian<T>
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
        //TODO: .. / h^2 ?
        let exponent = T::neg(T::half()) * dif.dot(&dif) / self.h; // -0.5 * ...
        let norm_factor = Real::recip(Real::sqrt(Real::powi(
            T::two_pi() * self.h,
            self.dim
                .try_into()
                .expect("Converting usize to i32 should not fail!"),
        )));

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
