use std::iter::Product;

use ndarray::ArrayView1;
use num_traits::real::Real;
use simba::scalar::RealField;

use crate::{
    dtype,
    kernel::{Bandwidth, Kernel, ARD, PDF},
};

#[derive(Clone, Debug)]
pub struct SphericalGaussian<T>
where
    T: dtype,
{
    l: T,
}

impl<T> Kernel<T> for SphericalGaussian<T>
where
    T: dtype + Real + Product + RealField,
{
    fn new(_: usize) -> Self {
        SphericalGaussian { l: T::one() }
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
        let exponent = T::neg(T::one() / (two)) * dif.dot(&dif) / self.l; // -0.5 * ...
        let k = T::one() / Real::sqrt(two * RealField::pi()) * Real::exp(exponent);
        T::one() / self.l * k
    }
}

impl<T> Bandwidth<T> for SphericalGaussian<T>
where
    T: dtype,
{
    fn l(&self) -> &T {
        &self.l
    }

    fn update_l(&mut self, new_l: &T) {
        self.l = *new_l
    }
}

impl<T> PDF for SphericalGaussian<T> where T: dtype {}
