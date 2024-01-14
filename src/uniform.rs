use std::iter::Product;

use ndarray::{Array1, ArrayView1};
use num_traits::real::Real;
use simba::scalar::RealField;

use crate::{
    dtype,
    kernel::{Bandwidth, Kernel, PDF},
};

#[derive(Clone, Debug)]
pub struct Uniform<T>
where
    T: dtype,
{
    dim: usize,
    l: T,
}

impl<T> Kernel<T> for Uniform<T>
where
    T: dtype + Real + Product + RealField,
{
    fn new(d: usize) -> Self {
        Uniform {
            dim: d,
            l: T::one(),
        }
    }

    fn k(&self, p: &[T], q: &[T]) -> T {
        let p: ArrayView1<'_, T> = p.into();
        let q: ArrayView1<'_, T> = q.into();

        #[cfg(debug_assertions)]
        if p.shape() != q.shape() {
            panic!("p and q should have the same shape!");
        }

        let dif: Array1<T> = &p - &q;

        if dif.iter().all(|val| val.abs() <= T::one() / self.l) {
            // all within 1/l box
            T::one() / self.l
                * Real::powi(
                    T::one() / (T::one() + T::one()),
                    self.dim.try_into().unwrap(),
                ) // (1/l) * (0.5)^d
        } else {
            T::zero()
        }
    }
}

impl<T> Bandwidth<T> for Uniform<T>
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

impl<T> PDF for Uniform<T> where T: dtype {}
