use ndarray::{Array1, ArrayView1};
use num_traits::real::Real;

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
    h: T,
}

impl<T> Kernel<T> for Uniform<T>
where
    T: dtype,
{
    fn new(d: usize) -> Self {
        Uniform {
            dim: d,
            h: T::one(),
        }
    }

    //TODO: Check normalization
    fn k(&self, p: &[T], q: &[T]) -> T {
        let p: ArrayView1<'_, T> = p.into();
        let q: ArrayView1<'_, T> = q.into();

        #[cfg(debug_assertions)]
        if p.shape() != q.shape() {
            panic!("p and q should have the same shape!");
        }

        let dif: Array1<T> = &p - &q;

        if dif.iter().all(|val| val.abs() <= T::one() / self.h) {
            // all within 1/h box
            Real::powi(
                T::one() / (T::one() + T::one()) * T::one() / self.h,
                self.dim
                    .try_into()
                    .expect("Conversion from usize to i32 should not fail!"),
            ) // ((1/h) * (0.5))^d
        } else {
            T::zero()
        }
    }
}

impl<T> Bandwidth<T> for Uniform<T>
where
    T: dtype,
{
    fn h(&self) -> &T {
        &self.h
    }

    fn update_h(&mut self, new_h: &T) {
        self.h = *new_h
    }
}

impl<T> PDF for Uniform<T> where T: dtype {}
