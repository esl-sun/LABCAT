use faer::ColRef;
use num_traits::real::Real;

use crate::{
    dtype,
    kernel::{Bandwidth, BaseKernel, PDF},
    utils::DtypeUtils,
};

#[derive(Clone, Debug)]
pub struct Uniform<T>
where
    T: dtype,
{
    dim: usize,
    h: T,
}

impl<T> BaseKernel<T> for Uniform<T>
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
        // let p: ArrayView1<'_, T> = p.into();
        // let q: ArrayView1<'_, T> = q.into();
        let p = ColRef::from_slice(p);
        let q = ColRef::from_slice(q);

        #[cfg(debug_assertions)]
        if p.shape() != q.shape() {
            panic!("p and q should have the same shape!");
        }

        let dif = &p - &q;

        if dif.iter().all(|val| val.abs() <= T::one() / self.h) {
            // all within 1/h box
            Real::powi(
                Real::recip(T::two() * self.h),
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
