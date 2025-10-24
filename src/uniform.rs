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

pub enum ThetaUniform {
    h,
}

impl<T> BaseKernel<T> for Uniform<T>
where
    T: dtype,
{
    type Theta = ThetaUniform;

    fn new(d: usize) -> Self {
        Uniform {
            dim: d,
            h: T::one(),
        }
    }

    fn theta_value(&self, theta: Self::Theta) -> T {
        match theta {
            ThetaUniform::h => self.h,
        }
    }

    fn theta_value_mut(&mut self, theta: Self::Theta) -> &mut T {
        match theta {
            ThetaUniform::h => &mut self.h,
        }
    }

    fn thetas(&self) -> impl Iterator<Item = Self::Theta> {
        std::iter::once(ThetaUniform::h)
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

        let dif = p - q;

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
