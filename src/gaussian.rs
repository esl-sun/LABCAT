use faer::ColRef;
// use ndarray::ArrayView1;
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

pub enum ThetaSphericalGuassian {
    h,
}

impl<T> BaseKernel<T> for SphericalGaussian<T>
where
    T: dtype + RealField,
{
    type Theta = ThetaSphericalGuassian;

    fn new(dim: usize) -> Self {
        SphericalGaussian { dim, h: T::one() }
    }

    fn theta_value(&self, theta: Self::Theta) -> T {
        match theta {
            ThetaSphericalGuassian::h => self.h,
        }
    }

    fn theta_value_mut(&mut self, theta: Self::Theta) -> &mut T {
        match theta {
            ThetaSphericalGuassian::h => &mut self.h,
        }
    }

    fn thetas(&self) -> impl Iterator<Item = Self::Theta> {
        std::iter::once(ThetaSphericalGuassian::h)
    }

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
        //TODO: .. / h^2 ?
        let exponent = T::neg(T::half()) * (dif.transpose() * &dif) / self.h; // -0.5 * ...
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
