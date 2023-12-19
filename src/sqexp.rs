use std::{iter::Product, ops::Div};

use ndarray::{Array2, ArrayView1};
use ndarray_linalg::Scalar;
use num_traits::{One, Zero};

use crate::{
    kernel::{BayesianKernel, Kernel, ARD},
    utils::VectorView,
};

pub struct SqExpARD<T>
where
    T: Scalar,
{
    dim: usize,
    sigma_f: T,
    sigma_n: T,
    l: Vec<T>,
    l_inv: Array2<T>,
}

impl<T> Kernel<T> for SqExpARD<T>
where
    T: Scalar,
{
    fn new(d: usize) -> Self {
        SqExpARD {
            dim: d,
            sigma_f: T::one(),
            sigma_n: T::zero(),
            l: vec![T::one(); d],
            l_inv: Array2::eye(d),
        }
    }

    fn k(&self, p: VectorView<T>, q: VectorView<T>) -> T {
        let p: ArrayView1<'_, T> = p.into();
        let q: ArrayView1<'_, T> = q.into();

        #[cfg(debug_assertions)]
        if p.shape() != q.shape() {
            panic!("p and q should have the same shape!");
        }

        let dif = &p - &q;
        let exponent = T::neg(T::one() / (T::one() + T::one())) * dif.dot(&self.l_inv).dot(&dif); // -0.5 * ...
        let val: T = self.sigma_f().powi(2) * exponent.exp();

        match { &p.eq(&q) } {
            true => val + self.sigma_n().powi(2),
            false => val,
        }
    }
}

impl<T> ARD<T> for SqExpARD<T>
where
    T: Scalar + Zero + One + Div<T> + Product,
{
    fn l(&self) -> &[T] {
        &self.l
    }

    fn update_l(&mut self, new_l: &[T]) {
        #[cfg(debug_assertions)]
        if new_l.len() != self.dim {
            panic!("New length-scales have different dim!");
        }

        self.l_inv
            .diag_mut()
            .iter_mut()
            .zip(self.l.iter())
            .for_each(|(old, new)| *old = T::one() / new.powi(2));
    }
}

impl<T> BayesianKernel<T> for SqExpARD<T>
where
    T: Scalar + Zero + One + Product,
{
    fn sigma_f(&self) -> &T {
        &self.sigma_f
    }

    fn sigma_n(&self) -> &T {
        &self.sigma_n
    }

    fn sigma_f_mut(&mut self) -> &mut T {
        &mut self.sigma_f
    }

    fn sigma_n_mut(&mut self) -> &mut T {
        &mut self.sigma_n
    }
}
