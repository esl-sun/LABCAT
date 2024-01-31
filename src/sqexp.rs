use std::{iter::Product, ops::Div};

use ndarray::{Array2, ArrayView1};
use num_traits::real::Real;

use crate::{
    dtype,
    kernel::{Bandwidth, BaseKernel, BayesianKernel, ARD},
};

#[derive(Clone, Debug)]
pub struct SqExp<T>
where
    T: dtype,
{
    sigma_f: T,
    sigma_n: T,
    l: T,
}

impl<T> BaseKernel<T> for SqExp<T>
where
    T: dtype + Real + Product + Div<Output = T>,
{
    fn new(_: usize) -> Self {
        SqExp {
            sigma_f: T::one(),
            sigma_n: T::zero(),
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

        let dif = &p - &q;
        let exponent = (T::one() + T::one()).recip().neg() * dif.dot(&dif) / self.l; // -0.5 * ...
        let val: T = self.sigma_f().powi(2) * exponent.exp();

        match &p.eq(&q) {
            true => val + self.sigma_n().powi(2),
            false => val,
        }
    }
}

impl<T> Bandwidth<T> for SqExp<T>
where
    T: dtype,
{
    fn h(&self) -> &T {
        &self.l
    }

    fn update_h(&mut self, new_h: &T) {
        self.l = *new_h
    }
}

impl<T> BayesianKernel<T> for SqExp<T>
where
    T: dtype + Real + Product + Div<Output = T>,
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

////////////////////////////////////////////////

#[derive(Clone, Debug)]
pub struct SqExpARD<T>
where
    T: dtype,
{
    dim: usize,
    sigma_f: T,
    sigma_n: T,
    l: Vec<T>,
    l_inv: Array2<T>,
}

impl<T> BaseKernel<T> for SqExpARD<T>
where
    T: dtype + Real + Product,
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

    fn k(&self, p: &[T], q: &[T]) -> T {
        let p: ArrayView1<'_, T> = p.into();
        let q: ArrayView1<'_, T> = q.into();

        #[cfg(debug_assertions)]
        if p.shape() != q.shape() {
            panic!("p and q should have the same shape!");
        }

        let dif = &p - &q;
        let exponent = T::neg(T::one() / (T::one() + T::one())) * dif.dot(&self.l_inv).dot(&dif); // -0.5 * ...
        let val: T = self.sigma_f().powi(2) * exponent.exp();

        match &p.eq(&q) {
            true => val + self.sigma_n().powi(2),
            false => val,
        }
    }
}

impl<T> ARD<T> for SqExpARD<T>
where
    T: dtype + Real + Product + Div<Output = T>,
{
    fn l(&self) -> &[T] {
        &self.l
    }

    fn update_l(&mut self, new_l: &[T]) {
        // #[cfg(debug_assertions)] TODO: Add unchecked ver?
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
    T: dtype + Real + Product,
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
