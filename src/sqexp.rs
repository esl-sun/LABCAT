use std::{iter::Product, ops::Div};

use faer::{unzip, zip, Mat};
use ndarray::{Array2, ArrayView1};
use num_traits::real::Real;

use crate::{
    dtype,
    gp::GPSurrogate,
    kernel::{Bandwidth, BaseKernel, BayesianKernel, ARD},
    memory::ObservationIO,
    utils::DtypeUtils,
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
        let exponent = T::half().neg() * dif.dot(&dif) / self.l; // -0.5 * ...
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

    fn sigma_f_gp_jac<S: GPSurrogate<T, KernType = Self>>(&self, gp: &S) -> Mat<T> {
        zip!(gp.K()).map_with_index(|i, j, unzip!(k)| {
            T::two()
                * if i == j {
                    *k - gp.kernel().sigma_n().powi(2)
                } else {
                    *k
                }
        })
    }

    fn sigma_n_gp_jac<S: GPSurrogate<T, KernType = Self>>(&self, _gp: &S) -> Mat<T> {
        todo!()
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
        let exponent = T::half().neg() * dif.dot(&self.l_inv).dot(&dif); // -0.5 * ...
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

    fn dim(&self) -> usize {
        self.dim
    }

    fn update_l(&mut self, new_l: &[T]) {
        // #[cfg(debug_assertions)] TODO: Add unchecked ver?
        if new_l.len() != self.dim {
            panic!("New length-scales have different dim!");
        }

        self.l
            .iter_mut()
            .zip(new_l.iter())
            .for_each(|(old_l, new_l)| *old_l = *new_l);

        self.l_inv
            .diag_mut()
            .iter_mut()
            .zip(self.l.iter())
            .for_each(|(old, new)| *old = T::one() / new.powi(2));
    }

    fn whiten_l(&mut self) {
        self.update_l(&vec![T::one(); self.dim()])
    }

    fn l_gp_jac<S: GPSurrogate<T, KernType = Self>>(&self, gp: &S) -> impl Iterator<Item = Mat<T>> {
        self.l().iter().enumerate().map(|(d, l)| {
            zip!(gp.K()).map_with_index(|i, j, unzip!(k)| {
                *k * (gp.memory().i(i).0[d] - gp.memory().i(j).0[d]).powi(2) / l.powi(2)
            })
        })
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

    fn sigma_f_gp_jac<S: GPSurrogate<T, KernType = Self>>(&self, gp: &S) -> Mat<T> {
        zip!(gp.K()).map_with_index(|i, j, unzip!(k)| {
            T::two()
                * if i == j {
                    *k - gp.kernel().sigma_n().powi(2)
                } else {
                    *k
                }
        })
    }

    fn sigma_n_gp_jac<S: GPSurrogate<T, KernType = Self>>(&self, _gp: &S) -> Mat<T> {
        todo!()
    }
}
