use std::ops::{MulAssign, SubAssign};

use ndarray::*;

use crate::f_;
use crate::memory::Memory;
use crate::utils::Array2Utils;

#[derive(Clone, Debug)]
pub enum KernelState {
    Fitted,
    Unfitted,
}

#[derive(Clone, Debug)]
pub enum DerivState {
    Calculated,
    Uncalculated,
}
pub trait Kernel {
    fn new(d: usize) -> Self;
    fn state(&self) -> &KernelState;
    fn set_fitted(&mut self);
    fn thetas(&self) -> &Array1<f_>;
    fn update_thetas(&mut self, new_thetas: &Array1<f_>);
    fn whiten_l(&mut self);
    fn l(&self) -> ArrayView1<f_>;
    fn ln_l(&self) -> Array1<f_>;
    fn sigma_f(&self) -> &f_;
    fn sigma_n(&self) -> &f_;
    fn k(&self, x1: ArrayView1<f_>, x2: ArrayView1<f_>) -> f_;
    fn thetas_jac(&self) -> &Array3<f_>;
    fn thetas_hess(&self) -> &Array4<f_>;
    fn calc_thetas_jac(&self, K: &Array2<f_>, mem: &Memory) -> Array3<f_>;
    fn calc_thetas_hess(&self, K: &Array2<f_>, mem: &Memory) -> Array4<f_>;
    fn k_diag(&self, X: ArrayView2<f_>, x_test: ArrayView1<f_>) -> Array1<f_>;
    fn obs_jac(&self, X: &Array2<f_>, x_test: ArrayView1<f_>) -> Array2<f_>;
}

#[derive(Clone)]
pub struct SquaredExponential {
    thetas: Array1<f_>,
    l_inv: Array2<f_>,
    state: KernelState,
    jac: Array3<f_>,
    jac_state: DerivState,
    hess: Array4<f_>,
    hess_state: DerivState,
}

impl SquaredExponential
where
    Self: Kernel,
{
    fn update_l_inv(&mut self) {
        let l = self.l().into_owned();

        self.l_inv
            .diag_mut()
            .iter_mut()
            .zip(l.iter())
            .for_each(|(old, new)| *old = 1.0 / new.powi(2));
    }
}

impl Kernel for SquaredExponential {
    fn new(d: usize) -> Self {
        let mut thetas = Array1::ones((d + 2,));
        thetas[1] = 1e-6; //set sigma_n;

        SquaredExponential {
            thetas,
            l_inv: Array2::eye(d),
            state: KernelState::Unfitted,
            jac: Array3::zeros((d + 1, 0, 0)),
            jac_state: DerivState::Uncalculated,
            hess: Array4::zeros((d, d, 0, 0)),
            hess_state: DerivState::Uncalculated,
        }
    }

    fn state(&self) -> &KernelState {
        &self.state
    }

    fn set_fitted(&mut self) {
        self.state = KernelState::Fitted;
    }

    fn thetas(&self) -> &Array1<f_> {
        &self.thetas
    }

    fn update_thetas(&mut self, new_thetas: &Array1<f_>) {
        if self.thetas.shape() != new_thetas.shape() {
            panic!("thetas shape mismatch!")
        }
        // dbg!(&new_thetas);
        if self.thetas != new_thetas {
            self.thetas.assign(new_thetas);
            self.update_l_inv();
            self.jac_state = DerivState::Uncalculated;
            self.hess_state = DerivState::Uncalculated;
            self.state = KernelState::Unfitted
        }
    }

    fn whiten_l(&mut self) {
        self.thetas.iter_mut().skip(2).for_each(|l| *l = 1.0);
        self.update_l_inv();
        self.jac_state = DerivState::Uncalculated;
        self.hess_state = DerivState::Uncalculated;
        self.state = KernelState::Unfitted;
    }

    fn l(&self) -> ArrayView1<f_> {
        self.thetas.slice(s![2..])
    }

    fn ln_l(&self) -> Array1<f_> {
        self.thetas.slice(s![2..]).map(|val| val.ln()) //TODO: Store ln thetas to avoid duplicate ln calculations
    }

    fn sigma_f(&self) -> &f_ {
        &self.thetas[0]
    }

    fn sigma_n(&self) -> &f_ {
        &self.thetas[1]
    }

    fn k(&self, x1: ArrayView1<f_>, x2: ArrayView1<f_>) -> f_ {
        #[cfg(debug_assertions)]
        if x1.shape() != x2.shape() {
            panic!("x1 and x2 should have the same shape!");
        }

        let dif = &x1 - &x2;

        let exponent = -0.5 * (&dif.dot(&self.l_inv).dot(&dif));

        let val = self.sigma_f().powi(2) * exponent.exp()
            + match { x1.eq(&x2) } {
                true => self.sigma_n().powi(2),
                false => 0.0,
            };

        val
    }

    fn thetas_jac(&self) -> &Array3<f_> {
        &self.jac
    }

    fn thetas_hess(&self) -> &Array4<f_> {
        &self.hess
    }

    // TODO: CONSIDER ONLY CALCULATING FOR LENGTH SCALES
    // first axis is hyperparam index, in ln space, use outer_iter, checked
    fn calc_thetas_jac(&self, K: &Array2<f_>, mem: &Memory) -> Array3<f_> {
        // match self.jac_state {
        //     DerivState::Calculated => return,
        //     DerivState::Uncalculated => assert!(true),
        // }

        let mut jacs =
            Array3::from_shape_fn((self.l().len() + 1, K.nrows(), K.ncols()), |(_, i, j)| {
                K[(i, j)] //fill every submatrix with K values
            });

        jacs.outer_iter_mut().take(1).for_each(|mut K| {
            K.diag_mut()
                .iter_mut()
                .for_each(|val| val.sub_assign(self.sigma_n().powi(2)));
            K.mul_assign(2.0)
        });

        jacs.outer_iter_mut()
            .skip(1)
            .enumerate()
            .for_each(|(d, mut K)| {
                K.indexed_iter_mut().for_each(|((i, j), val)| {
                    val.mul_assign(
                        (mem.X.column(i)[d] - mem.X.column(j)[d]).powi(2) / self.l()[d].powi(2),
                    )
                })
            });

        // self.jac_state = DerivState::Calculated;
        // self.jac = jacs;
        jacs
    }

    // first and second axes are hyperparam index, in ln space, use .slice(s![i, i, .., ..]), checked
    fn calc_thetas_hess(&self, K: &Array2<f_>, mem: &Memory) -> Array4<f_> {
        // match self.hess_state {
        //     DerivState::Calculated => return,
        //     DerivState::Uncalculated => assert!(true),
        // }

        // match self.jac_state {
        //     DerivState::Calculated => assert!(true),
        //     DerivState::Uncalculated => self.calc_thetas_jac(K, mem),
        // }

        // let jacs = self.thetas_jac();
        let jacs = self.calc_thetas_jac(K, mem);

        let mut hess = Array4::from_shape_fn(
            (self.l().len() + 1, self.l().len() + 1, K.nrows(), K.ncols()),
            |(x, y, i, j)| {
                match x <= y {
                    true => jacs[(y, i, j)],
                    false => 0.0, // filling top half with zeroes, seems inefficient
                }
            },
        );

        // d(sigma_f) (sigma_f)
        hess.slice_mut(s![0, 0, .., ..])
            .par_map_inplace(|val| val.mul_assign(2.0));

        // d(sigma_f) (l_a)
        for d in 0..self.l().len() {
            hess.slice_mut(s![0, d + 1, .., ..])
                .par_map_inplace(|val| val.mul_assign(2.0));
        }

        // d(l_a) (l_b)
        for d2 in 0..self.l().len() {
            for d1 in 0..d2 {
                hess.slice_mut(s![d1 + 1, d2 + 1, .., ..])
                    .indexed_iter_mut()
                    .for_each(|((i, j), val)| {
                        *val = *val * (mem.X.column(i)[d1] - mem.X.column(j)[d1]).powi(2)
                            / self.l()[d1].powi(2)
                    })
            }
        }

        // d(l_a) (l_a)
        for d in 0..self.l().len() {
            hess.slice_mut(s![d + 1, d + 1, .., ..])
                .indexed_iter_mut()
                .for_each(|((i, j), val)| {
                    *val = *val * (mem.X.column(i)[d] - mem.X.column(j)[d]).powi(2)
                        / self.l()[d].powi(2)
                        - 2.0 * *val
                })
        }

        // self.hess_state = DerivState::Calculated;
        // self.hess = hess;
        hess
    }

    fn k_diag(&self, X: ArrayView2<f_>, x_test: ArrayView1<f_>) -> Array1<f_> {
        Array1::from_shape_fn((X.ncols(),), |i| self.k(x_test, X.column(i)))
    }

    fn obs_jac(&self, X: &Array2<f_>, x_test: ArrayView1<f_>) -> Array2<f_> {
        let k_diag = self.k_diag(X.view(), x_test.view());

        let X = X.to_owned().sub_column_view(&x_test);

        self.l_inv.dot(&X).mul_row(&k_diag)
    }
}
