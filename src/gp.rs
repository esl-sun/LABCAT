use std::ops::{Add, Sub};

use anyhow::Result;
use ndarray::*;
use ndarray_linalg::{error::LinalgError, *};

use crate::acq::ExpectedImprovement;
use crate::bounds_array::ArrayBounds;
use crate::f_;
use crate::hyp_opt::HyperparameterOptimizer;
use crate::kernel::{Kernel, KernelState};
use crate::memory::{Memory, MemoryState};
use crate::utils::{Array1Utils, Array2Utils};
use statrs::distribution::Normal;

pub struct GP<kern: Kernel> {
    pub dim: usize,
    pub bounds: ArrayBounds,
    pub beta: f_,
    pub search_dom: ArrayBounds,
    pub kernel: kern,
    pub prior_sigma: f_,
    // pub state: GPState,
    pub mem: Memory,
    pub K: Array2<f_>,
    pub Kinv: Array2<f_>,
    pub L: CholeskyFactorized<OwnedRepr<f_>>,
    pub alpha: Array2<f_>,
    pub n: Normal,
}

impl<kern: Kernel> GP<kern>
where
    Self: HyperparameterOptimizer + ExpectedImprovement,
{
    pub fn new(bounds: ArrayBounds, beta: f_, prior_sigma: f_) -> GP<kern> {
        let dim = bounds.dim();
        let search_dom = ArrayBounds::new_continuous(dim, beta, -beta);
        // let search_dom_LHS = search_dom.LHS_sample(5);
        GP {
            dim,
            bounds,
            beta,
            search_dom,
            // search_dom_LHS,
            prior_sigma,
            kernel: kern::new(dim),
            mem: Memory::new(dim),
            K: Array2::eye(dim),
            Kinv: Array2::eye(dim),
            L: Array2::eye(dim)
                .factorizec(UPLO::Lower)
                .expect("Should never fail during init."),
            alpha: Array2::ones((dim, 1)),
            n: Normal::new(0.0, 1.0).unwrap(),
        }
    }

    pub fn fit(&mut self) -> Result<(), LinalgError> {
        match (self.kernel.state(), self.mem.state()) {
            (KernelState::Fitted, MemoryState::Fitted) => return Ok(()), // model has already been fitted, early return
            (_, _) => assert!(true),
        }

        self.K = Array2::zeros((self.mem.n(), self.mem.n()))
            .map_UPLO(UPLO::Lower, |(i, j)| {
                self.kernel.k(self.mem.X.column(i), self.mem.X.column(j))
            })
            .fill_with_UPLO(UPLO::Lower);

        self.L = self.K.factorizec(UPLO::Lower)?;
        self.Kinv = self.L.invc()?;

        let mut y = self.mem.y_m();
        // self.L.ln_detc()
        self.L.solvec_inplace(&mut y)?;
        self.alpha = y.into_col();

        // self.kernel.calc_thetas_jac(&self.K, &self.mem);
        // self.kernel.calc_thetas_hess(&self.K, &self.mem);

        self.kernel.set_fitted();
        self.mem.set_fitted();

        Ok(())
    }

    pub fn predict_single(&self, x: ArrayView1<f_>) -> Result<(f_, f_)> {
        match (self.kernel.state(), self.mem.state()) {
            (KernelState::Fitted, MemoryState::Fitted) => assert!(true),
            (_, _) => anyhow::bail!("Cannot predict with unfitted GP Model!"), // model has not been fitted, early return
        }

        //test x knownObs
        let k_diag = self.kernel.k_diag(self.mem.X.view(), x);

        let pred_mean = k_diag.dot(&self.alpha.column(0)) + self.mem.y_prime_mean(); //checked

        let v = self.L.solvec(&k_diag)?;
        let pred_sigma = self
            .kernel
            .k(x, x)
            .sub(k_diag.dot(&v))
            .add(self.kernel.sigma_n().powi(2))
            .abs()
            .sqrt();
        Ok((pred_mean, pred_sigma))
    }

    pub fn predict(&self, X: Array2<f_>) -> Result<(Array2<f_>, Array2<f_>)> {
        match (self.kernel.state(), self.mem.state()) {
            (KernelState::Fitted, MemoryState::Fitted) => assert!(true),
            (_, _) => anyhow::bail!("Cannot predict with unfitted GP Model!"), // model has not been fitted, early return
        }

        //test x knownObs
        let k_diag = Array2::from_shape_fn((X.ncols(), self.mem.X.ncols()), |(i, j)| {
            self.kernel.k(X.column(i), self.mem.X.column(j))
        });

        let pred_mean = k_diag.dot(&self.alpha) + self.mem.y_prime_mean(); //checked

        let pred_var = Array2::from_shape_fn((X.ncols(), 1), |(i, _)| {
            let v = self.L.solvec(&k_diag.row(i)).expect("Should not fail"); //TODO: Bubble up error
            (self.kernel.k(X.column(i), X.column(i)))
                .sub(k_diag.row(i).dot(&v))
                .add(self.kernel.sigma_n().powi(2))
                .abs()
                .sqrt()
        });

        Ok((pred_mean, pred_var))
    }
}
