use std::ops::Mul;

use anyhow::{Result, bail};
use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, ArrayView1, Axis};
use ndarray_linalg::SolveC;
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use rand::Rng;
use sobol::Sobol;
use sobol::params::JoeKuoD6;
use statrs::distribution::{Continuous, ContinuousCDF};

use crate::{
    f_,
    gp::GP,
    kernel::{Kernel, SquaredExponential},
    utils::{Array2Utils, ArrayView1Utils},
};

pub trait ExpectedImprovement {
    fn ei(&self, x: ArrayView1<f_>) -> f_;
    fn optimize_ei(&self, n: usize) -> Result<(f_, Array1<f_>)>;
    fn random_valid_pt(&self, n: usize) -> Result<(f_, Array1<f_>)>;
}

impl<kern: Kernel> ExpectedImprovement for GP<kern> {
    //checked
    fn ei(&self, x: ArrayView1<f_>) -> f_ {
        let (mean, sigma) = self.predict_single(x).unwrap();

        let z = (self.mem.y_prime_min() - mean) / sigma;
        
        sigma * (z * self.n.cdf(z.into()) as f_ + self.n.pdf(z.into()) as f_)
    }

    fn optimize_ei(&self, n: usize) -> Result<(f_, Array1<f_>)> {
        let X = Array2::random((self.dim, n), Uniform::new(-self.beta, self.beta));

        let mut res = vec![]; //working
        res.extend(
            X.columns()
                .into_iter()
                .map(|col| (self.ei(col), col.to_owned())),
        );

        let max = res
            .into_iter()
            .filter_map(|tup| match tup.0.is_finite() {
                true => Some(tup),
                false => None,
            })
            .filter_map(|tup| match !self.mem.in_memory(tup.1.view()) {
                true => Some(tup),
                false => None,
            })
            .filter_map(
                |tup| match self.bounds.inside((self.mem.x_test(tup.1.view())).view()) {
                    // Rejection sampling for target f bounds
                    true => Some(tup),
                    false => None,
                },
            )
            .max_by(|(a, _), (b, _)| (a).total_cmp(b));

        match max {
            Some(max) => Ok(max),
            None => self.random_valid_pt(self.dim * 100),
        }
    }

    fn random_valid_pt(&self, n: usize) -> Result<(f_, Array1<f_>)> {
        for _ in 0..n {
            let x = Array1::random((self.dim,), Uniform::new(-self.beta, self.beta));

            if self.mem.in_memory(x.view()) {
                continue;
            }

            if !self.bounds.inside((self.mem.x_test(x.view())).view()) {
                continue;
            }

            return Ok((self.ei(x.view()), x));
        }

        bail!("EI point not found!")
    }
}
