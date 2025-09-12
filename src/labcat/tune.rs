use std::iter::once;

use anyhow::{anyhow, Result};
use faer::linalg::solvers::DenseSolveCore;
use faer::{unzip, zip, Col, Mat};

use crate::{
    dtype,
    gp::GPSurrogate,
    kernel::{BayesianKernel, ARD},
    memory::ObservationVariance,
    tune::SurrogateTuning,
    utils::{DtypeUtils, MatRefUtils},
    SurrogateIO,
};

#[derive(Debug, Clone)]
pub struct LABCAT_GPTune<T>
where
    T: dtype,
{
    prior_sigma: T, // GIVING ISSUES
}

impl<T: dtype> Default for LABCAT_GPTune<T> {
    fn default() -> Self {
        Self {
            prior_sigma: T::one().powi(-1),
        }
    }
}

impl<T> LABCAT_GPTune<T>
where
    T: dtype,
{
    fn prior_sigma(&self) -> &T {
        &self.prior_sigma
    }

    fn log_lik_prior<S>(&self, gp: &S) -> Option<T>
    where
        S: SurrogateIO<T> + GPSurrogate<T, KernType: ARD<T> + BayesianKernel<T>>,
    {
        Some(
            gp.log_lik()?
                + T::half().neg()
                    * self.prior_sigma().powi(2).recip()
                    * gp.kernel()
                        .l()
                        .iter()
                        .fold(T::zero(), |acc, l| acc + l.ln() * l.ln()),
        )
    }

    fn log_lik_jac<S>(&self, gp: &S) -> Option<Col<T>>
    where
        S: SurrogateIO<T> + GPSurrogate<T, KernType: ARD<T> + BayesianKernel<T>>,
    {
        let inner =
            zip!(&(gp.alpha() * gp.alpha().transpose()), gp.K_inv()).map(|unzip!(a, k)| *a - *k);

        // let _sigma_f = gp.kernel().sigma_f_gp_jac(gp).product_trace(inner.as_ref()) * T::half();

        // let _l = gp.kernel().l_gp_jac(gp);

        let jac = once(gp.kernel().sigma_f_gp_jac(gp))
            .chain(gp.kernel().l_gp_jac(gp))
            .map(|jac| T::half() * jac.product_trace(inner.as_ref()));

        // let l = zipped!(gp.K())
        // .map_with_index(|i, j, unzipped!(k)| {  });

        todo!()
    }
}

impl<T, S> SurrogateTuning<T, S> for LABCAT_GPTune<T>
where
    T: dtype,
    S: SurrogateIO<T>
        + GPSurrogate<T, KernType: ARD<T> + BayesianKernel<T>, MemType: ObservationVariance<T>>,
{
    fn tune(&self, sur: &mut S) -> Result<()> {

        let base_log_lik = sur.log_lik().ok_or_else(|| anyhow!("Failed to calculate marginal log likelihood!"))?;
        let base_kernel = sur.kernel().clone();

        *sur.kernel_mut().sigma_f_mut() = sur.memory().Y_std(T::zero());

        sur.refit()?; // TODO: Refit neccesary as this point?

        let jac: Col<T> = Col::zeros(5);
        let hess: Mat<T> = Mat::zeros(5, 5);

        let eigs = hess
            .self_adjoint_eigenvalues(faer::Side::Lower)
            .map_err(|_| {
                anyhow!("Failed to calculate marginal log likelihood Hessian eigenvalues!")
            })?; // TODO: Assumes hessian is symmetric
        
        if eigs.into_iter().all(|eig| eig < T::zero()) {
            let delta = hess.llt(faer::Side::Lower)?.inverse() * jac;        
            ARDBacktrackingLineSearch::new(delta, T::half(), 5).tune(sur)?;
        } else {
            ARDBacktrackingLineSearch::new(jac, T::one().powi(-1), 5).tune(sur)?;
        }

        // No improvement to log marginal likelihood, revert kernel thetas
        if base_log_lik > sur.log_lik().ok_or_else(|| anyhow!("Failed to calculate marginal log likelihood!"))? {
            *sur.kernel_mut() = base_kernel;
            sur.refit()?
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ARDBacktrackingLineSearch<T>
where
    T: dtype,
{
    step_delta: faer::Col<T>,
    base_factor: T,
    step_n: usize,
}

impl<T> ARDBacktrackingLineSearch<T> 
where
    T: dtype
{
    pub fn new(delta: faer::Col<T>, base: T, n: usize) -> Self {
        Self { step_delta: delta, base_factor: base, step_n: n }
    }
}

impl<T, S> SurrogateTuning<T, S> for ARDBacktrackingLineSearch<T>
where
    T: dtype,
    S: GPSurrogate<T, KernType: ARD<T>>,
{
    fn tune(&self, sur: &mut S) -> Result<()> {
        let base_l = faer::ColRef::from_slice(sur.kernel().l()).to_owned();
        let base_lik = sur.log_lik().unwrap();

        for i in 0..self.step_n {
            let new_l = faer::zip!(&base_l, &self.step_delta).map(|faer::unzip!(base, delta)| {
                base.add(self.base_factor.powi(i as i32).mul(*delta))
            });
            sur.kernel_mut()
                .update_l(new_l.try_as_col_major().unwrap().as_slice());
            
            sur.refit()?;

            if sur.log_lik().unwrap() > base_lik {
                return Ok(());
            }
        }

        Ok(())
    }
}