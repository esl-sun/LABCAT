use std::iter::once;

use anyhow::Result;
use faer::{unzip, zip, Col};

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
            prior_sigma: T::half() * T::half() * T::half(), //TODO: REWRTIE 0.1
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
            zip!(gp.alpha() * gp.alpha().transpose(), gp.K_inv()).map(|unzip!(a, k)| *a - *k);

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
        // self.log_lik_prior(&sur);
        // self.log_lik_jac(sur);

        *sur.kernel_mut().sigma_f_mut() = sur.memory().Y_std(T::zero());

        sur.refit()?;

        Ok(())
    }
}

// impl<T, S, A, B, D> SurrogateTune<T> for LABCAT<T, S, A, B, D>
// where
//     T: dtype + OrdSubset,
//     S: SurrogateIO<T>
//         + GPSurrogate<T>
//         + Kernel<T, KernType: ARD<T>>
//         + Memory<T, MemType = LabcatMemory<T>>
//         + Refit<T>,
//     A: AcqFunction<T, S>,
//     B: Bounds<T>,
//     D: DoE<T>,
// { }
