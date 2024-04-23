use std::iter::once;

use anyhow::Result;
use faer::{unzipped, zipped, Col};
use ord_subset::OrdSubset;

use crate::{
    bounds::Bounds,
    doe::DoE,
    dtype,
    ei::AcqFunction,
    gp::GPSurrogate,
    kernel::{BayesianKernel, ARD},
    tune::{SurrogateTune, Tune},
    utils::{DtypeUtils, MatRefUtils},
    Kernel, Memory, Refit, SurrogateIO,
};

use super::{memory::LabcatMemory, LABCAT};

pub struct LABCAT_GPTune {
    // prior_sigma: T, // GIVING ISSUES
}

impl LABCAT_GPTune {
    fn prior_sigma<T: dtype>() -> Option<T> {
        T::from_f64(0.1)
    }

    fn log_lik_prior<T, S>(gp: &S) -> Option<T>
    where
        T: dtype,
        S: SurrogateIO<T> + GPSurrogate<T, KernType: ARD<T> + BayesianKernel<T>>,
    {
        Some(
            gp.log_lik()?
                + T::half().neg()
                    * Self::prior_sigma::<T>()?.powi(2).recip()
                    * gp.kernel()
                        .l()
                        .iter()
                        .fold(T::zero(), |acc, l| acc + l.ln() * l.ln()),
        )
    }
    fn log_lik_jac<T, S>(gp: &S) -> Option<Col<T>>
    where
        T: dtype,
        S: SurrogateIO<T> + GPSurrogate<T, KernType: ARD<T> + BayesianKernel<T>>,
    {
        let inner =
            zipped!(gp.alpha() * gp.alpha().transpose(), gp.K_inv()).map(|unzipped!(a, k)| *a - *k);

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

impl<T, S> Tune<T, S> for LABCAT_GPTune
where
    T: dtype,
    S: SurrogateIO<T> + GPSurrogate<T, KernType: ARD<T> + BayesianKernel<T>>,
{
    fn tune(sur: &mut S) -> Result<()> {
        Self::log_lik_prior(sur);
        todo!()
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
