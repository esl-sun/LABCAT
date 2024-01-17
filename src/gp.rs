#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

use ndarray::{Array1, Array2, OwnedRepr};
use ndarray_linalg::{CholeskyFactorized, FactorizeC, InverseC, Lapack, SolveC, UPLO};
use num_traits::real::Real;

use crate::kernel::{BayesianKernel, Kernel};
use crate::memory::{ObservationIO, ObservationMean};
use crate::ndarray_utils::{Array2Utils, ArrayView2Utils, RowColIntoNdarray};
use crate::{dtype, BayesianSurrogate, Memory, Surrogate};

pub struct GP<T, K, M>
where
    T: dtype,
    K: Kernel<T>,
    M: ObservationIO<T>,
{
    dim: usize,
    kernel: K,
    K: Array2<T>,
    Kinv: Array2<T>,
    L: CholeskyFactorized<OwnedRepr<T>>,
    alpha: Array1<T>,
    mem: M,
}

impl<T, K, M> GP<T, K, M>
where
    T: dtype + Lapack,
    K: Kernel<T>,
    M: ObservationIO<T>,
{
    pub fn new(_: usize) -> GP<T, K, M> {
        // GP { K: Array2::eye(3), alpha: Array1::zeros((3,)) }
        todo!()
    }

}

impl<T, K, M> Default for GP<T, K, M>
where
    T: dtype + Lapack,
    K: Kernel<T>,
    M: ObservationIO<T>,
{
    fn default() -> Self {
        Self {
            dim: 0,
            kernel: Kernel::new(0),
            K: Array2::eye(0),
            Kinv: Array2::eye(0),
            L: Array2::eye(0)
                .factorizec(UPLO::Lower)
                .expect("Should never fail during init."),
            alpha: Array1::zeros((0,)),
            mem: ObservationIO::new(0),
        }
    }
}

impl<T, K, M> Surrogate<T, M> for GP<T, K, M>
where
    T: dtype + Lapack,
    K: Kernel<T>,
    M: ObservationIO<T> + ObservationMean<T>,
{
    fn probe(&self, x: &[T]) -> Option<T> {
        Some(
            self.kernel
                .k_diag(self.memory().X().as_ref(), x)
                .as_ref()
                .into_ndarray()
                .as_1D()?
                .dot(&self.alpha)
                + self.memory().Y_mean()?,
        )
    }
}

impl<T, K, MI, MO> Memory<T, MI, MO> for GP<T, K, MO>
where
    T: dtype + Lapack,
    K: Kernel<T>,
    MI: ObservationIO<T>,
    MO: ObservationIO<T> + ObservationMean<T>,
{
    fn refit<E>(&mut self, mem: &MI) -> Result<(), E> {
        self.mem = ObservationIO::new(self.dim);
        self.mem.append_mult(mem.X().as_ref(), mem.Y());

        self.K = Array2::zeros((self.mem.n(), self.mem.n()))
            .map_UPLO(UPLO::Lower, |(i, j)| {
                self.kernel
                    .k(self.mem.X().col_as_slice(i), self.mem.X().col_as_slice(j))
            })
            .fill_with_UPLO(UPLO::Lower);

        self.L = self.K.factorizec(UPLO::Lower).unwrap();
        self.Kinv = self.L.invc().unwrap();

        // let mut y_m = self.mem.y_m();
        // // self.L.ln_detc()
        // self.L.solvec_inplace(&mut y)?;
        // self.alpha = y.into_col();

        Ok(())
    }

    fn memory(&self) -> &MO {
        &self.mem
    }

    fn memory_mut(&mut self) -> &mut MO {
        &mut self.mem
    }
}

//TODO: Multiple calls to into_ndarray as as_1D, two calls to k_diag() with probe() and probe_variance()
impl<T, K, M> BayesianSurrogate<T, M> for GP<T, K, M>
where
    T: dtype + Lapack,
    K: Kernel<T> + BayesianKernel<T>,
    M: ObservationIO<T> + ObservationMean<T>,
{
    fn probe_variance(&self, x: &[T]) -> Option<T> {
        let k_diag = self
            .kernel
            .k_diag(self.memory().X().as_ref(), x);

        let v = self.L.solvec(&k_diag.as_ref().into_ndarray().as_1D()?).ok()?;

        let sigma = Real::sqrt(Real::abs(
            self.kernel
                .k(x, x)
                .sub(k_diag.as_ref().into_ndarray().as_1D()?.dot(&v))
                .add(Real::powi(*self.kernel.sigma_n(), 2)),
        ));

        Some(sigma)
    }
}
