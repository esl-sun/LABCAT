#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

use anyhow::Result;
use ndarray::{Array1, Array2, OwnedRepr};
use ndarray_linalg::{CholeskyFactorized, FactorizeC, InverseC, Lapack, SolveC, UPLO};
use num_traits::real::Real;

use crate::kernel::{BaseKernel, BayesianKernel};
use crate::memory::{ObservationIO, ObservationMean};
use crate::ndarray_utils::{Array2Utils, ArrayView2Utils, RowColIntoNdarray};
use crate::{dtype, BayesianSurrogateIO, Kernel, Memory, Refit, RefitWith, SurrogateIO};

pub trait GPSurrogate<T>: SurrogateIO<T> + BayesianSurrogateIO<T> + Kernel<T>
where
    T: dtype,
{
}

// #[derive(Clone, Debug)]
pub struct GP<T, K, M>
where
    T: dtype,
    K: BaseKernel<T>,
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

// impl<T, K, M> GP<T, K, M>
// where
//     T: dtype + Lapack,
//     K: BaseKernel<T>,
//     M: ObservationIO<T>,
// {}

impl<T, K, M> Default for GP<T, K, M>
where
    Self: SurrogateIO<T>,
    T: dtype + Lapack,
    K: BaseKernel<T>,
    M: ObservationIO<T>,
{
    fn default() -> Self {
        Self::new(0)
    }
}

impl<T, K, M> GPSurrogate<T> for GP<T, K, M>
where
    Self: SurrogateIO<T> + BayesianSurrogateIO<T>,
    T: dtype + Lapack,
    K: BaseKernel<T>,
    M: ObservationIO<T>,
{
}

impl<T, K, M> SurrogateIO<T> for GP<T, K, M>
where
    T: dtype + Lapack,
    K: BaseKernel<T>,
    M: ObservationIO<T> + ObservationMean<T>,
{
    fn new(d: usize) -> Self {
        Self {
            dim: d,
            kernel: BaseKernel::new(d),
            K: Array2::eye(d),
            Kinv: Array2::eye(d),
            L: Array2::eye(d)
                .factorizec(UPLO::Lower)
                .expect("Should never fail during init."),
            alpha: Array1::zeros((d,)),
            mem: M::new(d),
        }
    }

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

impl<T, K, M> Refit<T> for GP<T, K, M>
where
    T: dtype + Lapack,
    K: BaseKernel<T>,
    M: ObservationIO<T> + ObservationMean<T>,
{
    fn refit(&mut self) -> Result<()>
    where
        Self: Memory<T>,
    {
        self.K = Array2::zeros((self.mem.n(), self.mem.n()))
            .map_UPLO(UPLO::Lower, |(i, j)| {
                self.kernel
                    .k(self.mem.X().col_as_slice(i), self.mem.X().col_as_slice(j))
            })
            .fill_with_UPLO(UPLO::Lower);

        self.L = self.K.factorizec(UPLO::Lower).unwrap();
        self.Kinv = self.L.invc().unwrap();

        self.alpha = Array1::from_iter(self.mem.Y()).mapv(|val| *val - self.mem.Y_mean().unwrap());
        self.L.solvec_inplace(&mut self.alpha)?;

        Ok(())
    }
}

impl<T, K, M, MI> RefitWith<T, MI> for GP<T, K, M>
where
    T: dtype + Lapack,
    K: BaseKernel<T>,
    M: ObservationIO<T> + ObservationMean<T>,
    MI: ObservationIO<T> + ObservationMean<T>,
{
    fn refit_from(&mut self, mem: &MI) -> Result<()> {
        self.mem = ObservationIO::new(self.dim);
        self.mem.append_mult(mem.X().as_ref(), mem.Y());

        self.refit()
    }
}

impl<T, K, M> Memory<T> for GP<T, K, M>
where
    T: dtype + Lapack,
    K: BaseKernel<T>,
    M: ObservationIO<T>,
{
    type MemType = M;

    fn memory(&self) -> &M {
        &self.mem
    }

    fn memory_mut(&mut self) -> &mut M {
        &mut self.mem
    }
}

impl<T, K, M> Kernel<T> for GP<T, K, M>
where
    T: dtype + Lapack,
    K: BaseKernel<T>,
    M: ObservationIO<T>,
{
    type KernType = K;

    fn kernel(&self) -> &Self::KernType {
        &self.kernel
    }

    fn kernel_mut(&mut self) -> &mut Self::KernType {
        &mut self.kernel
    }
}

//TODO: Multiple calls to into_ndarray as as_1D, two calls to k_diag() with probe() and probe_variance()
impl<T, K, M> BayesianSurrogateIO<T> for GP<T, K, M>
where
    T: dtype + Lapack,
    K: BaseKernel<T> + BayesianKernel<T>,
    M: ObservationIO<T> + ObservationMean<T>,
{
    fn probe_variance(&self, x: &[T]) -> Option<T> {
        let k_diag = self.kernel.k_diag(self.memory().X().as_ref(), x);

        let v = self
            .L
            .solvec(&k_diag.as_ref().into_ndarray().as_1D()?)
            .ok()?;

        let sigma = Real::sqrt(Real::abs(
            self.kernel
                .k(x, x)
                .sub(k_diag.as_ref().into_ndarray().as_1D()?.dot(&v))
                .add(Real::powi(*self.kernel.sigma_n(), 2)),
        ));

        Some(sigma)
    }
}
