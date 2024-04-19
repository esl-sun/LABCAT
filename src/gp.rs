#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

use anyhow::Result;
use faer::linalg::zip::Diag;
use faer::solvers::{Cholesky, SolverCore, SpSolver};
use faer::{unzipped, zipped, Col, ColRef, Mat, MatRef};

// use ndarray::{Array1, Array2, OwnedRepr};
// use ndarray_linalg::{CholeskyFactorized, FactorizeC, InverseC, Lapack, SolveC, UPLO};
// use num_traits::real::Real;

use crate::kernel::{BaseKernel, BayesianKernel};
use crate::memory::{ObservationIO, ObservationMean};
// use crate::ndarray_utils::{Array1IntoFaerRowCol, Array2Utils, ArrayView2Utils, RowColIntoNdarray};
use crate::utils::MatMutUtils;
use crate::{dtype, BayesianSurrogateIO, Kernel, Memory, Refit, RefitWith, SurrogateIO};

pub trait GPSurrogate<T>: 
    SurrogateIO<T> + BayesianSurrogateIO<T> + Kernel<T> + Memory<T, MemType: ObservationMean<T>>
where
    T: dtype,
{
    fn K(&self) -> MatRef<T>;
    fn L(&self) -> &Cholesky<T>;
    fn alpha(&self) -> ColRef<T>;
    fn log_lik(&self) -> Option<T> {
        let y_mean = self.memory().Y_mean()?;

        Some(
            (T::one() + T::one()).recip().neg() //-0.5
            * zipped!(faer::col::from_slice::<T>(self.memory().Y()), self.alpha())
                .map(|unzipped!(y, a)| (y.read() - y_mean) * a.read()).sum() // (y - y_mean).dot(alpha)
            - zipped!(self.L().compute_l().diagonal().column_vector())
                .map(|unzipped!(val)| val.ln()).sum(), // trace(ln(L)), precompute trace?
        )
    }

    // fn prior(&self) -> Option<T> 
    // where 
    //     Self: Kernel<T, KernType: ARD<T>>;


    // fn chol_solve(&self, x: &[T]) -> Result<Col<T>>;
    // fn chol_solve_inplace(&self, x: &mut Col<T>) -> Result<()>;
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
    K: Mat<T>,
    Kinv: Mat<T>,
    L: Cholesky<T>,
    alpha: Col<T>,
    mem: M,
}

impl<T, K, M> Default for GP<T, K, M>
where
    Self: SurrogateIO<T>,
    T: dtype,
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
    T: dtype,
    K: BaseKernel<T>,
    M: ObservationIO<T> + ObservationMean<T>,
{
    fn K(&self) -> MatRef<T> {
        self.K.as_ref()
    }

    fn L(&self) -> &Cholesky<T> {
        &self.L
    }

    fn alpha(&self) -> ColRef<T> {
        self.alpha.as_ref()
    }
}

impl<T, K, M> SurrogateIO<T> for GP<T, K, M>
where
    T: dtype,
    K: BaseKernel<T>,
    M: ObservationIO<T> + ObservationMean<T>,
{
    fn new(d: usize) -> Self {
        Self {
            dim: d,
            kernel: BaseKernel::new(d),
            K: Mat::identity(d, d),
            Kinv: Mat::identity(d, d),
            L: Mat::<T>::identity(d, d)
                .cholesky(faer::Side::Lower)
                .expect("Should never fail during init."),
            alpha: Col::zeros(d),
            mem: M::new(d),
        }
    }

    fn probe(&self, x: &[T]) -> Option<T> {
        Some(
            self.kernel.k_diag(self.memory().X().as_ref(), x) * self.alpha.as_ref()
                + self.memory().Y_mean()?,
        )
    }
}

impl<T, K, M> BayesianSurrogateIO<T> for GP<T, K, M>
where
    T: dtype,
    K: BaseKernel<T> + BayesianKernel<T>,
    M: ObservationIO<T> + ObservationMean<T>,
{
    fn probe_variance(&self, x: &[T]) -> Option<T> {
        let k_diag = self.kernel.k_diag(self.memory().X().as_ref(), x);

        let v = self.L.solve(k_diag.transpose());

        Some(
            self.kernel
                .k(x, x)
                .sub(k_diag * v)
                .add(self.kernel.sigma_n().powi(2))
                .abs(),
        )
    }
}

impl<T, K, M> Refit<T> for GP<T, K, M>
where
    T: dtype, //+ Lapack,
    K: BaseKernel<T>,
    M: ObservationIO<T> + ObservationMean<T>,
{
    fn refit(&mut self) -> Result<()>
    where
        Self: Memory<T>,
    {
        let n = self.mem.n();

        self.K.resize_with(n, n, |_, _| T::zero());

        zipped!(&mut self.K).for_each_triangular_lower_with_index(
            Diag::Include,
            |i, j, unzipped!(mut v)| {
                v.write(
                    self.kernel
                        .k(self.mem.X().col_as_slice(i), self.mem.X().col_as_slice(j)),
                )
            },
        );

        self.K.fill_with_side(faer::Side::Lower);

        self.L = self.K.cholesky(faer::Side::Lower)?;
        self.Kinv = self.L.inverse();

        self.alpha.resize_with(n, |_| T::zero());

        let y_mean = self.mem.Y_mean().unwrap_or(T::zero());
        zipped!(&mut self.alpha)
            .for_each_with_index(|i, unzipped!(mut v)| *v = self.mem.Y()[i] - y_mean);
        self.L.solve_in_place(&mut self.alpha);

        Ok(())
    }
}

impl<T, K, M, MI> RefitWith<T, MI> for GP<T, K, M>
where
    T: dtype,
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
    T: dtype,
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
    T: dtype,
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
