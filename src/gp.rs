#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

use anyhow::Result;
use faer::linalg::zip::Diag;
// use faer::linalg::cholesky::llt::factor::
use faer::linalg::solvers::{DenseSolveCore, Llt};
use faer::prelude::*;
// use faer::solvers::{Cholesky, SolverCore, SpSolver};
use faer::{unzip, zip, Col, ColRef, Mat, MatRef};

use crate::kernel::{BaseKernel, BayesianKernel};
use crate::memory::{ObservationIO, ObservationMean};
// use crate::ndarray_utils::{Array1IntoFaerRowCol, Array2Utils, ArrayView2Utils, RowColIntoNdarray};
use crate::utils::{MatMutUtils, MatRefUtils};
use crate::{
    dtype, kernel::Kernel, memory::Memory, BayesianSurrogateIO, Refit, RefitWith, SurrogateIO,
};

pub trait GPSurrogate<T>:
    SurrogateIO<T>
    + BayesianSurrogateIO<T>
    + Kernel<T>
    + Memory<T, MemType: ObservationMean<T>>
    + Refit<T>
where
    T: dtype,
{
    fn K(&self) -> MatRef<T>;
    fn K_inv(&self) -> MatRef<T>;
    fn L(&self) -> MatRef<T>;
    fn alpha(&self) -> ColRef<T>;
    fn log_lik(&self) -> Option<T> {
        let y_mean = self.memory().Y_mean()?;

        Some(
            (T::one() + T::one()).recip().neg() //-0.5
            * zip!(faer::ColRef::from_slice(self.memory().Y()), self.alpha())
                .map(|unzip!(y, a)| (*y - y_mean) * *a).sum() // (y - y_mean).dot(alpha)
            - zip!(self.L().diagonal().column_vector())
                .map(|unzip!(val)| val.ln()).sum(), // trace(ln(L)), precompute trace?
        )
    }

    // fn prior(&self) -> Option<T>
    // where
    //     Self: Kernel<T, KernType: ARD<T>>;

    // fn chol_solve(&self, x: &[T]) -> Result<Col<T>>;
    // fn chol_solve_inplace(&self, x: &mut Col<T>) -> Result<()>;
}

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
    L: Llt<T>,
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

    fn K_inv(&self) -> MatRef<T> {
        self.Kinv.as_ref()
    }

    fn L(&self) -> MatRef<T> {
        self.L.L()
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
            K: Mat::identity(0, 0),
            Kinv: Mat::identity(0, 0),
            L: Mat::<T>::identity(0, 0)
                .llt(faer::Side::Lower)
                .expect("Should never fail during init."),
            alpha: Col::zeros(0),
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

        //Calc lower triangular of K
        #[allow(unused_mut)]
        zip!(&mut self.K).for_each_triangular_lower_with_index(
            Diag::Include,
            |i, j, unzip!(mut v)| {
                *v = self.kernel.k(
                    self.mem.X().col(i).try_as_col_major().unwrap().as_slice(),
                    self.mem.X().col(j).try_as_col_major().unwrap().as_slice(),
                )
            },
        );

        //Fill upper triangular to ensure symmetry
        self.K.fill_with_side(faer::Side::Lower);
        // dbg!("HERE");
        self.L = self.K.llt(faer::Side::Lower)?;
        self.Kinv = self.L.inverse();

        self.alpha.resize_with(n, |_| T::zero());

        let y_mean = self.mem.Y_mean().unwrap_or(T::zero());
        #[allow(unused_mut)]
        zip!(self.alpha.rb_mut())
            .for_each_with_index(|i, unzip!(mut v)| *v = self.mem.Y()[i] - y_mean);
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

    fn memory(&self) -> &Self::MemType {
        &self.mem
    }

    fn memory_mut(&mut self) -> &mut Self::MemType {
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
