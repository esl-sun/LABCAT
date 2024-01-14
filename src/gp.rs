#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

use faer_core::Mat;
use ndarray::{Array1, Array2, OwnedRepr};
use ndarray_linalg::error::LinalgError;
use ndarray_linalg::{CholeskyFactorized, FactorizeC, InverseC, Lapack, UPLO};

use crate::kernel::Kernel;
use crate::memory::ObservationIO;
use crate::ndarray_utils::Array2Utils;
use crate::utils::MatRefUtils;
use crate::{dtype, BayesianSurrogate, Surrogate, SurrogateMemory};

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
    pub fn new(d: usize) -> GP<T, K, M> {
        // GP { K: Array2::eye(3), alpha: Array1::zeros((3,)) }
        todo!()
    }

    fn test(&mut self) -> Result<(), LinalgError> {
        self.K = self.K.invc()?;
        Ok(())
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

impl<T, K, M> Surrogate<T> for GP<T, K, M>
where
    T: dtype + Lapack,
    K: Kernel<T>,
    M: ObservationIO<T>,
{
    fn refit<E>(&mut self, X: Mat<T>, Y: &[T]) -> Result<(), E> {
        self.mem = ObservationIO::new(self.dim);
        self.mem.append_mult(X.as_ref(), Y);

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

    fn probe(&self, x: &[T]) -> T {
        todo!()
    }
}

impl<T, K, M> SurrogateMemory<T> for GP<T, K, M>
where
    T: dtype + Lapack,
    K: Kernel<T>,
    M: ObservationIO<T>,
{
    fn memory(&self) -> &impl ObservationIO<T> {
        &self.mem
    }

    fn memory_mut(&mut self) -> &mut impl ObservationIO<T> {
        &mut self.mem
    }
}

impl<T, K, M> BayesianSurrogate<T> for GP<T, K, M>
where
    T: dtype + Lapack,
    K: Kernel<T>,
    M: ObservationIO<T>,
{
    fn probe_variance(&self, x: &[T]) -> T {
        todo!()
    }
}
