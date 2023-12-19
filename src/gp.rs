#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

use ndarray::{Array1, Array2};
use ndarray_linalg::error::LinalgError;
use ndarray_linalg::{InverseC, Lapack, Scalar};

use crate::kernel::{Kernel, ARD};
use crate::{BayesianSurrogate, Surrogate};

pub struct GP<T, K>
where
    T: Scalar,
    K: Kernel<T>,
{
    kern: K,
    K: Array2<T>,
    alpha: Array1<T>,
}

impl<T, K> GP<T, K>
where
    T: Scalar + Lapack,
    K: Kernel<T>,
{
    pub fn new(d: usize) -> GP<T, K> {
        // GP { K: Array2::eye(3), alpha: Array1::zeros((3,)) }
        todo!()
    }

    fn test(&mut self) -> Result<(), LinalgError> {
        self.K = self.K.invc()?;
        Ok(())
    }
}

impl<T, K> Default for GP<T, K>
where
    T: Scalar + Lapack,
    K: Kernel<T>,
{
    fn default() -> Self {
        Self {
            kern: Kernel::new(2),
            K: Array2::eye(0),
            alpha: Array1::zeros((0,)),
        }
    }
}

impl<T, K> Surrogate<T> for GP<T, K>
where
    T: Scalar + Lapack,
    K: Kernel<T>,
{

    fn fit<E>(&mut self, X: crate::utils::Matrix<T>, Y: crate::utils::Vector<T>) -> Result<(), E> {
        // self.K.map
        self.K = match self.K.invc() {
            Ok(arr) => arr,
            Err(e) => return Err(e),
        };
        // self.K = self.K.invc()?;
        Ok(())
    }

    fn probe(&self, x: &crate::utils::Vector<T>) -> T {
        todo!()
    }
}

impl<T, K> BayesianSurrogate<T> for GP<T, K>
where
    T: Scalar + Lapack,
    K: Kernel<T>,
{
    fn probe_variance(&self, x: &crate::utils::Vector<T>) -> T {
        todo!()
    }
}
