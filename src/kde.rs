#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

use std::marker::PhantomData;

use nalgebra::Scalar;
use ndarray_linalg::error::LinalgError;

use crate::kernel::{Kernel, ARD};
use crate::Surrogate;

pub struct kde<T, K>
where
    T: Scalar,
    K: Kernel<T> + ARD<T>,
{
    data_type: PhantomData<T>,
    kernels: Vec<K>,
}

impl<T, K> Default for kde<T, K>
where
    T: Scalar,
    K: Kernel<T> + ARD<T>,
{
    fn default() -> Self {
        Self {
            data_type: PhantomData,
            kernels: vec![],
        }
    }
}

impl<T, K> Surrogate<T> for kde<T, K>
where
    T: Scalar,
    K: Kernel<T> + ARD<T>,
{
    fn fit<E>(&mut self, X: crate::utils::Matrix<T>, Y: crate::utils::Vector<T>) -> Result<(), E> {
        todo!()
    }

    fn probe(&self, x: &crate::utils::Vector<T>) -> T {
        todo!()
    }
}
