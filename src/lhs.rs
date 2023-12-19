#![allow(non_snake_case)]

use std::{marker::PhantomData, usize};

use egobox_doe::{Lhs, LhsKind, SamplingMethod};
use nalgebra::Scalar;
use ndarray::Array2;
use ndarray_rand::rand::{self, Rng};
use num_traits::Zero;

use crate::utils::Matrix;

pub trait DoE<T>: Default
where
    T: Scalar,
{
    // fn build_DoE(&self, bounds: ) -> Matrix<T>;
    fn build_DoE(&self, n: usize, bounds: Matrix<T>) -> impl Into<Matrix<T>>;
}

pub struct LHS<T>
where
    T: Scalar,
{
    data_type: PhantomData<T>,
}

impl<T> Default for LHS<T>
where
    T: Scalar,
{
    fn default() -> Self {
        Self {
            data_type: PhantomData,
        }
    }
}

impl<T> DoE<T> for LHS<T>
where
    T: Scalar + linfa::Float,
{
    #[allow(refining_impl_trait)]
    fn build_DoE(&self, n: usize, bounds: Matrix<T>) -> impl Into<Matrix<T>> {
        let b: Array2<T> = bounds.into();

        if n.is_zero() {
            Array2::zeros((b.nrows(), 0))
        } else {
            Lhs::new(&b)
                .kind(LhsKind::Classic)
                .sample(n)
                .reversed_axes()
        }
    }
}

pub struct RandomSampling<T>
where
    T: Scalar,
{
    data_type: PhantomData<T>,
}

impl<T> Default for RandomSampling<T>
where
    T: Scalar,
{
    fn default() -> Self {
        Self {
            data_type: PhantomData,
        }
    }
}

impl<T> DoE<T> for RandomSampling<T>
where
    T: Scalar + ndarray_rand::rand_distr::uniform::SampleUniform + PartialOrd + Copy,
{
    fn build_DoE(&self, n: usize, bounds: Matrix<T>) -> impl Into<Matrix<T>> {
        let b: Array2<T> = bounds.into();
        Array2::from_shape_fn((b.nrows(), n), |(i, _)| {
            rand::thread_rng().gen_range(b[(i, 0)]..b[(i, 1)])
        })
    }
}
