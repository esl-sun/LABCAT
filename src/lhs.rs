#![allow(non_snake_case)]

use std::{marker::PhantomData, usize};

use egobox_doe::{Lhs, LhsKind, SamplingMethod};
use faer_ext::{IntoFaer, IntoNdarray};
use faer::{Mat, MatRef};
use ndarray::{Array2, ArrayView2};
use ndarray_rand::rand::{self, Rng};
use num_traits::Zero;

use crate::{doe::DoE, dtype, utils::MatRefUtils};

pub struct LHS<T>
where
    T: dtype,
{
    data_type: PhantomData<T>,
}

impl<T> Default for LHS<T>
where
    T: dtype,
{
    fn default() -> Self {
        Self {
            data_type: PhantomData,
        }
    }
}

impl<T> DoE<T> for LHS<T>
where
    T: dtype + linfa::Float,
{
    #[allow(refining_impl_trait)]
    // fn build_DoE(&self, n: usize, bounds: MatRef<T>) -> impl IntoFaer<Faer = MatRef<T>> {
    fn build_DoE(&self, n: usize, bounds: MatRef<T>) -> Mat<T> {
        let b: ArrayView2<T> = bounds.into_ndarray();

        let doe = if n.is_zero() {
            Array2::zeros((b.nrows(), 0))
        } else {
            Lhs::new(&b)
                .kind(LhsKind::Classic)
                .sample(n)
                .reversed_axes()
        };

        doe.view().into_faer().to_owned_mat()
    }
}

pub struct RandomSampling<T>
where
    T: dtype,
{
    data_type: PhantomData<T>,
}

impl<T> Default for RandomSampling<T>
where
    T: dtype,
{
    fn default() -> Self {
        Self {
            data_type: PhantomData,
        }
    }
}

impl<T> DoE<T> for RandomSampling<T>
where
    T: dtype + ndarray_rand::rand_distr::uniform::SampleUniform + PartialOrd,
{
    fn build_DoE(&self, n: usize, bounds: MatRef<T>) -> Mat<T> {
        let b: ArrayView2<T> = bounds.into_ndarray();

        Array2::from_shape_fn((b.nrows(), n), |(i, _)| {
            rand::thread_rng().gen_range(b[(i, 0)]..b[(i, 1)])
        })
        .view()
        .into_faer()
        .to_owned_mat()
    }
}
