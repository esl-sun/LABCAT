#![allow(non_snake_case)]

use egobox_doe::{Lhs, LhsKind, SamplingMethod};
use faer::{Mat, MatRef};
use faer_ext::{IntoFaer, IntoNdarray};
use ndarray::{Array2, ArrayView2};
use ndarray_rand::rand::{self, Rng};
use num_traits::Zero;

use crate::{bounds::UpperLowerBounds, doe::DoE, dtype, utils::MatRefUtils};

#[derive(Debug, Clone)]
pub struct LHS<T>
where
    T: dtype,
{
    doe: Mat<T>,
}

impl<T> Default for LHS<T>
where
    T: dtype,
{
    fn default() -> Self {
        Self {
            doe: Mat::default(),
        }
    }
}

impl<T> DoE<T> for LHS<T>
where
    T: dtype + linfa::Float,
{
    // #[allow(refining_impl_trait)]
    fn build_DoE<B>(&mut self, n: usize, bounds: &B)
    where
        B: UpperLowerBounds<T>,
    {
        let m = bounds.as_mat();
        let b: ArrayView2<T> = m.as_ref().into_ndarray();

        let doe = if n.is_zero() {
            Array2::zeros((b.nrows(), 0))
        } else {
            Lhs::new(&b)
                .kind(LhsKind::Classic)
                .sample(n)
                .reversed_axes()
        };

        self.doe = doe.view().into_faer().to_owned_mat()
    }

    fn DoE(&self) -> MatRef<T> {
        self.doe.as_ref()
    }
}

#[derive(Debug, Clone)]
pub struct RandomSampling<T>
where
    T: dtype,
{
    doe: Mat<T>,
}

impl<T> Default for RandomSampling<T>
where
    T: dtype,
{
    fn default() -> Self {
        Self {
            doe: Mat::default(),
        }
    }
}

impl<T> DoE<T> for RandomSampling<T>
where
    T: dtype,
{
    fn build_DoE<B>(&mut self, n: usize, bounds: &B)
    where
        B: UpperLowerBounds<T>,
    {
        self.doe = Array2::from_shape_fn((bounds.dim(), n), |(i, _)| {
            T::from_f64(
                rand::thread_rng()
                    .gen_range(bounds.lb()[i].to_f64().unwrap()..bounds.ub()[i].to_f64().unwrap()),
            )
            .unwrap() //TODO: FIX UNWRAPS
        })
        .view()
        .into_faer()
        .to_owned_mat()
    }

    fn DoE(&self) -> MatRef<T> {
        self.doe.as_ref()
    }
}
