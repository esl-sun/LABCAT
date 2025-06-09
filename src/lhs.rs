#![allow(non_snake_case)]

use egobox_doe::Lhs;
use egobox_doe::{LhsKind, SamplingMethod};
use faer::{Mat, MatRef};
use ndarray::Array2;
// use ndarray::{Array2, ArrayView2};
// use ndarray_rand::rand::{self, Rng};
use num_traits::Zero;

use crate::{bounds::UpperLowerBounds, doe::DoE, dtype};

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
        Self { doe: Mat::new() }
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
        self.doe = if n.is_zero() {
            Mat::zeros(bounds.dim(), 0)
        } else {
            let v = bounds
                .lb_ub()
                .flat_map(|(&lb, &ub)| vec![lb, ub])
                .collect::<Vec<_>>();
            let b = Array2::from_shape_vec((bounds.dim(), 2), v).expect("Should never fail!");

            let lhs = Lhs::new(&b)
                .kind(LhsKind::Classic)
                .sample(n)
                .reversed_axes();
            faer::Mat::from_fn(lhs.nrows(), lhs.ncols(), |i, j| lhs[(i, j)])
        };
    }

    fn DoE(&self) -> MatRef<T> {
        self.doe.as_ref()
    }
}
