use egobox_doe::{Lhs, LhsKind, SamplingMethod};
use ndarray::{Array1, Array2, ArrayView1, Axis, array};
use ndarray_rand::rand_distr::num_traits::Zero;
use rand::Rng;

use crate::{
    bound_types::{BoundTrait, BoundType},
    f_,
};

#[derive(Debug, Clone)]
pub struct ArrayBounds {
    bounds_arr: Array2<f_>,
}

impl ArrayBounds {
    pub fn new(bounds: Vec<BoundType>) -> ArrayBounds {
        let mut bounds_arr = Array2::zeros((0, 2));

        bounds.iter().for_each(|bound| {
            bounds_arr
                .append(
                    Axis(0),
                    bound.bound_arr().into_shape((1, 2)).unwrap().view(),
                )
                .unwrap()
        });

        ArrayBounds { bounds_arr }
    }

    pub fn new_continuous(d: usize, upper: f_, lower: f_) -> ArrayBounds {
        let bounds_arr = Array2::from_shape_fn((d, 2), |(_, j)| match j {
            j if j == 0 => lower,
            j if j == 1 => upper,
            _ => panic!("Should never trigger"),
        });

        ArrayBounds { bounds_arr }
    }

    pub fn dim(&self) -> usize {
        self.bounds_arr.nrows()
    }

    pub fn bounds_arr(&self) -> &Array2<f_> {
        &self.bounds_arr
    }

    pub fn midpoint(&self) -> Array1<f_> {
        let b = &self.bounds_arr;
        Array1::from_shape_fn((b.nrows(),), |i| (b.row(i)[1] + b.row(i)[0]) / 2.0)
    }

    /// TODO: rename func. gives 0.5 * sidelength for each dim
    pub fn axes_len(&self) -> Array1<f_> {
        let b = &self.bounds_arr;
        Array1::from_shape_fn((b.nrows(),), |i| (b.row(i)[1] - b.row(i)[0]).abs() / 2.0)
    }

    pub fn inside(&self, x: ArrayView1<f_>) -> bool {
        if x.len() != self.dim() {
            panic!("Input point dim does not match bounds dim!")
        };

        self.bounds_arr()
            .view()
            .rows()
            .into_iter()
            .zip(x.iter())
            .all(|(bound, x)| x <= &bound[1] && x >= &bound[0])

    }

    pub fn random_sample(&self, n: usize) -> Array2<f_> {
        Array2::from_shape_fn((self.bounds_arr.nrows(), n), |(i, _)| {
            rand::thread_rng().gen_range(self.bounds_arr()[(i, 0)]..self.bounds_arr()[(i, 1)])
        })
    } 

    pub fn LHS_sample(&self, n: usize) -> Array2<f_> {
        if n.is_zero() {
            Array2::zeros((self.bounds_arr.nrows(), 0))
        } else {
            Lhs::new(self.bounds_arr())
                .kind(LhsKind::Classic)
                .sample(n)
                .reversed_axes()
        }
    }
}
