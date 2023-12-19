#![allow(non_snake_case)]

use nalgebra::Scalar;
use ndarray::{Array1, Array2};
use ord_subset::OrdSubset;

use crate::{
    ndarray_utils::{Array1Utils, Array2Utils, ArrayBaseUtils},
    utils::{Matrix, Vector},
};

pub trait ObservationMemory<T>
where
    T: Scalar,
{
    fn X(&self) -> impl Into<Matrix<T>>;
    fn Y(&self) -> impl Into<Vector<T>>;
    fn append(&mut self, x: &Vector<T>, y: T);
    fn discard(&mut self, index: Vec<usize>);
}

pub trait MemoryMaxMin<T>
where
    T: Scalar + OrdSubset,
{
    fn max(&self) -> Option<(usize, &Vector<T>, T)>;
    fn min(&self) -> Option<(usize, &Vector<T>, T)>;
}

pub struct BaseMemory<T>
where
    T: Scalar,
{
    X: Array2<T>, // (d, n)
    Y: Array1<T>, // (n, )
}

impl<T> Default for BaseMemory<T>
where
    T: Scalar + Default,
{
    fn default() -> Self {
        Self {
            X: Array2::default((0, 0)),
            Y: Array1::default((0,)),
        }
    }
}

impl<T> ObservationMemory<T> for BaseMemory<T>
where
    T: Scalar + Clone,
{
    fn X(&self) -> impl Into<Matrix<T>> {
        let x = self.X.clone();
        x
    }

    fn Y(&self) -> impl Into<Vector<T>> {
        let x = self.Y.clone();
        x
    }

    fn append(&mut self, x: &Vector<T>, y: T) {
        todo!()
    }

    fn discard(&mut self, indices: Vec<usize>) {
        self.X.rem_cols(indices.clone());
        self.Y.rem_at_indices(indices);
    }
}

impl<T> MemoryMaxMin<T> for BaseMemory<T>
where
    T: Scalar + OrdSubset,
{
    fn max(&self) -> Option<(usize, &Vector<T>, T)> {
        let (i, y_max) = self.Y.indexed_max()?;
        let x = self.X.column(i);
        // Some((i, x, y_max))
        todo!()
    }

    fn min(&self) -> Option<(usize, &Vector<T>, T)> {
        todo!()
    }
}
