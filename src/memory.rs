#![allow(non_snake_case)]

use faer_core::{Col, Mat, MatRef};
use num_traits::FromPrimitive;
use ord_subset::{OrdSubset, OrdSubsetIterExt};

use crate::{
    dtype,
    utils::{MatMutUtils, MatRefUtils},
};

pub trait ObservationIO<T>
where
    T: dtype,
{
    fn new(d: usize) -> Self;
    fn dim(&self) -> usize;
    fn n(&self) -> usize;
    fn X(&self) -> MatRef<T>;
    fn Y(&self) -> &[T];
    fn append(&mut self, x: &[T], y: T);
    fn append_mult(&mut self, X: MatRef<T>, Y: &[T]);
    fn discard(&mut self, i: usize);
    fn discard_mult(&mut self, idx: Vec<usize>);
    fn discard_all(&mut self);
}

pub trait ObservationMaxMin<T>: ObservationIO<T>
where
    T: dtype,
{
    fn max(&self) -> Option<(usize, &[T], &T)>;
    fn min(&self) -> Option<(usize, &[T], &T)>;
    fn max_quantile(&self, gamma: &T) -> (Mat<T>, Mat<T>);
    fn min_quantile(&self, gamma: &T) -> (Mat<T>, Mat<T>);
}

pub trait ObservationMean<T>: ObservationIO<T>
where
    T: dtype + FromPrimitive,
{
    /// Returns the [arithmetic mean] x̅ of all elements in the array:
    ///
    /// ```text
    ///     1   n
    /// x̅ = ―   ∑ xᵢ
    ///     n  i=1
    /// ```
    ///
    /// If the array is empty, `None` is returned.
    ///
    /// **Panics** if `T::from_usize()` fails to convert the number of elements in the array.
    ///
    /// [arithmetic mean]: https://en.wikipedia.org/wiki/Arithmetic_mean
    fn Y_mean(&self) -> Option<T> {
        
        let n = self.Y().len();
        if n == 0 {
            None
        } else {
            let n = T::from_usize(n)
                .expect("Converting number of elements to `T` must not fail.");
            Some(
                self.Y()
                .iter()
                .fold(T::zero(), |acc, elem| T::add(acc, *elem))/ n
                )
        }
    }

    fn X_mean(&self) -> Col<T> {
        Col::<T>::from_fn(self.dim(), |j| {
            self.X()
                .row_as_slice(j)
                .iter()
                .fold(T::zero(), |acc, elem| T::add(acc, *elem))
                / T::from_usize(self.n())
                .expect("Converting number of elements to `T` must not fail.")
        })
    }
}

pub trait ObservationTransform<T>: ObservationIO<T>
where
    T: dtype,
{
    fn X_prime(&self) -> Mat<T>; //TODO: May need to become owned refs
    fn Y_prime(&self) -> Vec<T>;
}

pub trait ObservationInputRescale<T>: ObservationIO<T>
where
    T: dtype,
{
    fn rescale(&mut self, l: &[T]);
    fn reset_scaling(&mut self);
}

pub trait ObservationOutputRescale<T>: ObservationIO<T>
where
    T: dtype,
{
    fn rescale(&mut self, l: T);
    fn reset_scaling(&mut self);
}

pub trait ObservationInputRecenter<T>: ObservationIO<T>
where
    T: dtype,
{
    fn recenter(&mut self, l: &[T]);
    fn reset_center(&mut self);
}

pub trait ObservationOutputRecenter<T>: ObservationIO<T>
where
    T: dtype,
{
    fn recenter(&mut self, l: T);
    fn reset_center(&mut self);
}

#[derive(Clone, Debug)]
pub struct BaseMemory<T>
where
    T: dtype,
{
    X: Mat<T>, // (d, n)
    Y: Col<T>, // (n, )
}

impl<T> Default for BaseMemory<T>
where
    T: dtype,
{
    fn default() -> Self {
        Self {
            X: Mat::default(),
            Y: Col::default(),
        }
    }
}

impl<T> ObservationIO<T> for BaseMemory<T>
where
    T: dtype,
{
    fn new(d: usize) -> Self {
        let mut X = Mat::new();
        unsafe { X.set_dims(d, 0) }

        Self { X, Y: Col::new() }
    }

    fn dim(&self) -> usize {
        self.X.nrows()
    }

    fn n(&self) -> usize {
        self.X.ncols()
    }

    fn X(&self) -> MatRef<T> {
        self.X.as_ref()
    }

    fn Y(&self) -> &[T] {
        self.Y.as_slice()
    }

    fn append(&mut self, x: &[T], y: T) {
        #[cfg(debug_assertions)]
        if x.len() != self.dim() {
            panic!("Dimensions of new input and memory do not match!");
        }

        let old_n = self.X.ncols();

        self.Y.resize_with(old_n + 1, |_| y);
        self.X.resize_with(self.dim(), old_n + 1, |i, _| x[i]);
    }

    fn append_mult(&mut self, X: MatRef<T>, Y: &[T]) {
        #[cfg(debug_assertions)]
        if X.nrows() != self.dim() {
            panic!(
                "Dimensions of new input ({}) and memory ({}) do not match!",
                X.nrows(),
                self.dim()
            );
        }

        #[cfg(debug_assertions)]
        if X.ncols() != Y.len() {
            panic!(
                "Number of observations in X ({}) and Y ({}) do not match!",
                X.ncols(),
                Y.len()
            );
        }

        let old_n = self.X.ncols();
        self.Y.resize_with(old_n + Y.len(), |i| Y[i - old_n]);
        self.X
            .resize_with(self.dim(), old_n + Y.len(), |i, j| *X.get(i, j - old_n));
    }

    fn discard(&mut self, i: usize) {
        // self.X.rem_cols(indices.clone());
        // self.Y.rem_at_indices(indices);
        todo!()
    }

    fn discard_mult(&mut self, idx: Vec<usize>) {
        // self.X.rem_cols(indices.clone());
        // self.Y.rem_at_indices(indices);
        todo!()
    }

    fn discard_all(&mut self) {
        let d = self.X.nrows();
        let mut X = Mat::new();
        unsafe { X.set_dims(d, 0) }
        self.X = X;
        self.Y = Col::new();
    }
}

impl<T> ObservationMaxMin<T> for BaseMemory<T>
where
    T: dtype + OrdSubset + FromPrimitive,
{
    fn max(&self) -> Option<(usize, &[T], &T)> {
        let (i, y_max) = self
            .Y()
            .into_iter()
            .enumerate()
            .ord_subset_max_by_key(|&(_, x)| x)?;
        let x_max = self.X.col_as_slice(i);
        Some((i, x_max, y_max))
    }

    fn min(&self) -> Option<(usize, &[T], &T)> {
        let (i, y_min) = self
            .Y()
            .into_iter()
            .enumerate()
            .ord_subset_min_by_key(|&(_, x)| x)?;
        let x_min = self.X.col_as_slice(i);
        Some((i, x_min, y_min))
    }

    fn max_quantile(&self, gamma: &T) -> (Mat<T>, Mat<T>) {
        
        #[cfg(debug_assertions)]
        if *gamma < T::zero() || *gamma > T::one() {
            panic!(
                "Supplied gamma ({:?}) must be int the interval [0, 1]!",
                gamma
            );
        }
        
        let n_upper = T::from_usize(self.n())
        .expect("Converting number of elements to `T` must not fail.").mul(*gamma).round().to_usize(); 
        
        let mut idx: Vec<_> = (0..self.n()).map(|id| (id, self.Y()[id])).collect();
        idx.sort_by(|(_, a), (_, b)| a.partial_cmp(b).expect("Partial cmp should not fail!"));

        

        todo!()
    }

    fn min_quantile(&self, gamma: &T) -> (Mat<T>, Mat<T>) {
        todo!()
    }
}

impl<T> ObservationMean<T> for BaseMemory<T> where
    T: dtype + FromPrimitive
{
}

////////////////////////////////////////////

struct LabcatMemory<T>
where
    T: dtype,
{
    base_mem: BaseMemory<T>,
    R: Mat<T>,
    R_inv: Mat<T>,
    S: Mat<T>,
    S_inv: Mat<T>,
    X_offset: Col<T>,
    y_offset: T,
    y_scale: T,
}

impl<T> ObservationIO<T> for LabcatMemory<T>
where
    T: dtype,
{
    fn new(d: usize) -> Self {
        Self {
            base_mem: BaseMemory::new(d),
            R: Mat::identity(d, d),
            R_inv: Mat::identity(d, d),
            S: Mat::identity(d, d),
            S_inv: Mat::identity(d, d),
            X_offset: Col::zeros(d),
            y_offset: T::zero(),
            y_scale: T::one(),
        }
    }

    fn dim(&self) -> usize {
        self.base_mem.dim()
    }

    fn n(&self) -> usize {
        self.base_mem.n()
    }

    fn X(&self) -> MatRef<T> {
        self.base_mem.X()
    }

    fn Y(&self) -> &[T] {
        self.base_mem.Y()
    }

    fn append(&mut self, x: &[T], y: T) {
        self.base_mem.append(x, y)
    }

    fn append_mult(&mut self, X: MatRef<T>, Y: &[T]) {
        self.base_mem.append_mult(X, Y)
    }

    fn discard(&mut self, i: usize) {
        self.base_mem.discard(i)
    }

    fn discard_mult(&mut self, idx: Vec<usize>) {
        self.base_mem.discard_mult(idx)
    }

    fn discard_all(&mut self) {
        self.base_mem.discard_all()
    }
}

impl<T> ObservationMean<T> for LabcatMemory<T> where
    T: dtype + FromPrimitive
{
}

impl<T> ObservationTransform<T> for LabcatMemory<T>
where
    T: dtype,
{
    fn X_prime(&self) -> Mat<T> {
        // self.R * self.S * self.base_mem.X() + self.X_offset
        todo!()
    }

    fn Y_prime(&self) -> Vec<T> {
        self.base_mem
            .Y()
            .iter()
            .map(|val| val.faer_mul(self.y_scale).faer_add(self.y_offset))
            .collect()
    }
}

impl<T> ObservationInputRescale<T> for LabcatMemory<T>
where
    T: dtype,
{
    fn rescale(&mut self, l: &[T]) {
        self.base_mem
            .X
            .col_chunks_mut(1)
            .for_each(|mut col| col.zip_apply_with_row_slice(l, |val, l| val / l));

        todo!()
    }

    fn reset_scaling(&mut self) {
        todo!()
    }
}
