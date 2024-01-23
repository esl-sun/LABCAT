#![allow(non_snake_case)]

use faer_core::{Mat, MatRef, Row};
use ord_subset::{OrdSubset, OrdSubsetIterExt};

use crate::{dtype, utils::MatUtils};

pub trait ObservationIO<T>
where
    T: dtype,
{
    fn new(d: usize) -> Self;
    fn dim(&self) -> usize;
    fn n(&self) -> usize;
    fn X(&self) -> &Mat<T>;
    fn Y(&self) -> &[T];
    fn append(&mut self, x: &[T], y: T);
    fn append_mult(&mut self, X: MatRef<T>, Y: &[T]);
    fn discard(&mut self, i: usize);
    fn discard_mult(&mut self, idx: Vec<usize>);
    fn discard_all(&mut self);
}

pub trait ObservationMaxMin<T>: ObservationIO<T>
where
    T: dtype + OrdSubset,
{
    fn max(&self) -> Option<(usize, &[T], &T)> {
        let (i, y_max) = self
            .Y()
            .iter()
            .enumerate()
            .ord_subset_max_by_key(|&(_, x)| x)?;
        let x_max = self.X().col_as_slice(i);
        Some((i, x_max, y_max))
    }

    fn min(&self) -> Option<(usize, &[T], &T)> {
        let (i, y_min) = self
            .Y()
            .iter()
            .enumerate()
            .ord_subset_min_by_key(|&(_, x)| x)?;
        let x_min = self.X().col_as_slice(i);
        Some((i, x_min, y_min))
    }

    /// (max_quantile, min_quantile)  TODO: expand doc
    fn max_quantile(&self, gamma: &T) -> (Self, Self)
    where
        Self: Sized + Clone,
    {
        #[cfg(debug_assertions)]
        if *gamma < T::zero() || *gamma > T::one() {
            panic!(
                "Supplied gamma ({:?}) must be int the interval [0, 1]!",
                gamma
            );
        }

        let n_upper = T::from_usize(self.n())
            .expect("Converting number of elements to `T` must not fail!")
            .mul(*gamma)
            .round()
            .to_usize()
            .expect("Converting rounded whole number to `T` must not fail!");

        let mut idx: Vec<(usize, &T)> = self.Y().iter().enumerate().collect();

        //reverse sorting: 5, 4, 3, 2, ..
        idx.sort_by(|(_, a), (_, b)| b.partial_cmp(a).expect("Partial cmp should not fail!"));

        let (upper, lower) = idx.split_at(n_upper);

        let mut m_upper = self.clone();
        m_upper.discard_mult(upper.iter().map(|(id, _)| *id).collect());

        let mut m_lower = self.clone();
        m_lower.discard_mult(lower.iter().map(|(id, _)| *id).collect());

        (m_upper, m_lower)
    }

    /// (max_quantile, min_quantile) TODO: expand doc
    fn min_quantile(&self, gamma: &T) -> (Self, Self)
    where
        Self: Sized + Clone,
    {
        #[cfg(debug_assertions)]
        if *gamma < T::zero() || *gamma > T::one() {
            panic!(
                "Supplied gamma ({:?}) must be int the interval [0, 1]!",
                gamma
            );
        }

        let n_lower = T::from_usize(self.n())
            .expect("Converting number of elements to `T` must not fail!")
            .mul(*gamma)
            .round()
            .to_usize()
            .expect("Converting rounded whole number to `T` must not fail!");

        let mut idx: Vec<(usize, &T)> = self.Y().iter().enumerate().collect();

        //normal sorting: 2, 3, 4, 5, ..
        idx.sort_by(|(_, a), (_, b)| a.partial_cmp(b).expect("Partial cmp should not fail!"));

        let (upper, lower) = idx.split_at(n_lower);

        let mut m_upper = self.clone();
        m_upper.discard_mult(upper.iter().map(|(id, _)| *id).collect());

        let mut m_lower = self.clone();
        m_lower.discard_mult(lower.iter().map(|(id, _)| *id).collect());

        (m_lower, m_upper)
    }
}

pub trait ObservationMean<T>: ObservationIO<T>
where
    T: dtype,
{
    /// Returns the [arithmetic mean] ȳ of all elements:
    ///
    /// ```text
    ///     1   n
    /// ȳ = ―   ∑ yᵢ
    ///     n  i=1
    /// ```
    ///
    /// If the observation memory is empty, `None` is returned.
    ///
    /// **Panics** if `T::from_usize()` fails to convert the number of observations.
    ///
    /// [arithmetic mean]: https://en.wikipedia.org/wiki/Arithmetic_mean
    fn Y_mean(&self) -> Option<T> {
        let n = self.Y().len();
        if n == 0 {
            None
        } else {
            let n =
                T::from_usize(n).expect("Converting number of observations to `T` must not fail.");
            Some(
                self.Y()
                    .iter()
                    .fold(T::zero(), |acc, elem| T::add(acc, *elem))
                    / n,
            )
        }
    }
}

pub trait ObservationVariance<T>: ObservationIO<T>
where
    T: dtype,
{
    /// Returns the variance of the observations.
    ///
    /// The variance is computed using the [Welford one-pass
    /// algorithm](https://www.jstor.org/stable/1266577).
    ///
    /// The parameter `ddof` specifies the "delta degrees of freedom". For
    /// example, to calculate the population variance, use `ddof = 0`, or to
    /// calculate the sample variance, use `ddof = 1`.
    ///
    /// The variance is defined as:
    ///
    /// ```text
    ///               1       n
    /// variance = ――――――――   ∑ (yᵢ - ȳ)²
    ///            n - ddof  i=1
    /// ```
    ///
    /// where
    ///
    /// ```text
    ///     1   n
    /// ȳ = ―   ∑ yᵢ
    ///     n  i=1
    /// ```
    ///
    /// and `n` is the length of the array.
    ///
    /// **Panics** if `ddof` is less than zero or greater than `n`
    fn Y_var(&self, ddof: T) -> T {
        let zero = T::zero();
        let n = T::from_usize(self.Y().len())
            .expect("Converting number of observations to `T` must not fail.");
        assert!(
            !(ddof < zero || ddof > n),
            "`ddof` must not be less than zero or greater than the number of observations",
        );

        let dof = n - ddof;
        let mut mean = T::zero();
        let mut sum_sq = T::zero();
        let mut i = 0;

        self.Y().iter().for_each(|&x| {
            let count = T::from_usize(i + 1)
                .expect("Converting number of observations to `T` must not fail.");
            let delta = x - mean;
            mean = mean + delta / count;
            sum_sq = (x - mean).mul_add(delta, sum_sq);
            i += 1;
        });

        sum_sq / dof
    }

    /// Returns the variance of the observations.
    ///
    /// The variance is computed using the [Welford one-pass
    /// algorithm](https://www.jstor.org/stable/1266577).
    ///
    /// The parameter `ddof` specifies the "delta degrees of freedom". For
    /// example, to calculate the population variance, use `ddof = 0`, or to
    /// calculate the sample variance, use `ddof = 1`.
    ///
    /// The variance is defined as:
    ///
    /// ```text
    ///               ⎛    1       n          ⎞
    /// stddev = sqrt ⎜ ――――――――   ∑ (xᵢ - ȳ)²⎟
    ///               ⎝ n - ddof  i=1         ⎠
    /// ```
    ///
    /// where
    ///
    /// ```text
    ///     1   n
    /// ȳ = ―   ∑ yᵢ
    ///     n  i=1
    /// ```
    ///
    /// and `n` is the length of the array.
    ///
    /// **Panics** if `ddof` is less than zero or greater than `n`
    fn Y_std(&self, ddof: T) -> T {
        self.Y_var(ddof).sqrt()
    }
}

pub trait ObservationTransform<T>: ObservationIO<T>
where
    T: dtype,
{
    fn x_prime(&self) -> impl Fn(&[T]) -> &[T];
    fn X_prime(&self) -> Mat<T>; //TODO: May need to become owned refs
    fn y_prime(&self) -> impl Fn(&T) -> T;
    fn Y_prime(&self) -> Row<T>;
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
    pub(crate) X: Mat<T>, // (d, n)
    pub(crate) Y: Row<T>, // (n, )
}

impl<T> Default for BaseMemory<T>
where
    T: dtype,
{
    fn default() -> Self {
        Self {
            X: Mat::default(),
            Y: Row::default(),
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

        Self { X, Y: Row::new() }
    }

    fn dim(&self) -> usize {
        self.X.nrows()
    }

    fn n(&self) -> usize {
        self.X.ncols()
    }

    fn X(&self) -> &Mat<T> {
        &self.X
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
        self.X.remove_cols(vec![i]);
        // self.Y.rem_at_indices(indices);
        todo!()
    }

    fn discard_mult(&mut self, idx: Vec<usize>) {
        self.X.remove_cols(idx);
        // self.Y.rem_at_indices(indices);
        todo!()
    }

    fn discard_all(&mut self) {
        let d = self.X.nrows();
        let mut X = Mat::new();
        unsafe { X.set_dims(d, 0) }
        self.X = X;
        self.Y = Row::new();
    }
}

impl<T> ObservationMaxMin<T> for BaseMemory<T> where T: dtype + OrdSubset {}

impl<T> ObservationMean<T> for BaseMemory<T> where T: dtype {}

impl<T> ObservationVariance<T> for BaseMemory<T> where T: dtype {}

////////////////////////////////////////////
