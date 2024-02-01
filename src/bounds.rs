use faer_core::Mat;
use itertools::izip;

use crate::dtype;

pub trait Bounds<T>
where
    T: dtype,
{
    fn inside(&self, x: &[T]) -> bool;
    fn dim(&self) -> usize;
}

pub trait UpperLowerBounds<T>: Bounds<T>
where
    T: dtype,
{
    fn ub(&self) -> &[T];
    fn lb(&self) -> &[T];
    fn as_mat(&self) -> Mat<T> {
        Mat::<T>::from_fn(self.dim(), 2, |i, j| match j {
            0 => self.lb()[i],
            1 => self.ub()[i],
            _ => panic!("Should never fail! Matrix has been defined with 2 cols."),
        })
    }
}

#[derive(Debug, Clone)]
pub struct ContinuousBounds<T>
where
    T: dtype,
{
    bounds_mat: Mat<T>,
}

impl<T> ContinuousBounds<T>
where
    T: dtype,
{
    pub fn new(d: usize, lb: &[T], ub: &[T]) -> Self {
        #[cfg(debug_assertions)]
        if lb.len() != d {
            panic!(
                "Dimension lower bounds {} does not match bounds dimensions {}!",
                lb.len(),
                d
            );
        }

        #[cfg(debug_assertions)]
        if ub.len() != d {
            panic!(
                "Dimension upper bounds {} does not match bounds dimensions {}!",
                ub.len(),
                d
            );
        }

        let bounds_mat = Mat::<T>::from_fn(d, 2, |i, j| match j {
            0 => lb[i],
            1 => ub[i],
            _ => panic!("Should never fail! Matrix has been defined with 2 cols."),
        });

        Self { bounds_mat }
    }

    pub fn unit(d: usize) -> Self {
        Self::new(
            d,
            vec![T::neg(T::one()); d].as_slice(),
            vec![T::one(); d].as_slice(),
        )
    }

    pub fn scaled_unit(d: usize, k: T) -> Self {
        Self::new(
            d,
            vec![T::neg(T::one()) * k; d].as_slice(),
            vec![T::one() * k; d].as_slice(),
        )
    }
}

impl<T> Bounds<T> for ContinuousBounds<T>
where
    T: dtype,
{
    fn inside(&self, x: &[T]) -> bool {
        #[cfg(debug_assertions)]
        if x.len() != self.dim() {
            panic!(
                "Dimension of provided point {} does not match bounds dimensions {}!",
                x.len(),
                self.dim()
            );
        }

        izip!(x, self.lb(), self.ub()).all(|(x, lb, ub)| lb <= x && x <= ub)
    }

    fn dim(&self) -> usize {
        self.bounds_mat.nrows()
    }
}

impl<T> UpperLowerBounds<T> for ContinuousBounds<T>
where
    T: dtype,
{
    fn ub(&self) -> &[T] {
        self.bounds_mat.col_as_slice(1)
    }

    fn lb(&self) -> &[T] {
        self.bounds_mat.col_as_slice(0)
    }
}
