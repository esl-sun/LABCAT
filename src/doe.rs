use faer::{Mat, MatRef};
use rand::Rng;

use crate::{bounds::UpperLowerBounds, dtype};

pub trait DoE<T>: Default
where
    Self: Clone,
    T: dtype,
{
    fn build_DoE<B: UpperLowerBounds<T>>(&mut self, n: usize, bounds: &B);
    fn DoE(&self) -> MatRef<'_, T>;
    fn n(&self) -> usize {
        self.DoE().ncols()
    }
    fn i(&self, id: usize) -> &[T] {
        self.DoE().col(id).try_as_col_major().unwrap().as_slice()
    }
    fn iter(&self) -> DoeIter<Self, T> {
        DoeIter {
            doe: self.clone(),
            current_n: 0,
            dtype: std::marker::PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DoeIter<D, T>
where
    D: DoE<T>,
    T: dtype,
{
    doe: D,
    current_n: usize,
    dtype: std::marker::PhantomData<T>,
}

// impl<D, T> DoeIter<D, T>
// where
//     D: DoE<T>,
//     T: dtype,
// {
//     pub fn empty(&self) -> bool {
//         self.current_n > self.doe.DoE().ncols()
//     }
// }

impl<D, T> Iterator for DoeIter<D, T>
where
    D: DoE<T>,
    T: dtype,
{
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.len() == 0 {
            return None;
        }

        let c = self
            .doe
            .DoE()
            .col(self.current_n)
            .try_as_col_major()
            .unwrap()
            .as_slice();
        self.current_n += 1;
        Some(c.into())
    }
}

impl<D, T> ExactSizeIterator for DoeIter<D, T>
where
    D: DoE<T>,
    T: dtype,
{
    fn len(&self) -> usize {
        if self.current_n <= self.doe.DoE().ncols() {
            self.doe.DoE().ncols() - self.current_n
        } else {
            0
        }
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
        Self { doe: Mat::new() }
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
        self.doe =
            Mat::from_fn(bounds.dim(), n, |i, _| {
                T::from_f64(rand::rng().random_range(
                    bounds.lb()[i].to_f64().unwrap()..bounds.ub()[i].to_f64().unwrap(),
                ))
                .unwrap()
            })
    }

    fn DoE(&self) -> MatRef<'_, T> {
        self.doe.as_ref()
    }
}
