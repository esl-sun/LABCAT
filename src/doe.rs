use faer::Mat;

use crate::{bounds::UpperLowerBounds, dtype};

pub trait DoE<T>: Default
where
    T: dtype,
{
    fn build_DoE<B: UpperLowerBounds<T>>(&mut self, n: usize, bounds: &B);
}
