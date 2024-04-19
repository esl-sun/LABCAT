use faer::MatRef;

use crate::{bounds::UpperLowerBounds, dtype};

pub trait DoE<T>: Default
where
    T: dtype,
{
    fn build_DoE<B: UpperLowerBounds<T>>(&mut self, n: usize, bounds: &B);
    fn DoE(&self) -> MatRef<T>;
}
