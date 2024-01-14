use faer_core::{Mat, MatRef};

use crate::dtype;

pub trait DoE<T>: Default
where
    T: dtype,
{
    fn build_DoE(&self, n: usize, bounds: MatRef<T>) -> Mat<T>;
}
