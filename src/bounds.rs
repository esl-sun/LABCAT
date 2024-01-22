use crate::dtype;

pub trait Bounds<T>
where
    T: dtype,
{
    fn inside(x: &[T]) -> bool;
}

pub struct ContinuousBounds {}

impl<T> Bounds<T> for ContinuousBounds
where
    T: dtype,
{
    fn inside(_: &[T]) -> bool {
        todo!()
    }
}
