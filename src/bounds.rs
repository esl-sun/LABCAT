use faer_core::ComplexField;

pub trait Bounds<T>
where
    T: ComplexField<Unit = T>,
{
    fn inside(x: &[T]) -> bool;
}
