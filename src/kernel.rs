use nalgebra::Scalar;

use crate::utils::VectorView;

pub trait Kernel<T>
where
    T: Scalar,
{
    fn new(d: usize) -> Self;
    fn k(&self, p: VectorView<T>, q: VectorView<T>) -> T;
}

pub trait ARD<T>
where
    T: Scalar,
{
    fn l(&self) -> &[T];
    fn update_l(&mut self, new_l: &[T]);
}

pub trait BayesianKernel<T>
where
    T: Scalar,
{
    fn sigma_f(&self) -> &T;
    fn sigma_n(&self) -> &T;

    fn sigma_f_mut(&mut self) -> &mut T;
    fn sigma_n_mut(&mut self) -> &mut T;
}
