use std::marker::PhantomData;

use crate::dtype;

pub trait Kernel<T>
where
    Self: Sized,
    T: dtype,
{
    fn new(d: usize) -> Self;
    fn k(&self, p: &[T], q: &[T]) -> T;

    fn sum<K: Kernel<T>>(self, other: K) -> KernelSum<T, Self, K> {
        KernelSum {
            data_type: PhantomData,
            kernel_1: self,
            kernel_2: other,
        }
    }
}

pub trait Bandwidth<T>
where
    T: dtype,
{
    fn l(&self) -> &T;
    fn update_l(&mut self, new_l: &T);
}

pub trait ARD<T>
where
    T: dtype,
{
    fn l(&self) -> &[T];
    fn update_l(&mut self, new_l: &[T]);
}

pub trait BayesianKernel<T>
where
    T: dtype,
{
    fn sigma_f(&self) -> &T;
    fn sigma_n(&self) -> &T;

    fn sigma_f_mut(&mut self) -> &mut T;
    fn sigma_n_mut(&mut self) -> &mut T;
}

pub trait PDF {}

#[derive(Clone, Debug)]
pub struct KernelSum<T, K1, K2>
where
    T: dtype,
    K1: Kernel<T>,
    K2: Kernel<T>,
{
    data_type: PhantomData<T>,
    kernel_1: K1,
    kernel_2: K2,
}

impl<T, K1, K2> Default for KernelSum<T, K1, K2>
where
    T: dtype,
    K1: Kernel<T> + Default,
    K2: Kernel<T> + Default,
{
    fn default() -> Self {
        Self {
            data_type: Default::default(),
            kernel_1: Default::default(),
            kernel_2: Default::default(),
        }
    }
}

impl<T, K1, K2> Kernel<T> for KernelSum<T, K1, K2>
where
    T: dtype,
    K1: Kernel<T>,
    K2: Kernel<T>,
{
    fn new(d: usize) -> Self {
        KernelSum {
            data_type: PhantomData,
            kernel_1: Kernel::new(d),
            kernel_2: Kernel::new(d),
        }
    }

    fn k(&self, p: &[T], q: &[T]) -> T {
        self.kernel_1.k(p, q) + self.kernel_2.k(p, q)
    }
}
