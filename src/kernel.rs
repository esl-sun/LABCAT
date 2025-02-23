use std::marker::PhantomData;

use faer::{Mat, MatRef, Row};

use crate::{dtype, gp::GPSurrogate, utils::MatRefUtils};

pub trait Kernel<T>
where
    T: dtype,
{
    type KernType: BaseKernel<T>;

    fn kernel(&self) -> &Self::KernType;
    fn kernel_mut(&mut self) -> &mut Self::KernType;
}

pub trait BaseKernel<T>
where
    Self: Sized,
    T: dtype,
{
    fn new(d: usize) -> Self;
    fn k(&self, p: &[T], q: &[T]) -> T;

    fn k_diag(&self, X: MatRef<T>, x: &[T]) -> Row<T> {
        Row::<T>::from_fn(X.ncols(), |i| self.k(X.col_as_slice(i), x))
    }

    fn sum<K: BaseKernel<T>>(self, other: K) -> KernelSum<T, Self, K> {
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
    fn h(&self) -> &T;
    fn update_h(&mut self, new_h: &T);
}

pub trait ARD<T>
where
    T: dtype,
{
    fn l(&self) -> &[T];
    fn dim(&self) -> usize;
    fn update_l(&mut self, new_l: &[T]);
    fn whiten_l(&mut self);

    fn l_gp_jac<S: GPSurrogate<T, KernType = Self>>(&self, gp: &S) -> impl Iterator<Item = Mat<T>>; //TODO: MOVE TO SEPERATE TRAIT
}

pub trait BayesianKernel<T>
where
    T: dtype,
{
    fn sigma_f(&self) -> &T;
    fn sigma_n(&self) -> &T;

    fn sigma_f_mut(&mut self) -> &mut T;
    fn sigma_n_mut(&mut self) -> &mut T;

    fn sigma_f_gp_jac<S: GPSurrogate<T, KernType = Self>>(&self, gp: &S) -> Mat<T>;
    fn sigma_n_gp_jac<S: GPSurrogate<T, KernType = Self>>(&self, _gp: &S) -> Mat<T>; //TODO: MOVE TO SEPERATE TRAIT
}

pub trait PDF {}

#[derive(Clone, Debug)]
pub struct KernelSum<T, K1, K2>
where
    T: dtype,
    K1: BaseKernel<T>,
    K2: BaseKernel<T>,
{
    data_type: PhantomData<T>,
    kernel_1: K1,
    kernel_2: K2,
}

impl<T, K1, K2> Default for KernelSum<T, K1, K2>
where
    T: dtype,
    K1: BaseKernel<T> + Default,
    K2: BaseKernel<T> + Default,
{
    fn default() -> Self {
        Self {
            data_type: Default::default(),
            kernel_1: Default::default(),
            kernel_2: Default::default(),
        }
    }
}

impl<T, K1, K2> BaseKernel<T> for KernelSum<T, K1, K2>
where
    T: dtype,
    K1: BaseKernel<T>,
    K2: BaseKernel<T>,
{
    fn new(d: usize) -> Self {
        KernelSum {
            data_type: PhantomData,
            kernel_1: BaseKernel::new(d),
            kernel_2: BaseKernel::new(d),
        }
    }

    fn k(&self, p: &[T], q: &[T]) -> T {
        self.kernel_1.k(p, q) + self.kernel_2.k(p, q)
    }
}
