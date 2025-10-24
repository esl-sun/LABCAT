use std::marker::PhantomData;

use faer::{Mat, MatRef, Row};

use crate::{dtype, gp::GPSurrogate};

pub trait Kernel<T>
where
    T: dtype,
{
    type KernType: BaseKernel<T>;

    fn kernel(&self) -> &Self::KernType;
    fn kernel_mut(&mut self) -> &mut Self::KernType;
}

pub trait BaseKernel<T>: Clone
where
    Self: Sized,
    T: dtype,
{
    type Theta;

    fn new(d: usize) -> Self;
    fn theta_value(&self, theta: Self::Theta) -> T;
    fn theta_value_mut(&mut self, theta: Self::Theta) -> &mut T;

    fn thetas(&self) -> impl Iterator<Item = Self::Theta>;

    fn k(&self, p: &[T], q: &[T]) -> T;

    fn k_diag(&self, X: MatRef<T>, x: &[T]) -> Row<T> {
        Row::<T>::from_fn(X.ncols(), |i| {
            self.k(X.col(i).try_as_col_major().unwrap().as_slice(), x)
        })
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

pub enum ThetaKernelSum<Theta1, Theta2> {
    theta1(Theta1),
    theta2(Theta2),
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
    type Theta = ThetaKernelSum<K1::Theta, K2::Theta>;

    fn new(d: usize) -> Self {
        KernelSum {
            data_type: PhantomData,
            kernel_1: BaseKernel::new(d),
            kernel_2: BaseKernel::new(d),
        }
    }

    fn theta_value(&self, theta: Self::Theta) -> T {
        match theta {
            ThetaKernelSum::theta1(theta) => self.kernel_1.theta_value(theta),
            ThetaKernelSum::theta2(theta) => self.kernel_2.theta_value(theta),
        }
    }

    fn theta_value_mut(&mut self, theta: Self::Theta) -> &mut T {
        match theta {
            ThetaKernelSum::theta1(theta) => self.kernel_1.theta_value_mut(theta),
            ThetaKernelSum::theta2(theta) => self.kernel_2.theta_value_mut(theta),
        }
    }

    fn thetas(&self) -> impl Iterator<Item = Self::Theta> {
        self.kernel_1
            .thetas()
            .map(ThetaKernelSum::theta1)
            .chain(
                self.kernel_2
                    .thetas()
                    .map(ThetaKernelSum::theta2),
            )
    }

    fn k(&self, p: &[T], q: &[T]) -> T {
        self.kernel_1.k(p, q) + self.kernel_2.k(p, q)
    }
}
