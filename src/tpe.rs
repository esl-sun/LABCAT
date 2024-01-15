#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

use std::marker::PhantomData;

use faer_core::Mat;
use num_traits::real::Real;

use crate::kde::KDE;
use crate::kernel::Kernel;
use crate::memory::{BaseMemory, ObservationIO};
use crate::{dtype, Surrogate};

#[derive(Debug, Clone)]
pub struct TPE<T, KL, KG>
where
    T: dtype,
    KL: Kernel<T>,
    KG: Kernel<T>,
{
    data_type: PhantomData<T>,
    mem: BaseMemory<T>,
    gamma: T,
    l: KDE<T, KL>, // good distribution
    g: KDE<T, KG>, // bad distribution
}

impl<T, KL, KG> TPE<T, KL, KG>
where
    T: dtype,
    KL: Kernel<T> + Default,
    KG: Kernel<T> + Default,
{
    fn new(d: usize, gamma: T) -> Self {
        Self {
            data_type: PhantomData,
            mem: BaseMemory::new(d),
            gamma,
            l: KDE::<T, KL>::new(d),
            g: KDE::<T, KG>::new(d),
        }
    }
}

impl<T, KL, KG> Default for TPE<T, KL, KG>
where
    T: dtype,
    KL: Kernel<T> + Default,
    KG: Kernel<T> + Default,
{
    fn default() -> Self {
        Self {
            data_type: PhantomData,
            mem: BaseMemory::default(),
            gamma: T::one(),
            l: KDE::default(),
            g: KDE::default(),
        }
    }
}

impl<T, KL, KG> Surrogate<T> for TPE<T, KL, KG>
where
    T: dtype,
    KL: Kernel<T>,
    KG: Kernel<T>,
{
    fn refit<E>(&mut self, X: Mat<T>, Y: &[T]) -> Result<(), E> {
        todo!()
    }

    fn probe(&self, x: &[T]) -> T {
        self.g.probe(x) / self.l.probe(x)
    }
}
