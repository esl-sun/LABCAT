#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

use std::marker::PhantomData;

use faer_core::Mat;

use crate::kernel::Kernel;
use crate::memory::{BaseMemory, ObservationIO};
use crate::{dtype, Surrogate, SurrogateMemory};

#[derive(Debug, Clone)]
pub struct KDE<T, K>
where
    T: dtype,
    K: Kernel<T>,
{
    data_type: PhantomData<T>,
    mem: BaseMemory<T>,
    kernel: K,
}

impl<T, K> Default for KDE<T, K>
where
    T: dtype,
    K: Kernel<T>,
{
    fn default() -> Self {
        Self {
            data_type: PhantomData,
            mem: BaseMemory::default(),
            kernel: Kernel::new(0),
        }
    }
}

impl<T, K> KDE<T, K>
where
    T: dtype,
    K: Kernel<T>,
{
    pub fn new(d: usize) -> KDE<T, K> {
        KDE {
            data_type: PhantomData,
            mem: BaseMemory::new(d),
            kernel: Kernel::new(d),
        }
    }
}

impl<T, K> Surrogate<T> for KDE<T, K>
where
    T: dtype,
    K: Kernel<T>,
{
    fn refit<E>(&mut self, _: Mat<T>, _: &[T]) -> Result<(), E> {
        Ok(())
    }

    fn probe(&self, x: &[T]) -> T {
        // self.mem.X().cols()
        todo!()
    }
}

impl<T, K> SurrogateMemory<T> for KDE<T, K>
where
    T: dtype,
    K: Kernel<T>,
{
    fn memory(&self) -> &impl ObservationIO<T> {
        &self.mem
    }

    fn memory_mut(&mut self) -> &mut impl ObservationIO<T> {
        &mut self.mem
    }
}
