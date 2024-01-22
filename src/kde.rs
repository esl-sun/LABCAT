#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

use std::marker::PhantomData;

use anyhow::Result;

use crate::kernel::Kernel;
use crate::memory::{BaseMemory, ObservationIO};
use crate::utils::{ColRefUtils, MatRefUtils};
use crate::{dtype, Memory, Refit, Surrogate};

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

impl<T, K, M> Surrogate<T, M> for KDE<T, K>
where
    T: dtype,
    K: Kernel<T>,
    M: ObservationIO<T>,
{
    fn probe(&self, x: &[T]) -> Option<T> {
        Some(self.mem.X().as_ref().cols().fold(T::zero(), |acc, col| acc + self.kernel.k(col.as_slice(), x)))
    }
}

impl<T, K, M> Refit<T, M> for KDE<T, K>
where
    T: dtype,
    K: Kernel<T>,
    M: ObservationIO<T>,
{
    fn refit(&mut self, mem: &M) -> Result<()> {
        self.mem.discard_all();
        self.mem.append_mult(mem.X().as_ref(), mem.Y());
        Ok(())
    }
}

impl<T, K> Memory<T, BaseMemory<T>> for KDE<T, K>
where
    T: dtype,
    K: Kernel<T>,
{
    fn memory(&self) -> &BaseMemory<T> {
        &self.mem
    }

    fn memory_mut(&mut self) -> &mut BaseMemory<T> {
        &mut self.mem
    }
}
