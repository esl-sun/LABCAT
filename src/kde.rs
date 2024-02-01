#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

use std::marker::PhantomData;

use anyhow::Result;

use crate::kernel::BaseKernel;
use crate::memory::{BaseMemory, ObservationIO};
use crate::utils::{ColRefUtils, MatRefUtils};
use crate::{dtype, Memory, Refit, RefitWith, SurrogateIO};

#[derive(Debug, Clone)]
pub struct KDE<T, K>
where
    T: dtype,
    K: BaseKernel<T>,
{
    data_type: PhantomData<T>,
    mem: BaseMemory<T>,
    kernel: K,
}

impl<T, K> Default for KDE<T, K>
where
    T: dtype,
    K: BaseKernel<T>,
{
    fn default() -> Self {
        Self {
            data_type: PhantomData,
            mem: BaseMemory::default(),
            kernel: BaseKernel::new(0),
        }
    }
}

// impl<T, K> KDE<T, K>
// where
//     T: dtype,
//     K: BaseKernel<T>,
// {}

impl<T, K> SurrogateIO<T> for KDE<T, K>
where
    T: dtype,
    K: BaseKernel<T>,
{
    fn new(d: usize) -> Self {
        Self {
            data_type: PhantomData,
            mem: BaseMemory::new(d),
            kernel: BaseKernel::new(d),
        }
    }
    
    fn probe(&self, x: &[T]) -> Option<T> {
        Some(
            self.mem
                .X()
                .as_ref()
                .cols()
                .fold(T::zero(), |acc, col| acc + self.kernel.k(col.as_slice(), x)),
        )
    }
}

impl<T, K> Refit<T> for KDE<T, K>
where
    T: dtype,
    K: BaseKernel<T>,
{
    //TODO: Add warning that refitting a BaseMemory KDE does nothing
    fn refit(&mut self) -> Result<()>
    where
        Self: Memory<T>,
    {
        Ok(())
    }
}

impl<T, K, M> RefitWith<T, M> for KDE<T, K>
where
    T: dtype,
    K: BaseKernel<T>,
    M: ObservationIO<T>,
{
    fn refit_from(&mut self, mem: &M) -> Result<()> {
        self.mem.discard_all();
        self.mem.append_mult(mem.X().as_ref(), mem.Y());
        Ok(())
    }
}

impl<T, K> Memory<T> for KDE<T, K>
where
    T: dtype,
    K: BaseKernel<T>,
{
    type MemType = BaseMemory<T>;

    fn memory(&self) -> &BaseMemory<T> {
        &self.mem
    }

    fn memory_mut(&mut self) -> &mut BaseMemory<T> {
        &mut self.mem
    }
}
