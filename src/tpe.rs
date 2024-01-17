#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

use std::marker::PhantomData;

use ord_subset::OrdSubset;

use crate::kde::KDE;
use crate::kernel::Kernel;
use crate::memory::{ObservationIO, ObservationMaxMin};
use crate::{dtype, Memory, Surrogate};

#[derive(Debug, Clone)]
pub struct TPE<T, KL, KG, M>
where
    T: dtype + OrdSubset,
    KL: Kernel<T>,
    KG: Kernel<T>,
    M: ObservationIO<T> + ObservationMaxMin<T>,
{
    data_type: PhantomData<T>,
    mem: M,
    gamma: T,
    l: KDE<T, KL>, // good distribution
    g: KDE<T, KG>, // bad distribution
}

impl<T, KL, KG, M> TPE<T, KL, KG, M>
where
    T: dtype + OrdSubset,
    KL: Kernel<T> + Default,
    KG: Kernel<T> + Default,
    M: ObservationIO<T> + ObservationMaxMin<T>,
{
    fn new(d: usize, gamma: T) -> Self {
        Self {
            data_type: PhantomData,
            mem: ObservationIO::new(0),
            gamma,
            l: KDE::<T, KL>::new(d),
            g: KDE::<T, KG>::new(d),
        }
    }
}

impl<T, KL, KG, M> Default for TPE<T, KL, KG, M>
where
    T: dtype + OrdSubset,
    KL: Kernel<T> + Default,
    KG: Kernel<T> + Default,
    M: ObservationIO<T> + ObservationMaxMin<T>,
{
    fn default() -> Self {
        Self {
            data_type: PhantomData,
            mem: ObservationIO::new(0),
            gamma: T::one(),
            l: KDE::default(),
            g: KDE::default(),
        }
    }
}

impl<T, KL, KG, M> Surrogate<T, M> for TPE<T, KL, KG, M>
where
    T: dtype + OrdSubset,
    KL: Kernel<T>,
    KG: Kernel<T>,
    M: ObservationIO<T> + ObservationMaxMin<T>,
{
    fn probe(&self, x: &[T]) -> Option<T> {
        Some(Surrogate::<T, M>::probe(&self.g, x)? / Surrogate::<T, M>::probe(&self.l, x)?)
    }
}

impl<T, KL, KG, MI, MO> Memory<T, MI, MO> for TPE<T, KL, KG, MO>
where
    T: dtype + OrdSubset,
    KL: Kernel<T>,
    KG: Kernel<T>,
    MI: ObservationIO<T> + ObservationMaxMin<T>,
    MO: ObservationIO<T> + ObservationMaxMin<T>,
{   
    #[allow(refining_impl_trait)]
    fn refit<E>(&mut self, mem: &MI) -> Result<(), E> {
        let (m_l, m_g) = mem.max_quantile(&self.gamma);
        // self.mem = mem.clone();
        todo!()
    }

    fn memory(&self) -> &M {
        &self.mem
    }

    fn memory_mut(&mut self) -> &mut M {
        &mut self.mem
    }
}
