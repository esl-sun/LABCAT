#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

use std::marker::PhantomData;

use anyhow::Result;
use ord_subset::OrdSubset;

use crate::kde::KDE;
use crate::kernel::Kernel;
use crate::memory::{ObservationIO, ObservationMaxMin};
use crate::{dtype, Memory, Refit, Surrogate};

pub trait TPESurrogate<T, M>
where
    T: dtype,
    M: ObservationIO<T>,
{
    type Surrogate_l: Surrogate<T, M>;
    type Surrogate_g: Surrogate<T, M>;

    fn l(&self) -> &Self::Surrogate_l;
    fn probe_l(&self, x: &[T]) -> Option<T> {
        self.l().probe(x)
    }

    fn g(&self) -> &Self::Surrogate_g;
    fn probe_g(&self, x: &[T]) -> Option<T> {
        self.g().probe(x)
    }
}

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

impl<T, KL, KG, M> TPESurrogate<T, M> for TPE<T, KL, KG, M>
where
    T: dtype + OrdSubset,
    KL: Kernel<T>,
    KG: Kernel<T>,
    M: ObservationIO<T> + ObservationMaxMin<T>,
{
    type Surrogate_l = KDE<T, KL>;

    type Surrogate_g = KDE<T, KG>;

    fn l(&self) -> &Self::Surrogate_l {
        &self.l
    }

    fn g(&self) -> &Self::Surrogate_g {
        &self.g
    }
}

impl<T, KL, KG, M, MS> Refit<T, M> for TPE<T, KL, KG, MS>
where
    T: dtype + OrdSubset,
    KL: Kernel<T>,
    KG: Kernel<T>,
    M: ObservationIO<T> + ObservationMaxMin<T> + Clone,
    MS: ObservationIO<T> + ObservationMaxMin<T>,
{
    // #[allow(refining_impl_trait)]
    fn refit(&mut self, mem: &M) -> Result<()> {
        self.mem.discard_all();
        self.mem.append_mult(mem.X().as_ref(), mem.Y());

        let (m_l, m_g) = mem.min_quantile(&self.gamma); //TODO: match on argmax/argmin optimzation
        self.l.refit(&m_l)?;
        self.g.refit(&m_g)?;
        Ok(())
    }
}

impl<T, KL, KG, M> Memory<T, M> for TPE<T, KL, KG, M>
where
    T: dtype + OrdSubset,
    KL: Kernel<T>,
    KG: Kernel<T>,
    M: ObservationIO<T> + ObservationMaxMin<T>,
{
    fn memory(&self) -> &M {
        &self.mem
    }

    fn memory_mut(&mut self) -> &mut M {
        &mut self.mem
    }
}
