#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

use std::marker::PhantomData;

use anyhow::Result;
use ord_subset::OrdSubset;

use crate::kde::KDE;
use crate::kernel::BaseKernel;
use crate::memory::{ObservationIO, ObservationMaxMin};
use crate::{dtype, Memory, Refit, RefitWith, SurrogateIO};

pub trait TPESurrogate<T>
where
    T: dtype,
{
    type Surrogate_l: SurrogateIO<T>;
    type Surrogate_g: SurrogateIO<T>;

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
    KL: BaseKernel<T>,
    KG: BaseKernel<T>,
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
    KL: BaseKernel<T> + Default,
    KG: BaseKernel<T> + Default,
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
    KL: BaseKernel<T> + Default,
    KG: BaseKernel<T> + Default,
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

impl<T, KL, KG, M> TPESurrogate<T> for TPE<T, KL, KG, M>
where
    T: dtype + OrdSubset,
    KL: BaseKernel<T>,
    KG: BaseKernel<T>,
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

impl<T, KL, KG, MS> Refit<T> for TPE<T, KL, KG, MS>
where
    T: dtype + OrdSubset,
    KL: BaseKernel<T>,
    KG: BaseKernel<T>,
    MS: ObservationIO<T> + ObservationMaxMin<T>,
{
    fn refit(&mut self) -> Result<()>
    where
        Self: Memory<T>,
    {
        <KDE<T, KL> as Refit<T>>::refit(&mut self.l)?;
        <KDE<T, KG> as Refit<T>>::refit(&mut self.g)?;
        Ok(())
    }
}

impl<T, KL, KG, M, MI> RefitWith<T, MI> for TPE<T, KL, KG, M>
where
    T: dtype + OrdSubset,
    KL: BaseKernel<T>,
    KG: BaseKernel<T>,
    M: ObservationIO<T> + ObservationMaxMin<T>,
    MI: ObservationIO<T> + ObservationMaxMin<T> + Clone,
{
    // #[allow(refining_impl_trait)]
    fn refit_from(&mut self, mem: &MI) -> Result<()> {
        self.mem.discard_all();
        self.mem.append_mult(mem.X().as_ref(), mem.Y());

        let (m_l, m_g) = mem.min_quantile(&self.gamma); //TODO: match on argmax/argmin optimzation
        self.l.refit_from(&m_l)?;
        self.l.refit_from(&m_g)?;
        Ok(())
    }
}

impl<T, KL, KG, M> Memory<T> for TPE<T, KL, KG, M>
where
    T: dtype + OrdSubset,
    KL: BaseKernel<T>,
    KG: BaseKernel<T>,
    M: ObservationIO<T> + ObservationMaxMin<T>,
{
    type MemType = M;

    fn memory(&self) -> &M {
        &self.mem
    }

    fn memory_mut(&mut self) -> &mut M {
        &mut self.mem
    }
}
