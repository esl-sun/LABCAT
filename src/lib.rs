//#![feature(type_alias_impl_trait)]
#![feature(trait_alias)]
#![feature(min_specialization)]
// #![feature(associated_type_bounds)]
//Fallible
// #![feature(try_trait_v2)]
// #![feature(const_trait_impl)]
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

use faer_core::{ComplexField, IdentityGroup};
use memory::{BaseMemory, ObservationIO};
use num_traits::{real::Real, FromPrimitive, ToPrimitive};
use std::marker::PhantomData;

pub mod bounds;
pub mod doe;
pub mod ei;
pub mod gaussian;
pub mod gp;
pub mod kde;
pub mod kernel;
pub mod lhs;
pub mod memory;
pub mod ndarray_utils;
pub mod sqexp;
pub mod tpe;
pub mod uniform;
pub mod utils;

use bounds::Bounds;
use doe::DoE;
use ei::AcqFunction;

pub trait dtype:
    ComplexField<Unit = Self, Group = IdentityGroup> + Real + FromPrimitive + ToPrimitive
{
}

impl<T> dtype for T where
    T: ComplexField<Unit = Self, Group = IdentityGroup> + Real + FromPrimitive + ToPrimitive
{
}

pub trait Surrogate<T, M>
where
    T: dtype,
    M: ObservationIO<T>,
{
    fn probe(&self, x: &[T]) -> Option<T>;
}

pub trait BayesianSurrogate<T, M>: Surrogate<T, M>
where
    T: dtype,
    M: ObservationIO<T>,
{
    fn probe_variance(&self, x: &[T]) -> Option<T>;
}

pub trait Memory<T, MI, MO>
where
    T: dtype,
    MI: ObservationIO<T>,
    MO: ObservationIO<T>,
{
    fn refit<E>(&mut self, mem: &MI) -> Result<(), E>;
    fn memory(&self) -> &MO;
    fn memory_mut(&mut self) -> &mut MO;
}

pub trait AskTell<T>
where
    T: dtype,
{
    fn ask(&mut self) -> Vec<T>;
    fn tell(&mut self, x: &[T], y: &T);
}

#[derive(Debug, Clone)]
pub struct SMBO<T, B, D, M, S, A>
where
    T: dtype,
    B: Bounds<T>,
    D: DoE<T>,
    M: ObservationIO<T>,
    S: Surrogate<T, M>,
    A: AcqFunction<T, M, S>,
{
    surrogate_mem_type: PhantomData<M>,
    bounds: B,
    doe: D,
    mem: BaseMemory<T>,
    acq_func: A,
    surrogate: S,
}

impl<T, B, D, M, S, A> SMBO<T, B, D, M, S, A>
where
    T: dtype,
    B: Bounds<T>,
    D: DoE<T>,
    S: Surrogate<T, M>,
    M: ObservationIO<T>,
    A: AcqFunction<T, M, S>,
{
    pub fn new() -> SMBO<T, B, D, M, S, A> {
        todo!()
    }

    fn optimize(self) {
        // let doe = self.doe.build_DoE();
        todo!()
    }
}

impl<T, B, D, M, S, A> AskTell<T> for SMBO<T, B, D, M, S, A>
where
    T: dtype,
    B: Bounds<T>,
    D: DoE<T>,
    S: Surrogate<T, M>,
    M: ObservationIO<T>,
    A: AcqFunction<T, M, S>,
{
    fn ask(&mut self) -> Vec<T> {
        todo!()
    }

    fn tell(&mut self, x: &[T], y: &T) {
        todo!()
    }
}
