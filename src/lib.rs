//#![feature(type_alias_impl_trait)]
// #![feature(trait_alias)]
// #![feature(min_specialization)]
#![feature(associated_type_bounds)]
//Fallible
// #![feature(try_trait_v2)]
// #![feature(const_trait_impl)]
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(dead_code)] //TODO: Remove

use anyhow::Result; //TODO: Make crate Error types
use faer_core::{ComplexField, IdentityGroup};
use kernel::BaseKernel;
use memory::{BaseMemory, ObservationIO};
use num_traits::{real::Real, FromPrimitive, ToPrimitive};

pub mod bounds;
pub mod doe;
pub mod ei;
pub mod gaussian;
pub mod gp;
pub mod kde;
pub mod kernel;
pub mod labcat;
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

pub trait Surrogate<T>
where
    T: dtype,
    // M: ObservationIO<T>,
{
    fn probe(&self, x: &[T]) -> Option<T>;
}

pub trait BayesianSurrogate<T>: Surrogate<T>
where
    T: dtype,
    // M: ObservationIO<T>,
{
    fn probe_variance(&self, x: &[T]) -> Option<T>;
}

pub trait Refit<T>
where
    T: dtype,

{
    // fn refit_from(&mut self, mem: &impl ObservationIO<T>) -> Result<()>;
    fn refit(&mut self) -> Result<()> where Self: Memory<T>;
}

pub trait RefitWith<T, M> 
where
    T: dtype,
    M: ObservationIO<T>,
{
    fn refit_from(&mut self, mem: &M) -> Result<()>;
}

pub trait Memory<T>
where
    T: dtype,
{
    
    type MemType: ObservationIO<T>;
    
    fn memory(&self) -> &Self::MemType;
    fn memory_mut(&mut self) -> &mut Self::MemType;
}

pub trait Kernel<T>
where
    T: dtype,
{
    
    type KernType: BaseKernel<T>;
    
    fn kernel(&self) -> &Self::KernType;
    fn kernel_mut(&mut self) -> &mut Self::KernType;
}

pub trait AskTell<T>
where
    T: dtype,
{
    fn ask(&mut self) -> Vec<T>;
    fn tell(&mut self, x: &[T], y: &T);
}

#[derive(Debug, Clone)]
pub struct SMBO<T, B, D, S, A>
where
    T: dtype,
    B: Bounds<T>,
    D: DoE<T>,
    S: Surrogate<T>,
    A: AcqFunction<T, S>,
{
    bounds: B,
    doe: D,
    mem: BaseMemory<T>,
    acq_func: A,
    surrogate: S,
}

impl<T, B, D, S, A> SMBO<T, B, D, S, A>
where
    T: dtype,
    B: Bounds<T>,
    D: DoE<T>,
    S: Surrogate<T>,
    A: AcqFunction<T, S>,
{
    pub fn new() -> SMBO<T, B, D, S, A> {
        todo!()
    }

    fn optimize(self) {
        // let doe = self.doe.build_DoE();
        todo!()
    }
}

impl<T, B, D, S, A> AskTell<T> for SMBO<T, B, D, S, A>
where
    T: dtype,
    B: Bounds<T>,
    D: DoE<T>,
    S: Surrogate<T>,
    A: AcqFunction<T, S>,
{
    fn ask(&mut self) -> Vec<T> {
        todo!()
    }

    fn tell(&mut self, _: &[T], _: &T) {
        todo!()
    }
}