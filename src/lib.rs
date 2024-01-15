//#![feature(type_alias_impl_trait)]
#![feature(trait_alias)]
#![feature(min_specialization)]
// #![feature(associated_type_bounds)]
//Fallible
// #![feature(try_trait_v2)]
// #![feature(const_trait_impl)]
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

use faer_core::{ComplexField, IdentityGroup, Mat};
use memory::ObservationIO;
use num_traits::real::Real;
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

pub trait dtype: ComplexField<Unit = Self, Group = IdentityGroup> + Real {}

impl<T> dtype for T where T: ComplexField<Unit = Self, Group = IdentityGroup> + Real {}

pub trait Surrogate<T>
where
    T: dtype,
{
    fn refit<E>(&mut self, X: Mat<T>, Y: &[T]) -> Result<(), E>;
    fn probe(&self, x: &[T]) -> T;
    // fn bounds(&self) -> Vec<>
}

pub trait SurrogateMemory<T>: Surrogate<T>
where
    T: dtype,
{
    fn memory(&self) -> &impl ObservationIO<T>;
    fn memory_mut(&mut self) -> &mut impl ObservationIO<T>;
}

pub trait BayesianSurrogate<T>: Surrogate<T>
where
    T: dtype,
{
    fn probe_variance(&self, x: &[T]) -> T;
    // fn bounds(&self) -> Vec<>
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
    S: Surrogate<T>,
    A: AcqFunction<T>,
{
    data_type: PhantomData<T>,
    bounds: B,
    doe: D,
    mem: M,
    acq_func: A,
    surrogate: S,
}

impl<T, B, D, M, S, A> SMBO<T, B, D, M, S, A>
where
    T: dtype,
    B: Bounds<T>,
    D: DoE<T>,
    S: Surrogate<T>,
    M: ObservationIO<T>,
    A: AcqFunction<T>,
{
    pub fn new() -> SMBO<T, B, D, M, S, A> {
        todo!()
    }

    fn optimize(self) {
        // let doe = self.doe.build_DoE();
    }

    fn test(self) {
        // let x: Vector<T> = self.surrogate.ask().into();
        // self.surrogate.tell(x, T);
    }
}

impl<T, B, D, M, S, A> AskTell<T> for SMBO<T, B, D, M, S, A>
where
    T: dtype,
    B: Bounds<T>,
    D: DoE<T>,
    S: Surrogate<T>,
    M: ObservationIO<T>,
    A: AcqFunction<T>,
{
    fn ask(&mut self) -> Vec<T> {
        todo!()
    }

    fn tell(&mut self, x: &[T], y: &T) {
        todo!()
    }
}
