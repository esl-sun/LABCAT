//#![feature(type_alias_impl_trait)]
// #![feature(trait_alias)]
// #![feature(min_specialization)]
// #![feature(associated_type_bounds)]
// #![feature(associated_type_defaults)]
//Fallible
// #![feature(try_trait_v2)]
// #![feature(const_trait_impl)]
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(dead_code)] //TODO: Remove

use std::marker::PhantomData;

use anyhow::Result; //TODO: Make crate Error types
use faer_traits::ComplexField;
use memory::{BaseMemory, Memory, ObservationIO};
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
pub mod tune;
pub mod uniform;
pub mod utils;

use bounds::{Bounds, UpperLowerBounds};
use doe::DoE;
use ei::AcqFunction;
use tune::SurrogateTuning;

pub trait dtype: ComplexField<Unit = Self> + Real + FromPrimitive + ToPrimitive {}

impl<T> dtype for T where T: ComplexField<Unit = Self> + Real + FromPrimitive + ToPrimitive {}

pub trait SurrogateIO<T>
where
    T: dtype,
{
    fn new(d: usize) -> Self;
    fn probe(&self, x: &[T]) -> Option<T>;
}

pub trait BayesianSurrogateIO<T>: SurrogateIO<T>
where
    T: dtype,
{
    fn probe_variance(&self, x: &[T]) -> Option<T>;
}

pub trait Refit<T>
where
    T: dtype,
{
    fn refit(&mut self) -> Result<()>
    where
        Self: Memory<T>;
}

pub trait RefitWith<T, M>
where
    T: dtype,
    M: ObservationIO<T>,
{
    fn refit_from(&mut self, mem: &M) -> Result<()>;
}

pub trait Surrogate<T>
where
    T: dtype,
{
    type SurType: SurrogateIO<T>;

    fn surrogate(&self) -> &Self::SurType;
    fn surrogate_mut(&mut self) -> &mut Self::SurType;
}

pub trait AskTell<T>
where
    T: dtype,
{
    fn ask(&mut self) -> Vec<T>;
    fn tell(&mut self, x: &[T], y: &T);
}

#[derive(Debug, Clone)]
pub struct SMBO<T, B, D, S, H, A>
where
    T: dtype,
    B: Bounds<T>,
    D: DoE<T>,
    S: SurrogateIO<T>,
    H: SurrogateTuning<T, S>,
    A: AcqFunction<T, S>,
{
    bounds: B,
    doe: D,
    mem: BaseMemory<T>,
    acq_func: A,
    surrogate: S,
    tuning_strategy: H,
}

impl<T, B, D, S, H, A> SMBO<T, B, D, S, H, A>
where
    T: dtype,
    B: Bounds<T> + UpperLowerBounds<T>,
    D: DoE<T>,
    S: SurrogateIO<T>,
    H: SurrogateTuning<T, S> + Default,
    A: AcqFunction<T, S> + Default,
{
    pub fn new(d: usize, bounds: B, n_init: usize) -> SMBO<T, B, D, S, H, A> {
        let mut doe = D::default();
        doe.build_DoE(n_init, &bounds);

        Self {
            bounds,
            doe,
            mem: BaseMemory::new(d),
            acq_func: A::default(),
            surrogate: S::new(d),
            tuning_strategy: H::default(),
        }
    }
}

impl<T, B, D, S, H, A> AskTell<T> for SMBO<T, B, D, S, H, A>
where
    T: dtype,
    B: Bounds<T>,
    D: DoE<T>,
    S: SurrogateIO<T>,
    H: SurrogateTuning<T, S>,
    A: AcqFunction<T, S>,
{
    fn ask(&mut self) -> Vec<T> {
        if let Some(doe_x) = self.doe.get(self.mem.n()) {
            return doe_x.to_vec();
        }

        todo!()
    }

    fn tell(&mut self, _: &[T], _: &T) {
        todo!()
    }
}

pub struct Auto<T, F, O>
where
    T: dtype,
    F: Fn(&[T]) -> T,
    O: AskTell<T>,
{
    dtype: PhantomData<T>,
    opt: O,
    obj: F,
}

impl<T, F, O> Auto<T, F, O>
where
    T: dtype,
    F: Fn(&[T]) -> T,
    O: AskTell<T>,
{
    pub fn new(opt: O, obj: F) -> Self {
        Self {
            dtype: PhantomData,
            opt,
            obj,
        }
    }

    pub fn optimize(&mut self) {
        for i in 0..10 {
            dbg!(i);
            let x = self.opt.ask();
            let y = (self.obj)(&x);
            dbg!(&x);
            dbg!(&y);
            self.opt.tell(&x, &y);
        }
    }
}
