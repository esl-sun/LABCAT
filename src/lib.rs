//#![feature(type_alias_impl_trait)]
#![feature(return_position_impl_trait_in_trait)]
// #![feature(trait_alias)]
// #![feature(anonymous_lifetime_in_impl_trait)]
//#![feature(associated_type_bounds)]
//Fallible
// #![feature(try_trait_v2)]
// #![feature(const_trait_impl)]

use memory::ObservationMemory;
use nalgebra::Scalar;
use std::{marker::PhantomData, error::Error};

pub mod kernel;
pub mod memory;
pub mod ndarray_utils;
pub mod utils;

pub mod ei;
pub mod gp;
pub mod kde;
pub mod lhs;
pub mod sqexp;

use ei::AcqFunction;
use lhs::DoE;
use utils::{Matrix, Vector};

// pub trait Input<T: Scalar> = Into<Vector<T>>;
// trait Input<T: Scalar> {}

pub trait Surrogate<T>: Default
where
    T: Scalar,
{
    fn fit<E>(&mut self, X: Matrix<T>, Y: Vector<T>) -> Result<(), E>;
    fn probe(&self, x: &Vector<T>) -> T;
    // fn bounds(&self) -> Vec<>
}

pub trait BayesianSurrogate<T>: Surrogate<T>
where
    T: Scalar,
{
    fn probe_variance(&self, x: &Vector<T>) -> T;
    // fn bounds(&self) -> Vec<>
}

pub trait AskTell<T>
where
    T: Scalar,
{
    fn ask(&mut self) -> impl Into<Vector<T>>;
    fn tell(&mut self, x: Vector<T>, y: T);
}

#[derive(Debug, Clone)]
pub struct SMBO<T, D, M, S, A>
where
    T: Scalar,
    D: DoE<T>,
    M: ObservationMemory<T>,
    S: Surrogate<T>,
    A: AcqFunction<T>,
{
    data_type: PhantomData<T>,
    doe: D,
    mem: M,
    acq_func: A,
    surrogate: S,
}

impl<T, D, M, S, A> SMBO<T, D, M, S, A>
where
    // Self: AskTell<T>,
    T: Scalar,
    D: DoE<T>,
    S: Surrogate<T>,
    M: ObservationMemory<T>,
    A: AcqFunction<T>,
{
    pub fn new() -> SMBO<T, D, M, S, A> {
        todo!()
    }

    fn optimize(mut self) {
        // let doe = self.doe.build_DoE();
    }

    fn test(mut self) {
        // let x: Vector<T> = self.surrogate.ask().into();
        // self.surrogate.tell(x, T);
    }
}

// impl<T, S, A> AskTell<T> for SMBO<T, S, A>
// where
//     T: Scalar,
//     S: Surrogate<T>,
//     A: AcqFunction,
// {
//     fn ask(&mut self) -> impl Input<T> {
//         todo!()
//     }

//     fn tell(&mut self, x: impl Input<T>, y: T) {
//         todo!()
//     }
// }
