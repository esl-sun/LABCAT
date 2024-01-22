use std::marker::PhantomData;

use lazy_static::lazy_static;
use ord_subset::OrdSubset;
use statrs::distribution::{Continuous, ContinuousCDF, Normal};
// use rand_distr::{Normal, Distribution};

use crate::{
    dtype,
    memory::{ObservationIO, ObservationMaxMin},
    tpe::TPESurrogate,
    BayesianSurrogate, Memory, Surrogate,
};

pub trait AcqFunction<T, M, S>
where
    T: dtype,
    M: ObservationIO<T>,
{
    fn probe_acq(&self, surrogate: &S, x: &[T]) -> Option<T>;
}

pub trait AcqJacobian<T, M, S>: AcqFunction<T, M, S>
where
    T: dtype,
    M: ObservationIO<T>,
    S: Surrogate<T, M>,
{
    fn acq_jacobian(&self, surrogate: &S, x: &[T]) -> &[T];
}

lazy_static! {
    static ref NORMAL: Normal = Normal::new(0.0, 1.0).unwrap();
}

pub struct EI<T>
where
    T: dtype + OrdSubset,
{
    data_type: PhantomData<T>,
    xi: T,
}

impl<T> Default for EI<T>
where
    T: dtype + OrdSubset,
{
    fn default() -> Self {
        Self {
            data_type: Default::default(),
            xi: T::zero(),
        }
    }
}

impl<T> EI<T>
where
    T: dtype + OrdSubset,
{
    fn new(xi: T) -> Self {
        Self {
            data_type: PhantomData,
            xi,
        }
    }
}

impl<T, M, S> AcqFunction<T, M, S> for EI<T>
where
    T: dtype + OrdSubset,
    M: ObservationIO<T> + ObservationMaxMin<T>,
    S: Surrogate<T, M> + BayesianSurrogate<T, M> + Memory<T, M>,
{
    fn probe_acq(&self, surrogate: &S, x: &[T]) -> Option<T> {
        let mean = surrogate.probe(x)?;
        let sigma = surrogate.probe_variance(x)?.sqrt();
        let min = *surrogate
            .memory()
            .min()
            .expect("Obeservations must not be empty!")
            .2;

        let z = (min - mean - self.xi) / sigma;

        let cdf =
            T::from_f64(NORMAL.cdf(T::to_f64(&z).expect("Converting `T` to f64 must not fail.")))
                .expect("Converting f64 to `T` must not fail.");
        let pdf =
            T::from_f64(NORMAL.pdf(T::to_f64(&z).expect("Converting `T` to f64 must not fail.")))
                .expect("Converting f64 to `T` must not fail.");

        Some(sigma * (z * cdf + pdf))
    }
}

// impl AcqJacobian for EI<T>

pub struct TPE_EI<T>
where
    T: dtype + OrdSubset,
{
    data_type: PhantomData<T>,
}

impl<T> Default for TPE_EI<T>
where
    T: dtype + OrdSubset,
{
    fn default() -> Self {
        Self {
            data_type: Default::default(),
        }
    }
}

impl<T> TPE_EI<T>
where
    T: dtype + OrdSubset,
{
    fn new() -> Self {
        Self {
            data_type: PhantomData,
        }
    }
}

impl<T, M, S> AcqFunction<T, M, S> for TPE_EI<T>
where
    T: dtype + OrdSubset,
    M: ObservationIO<T> + ObservationMaxMin<T>,
    S: TPESurrogate<T, M> + Memory<T, M>,
{
    fn probe_acq(&self, surrogate: &S, x: &[T]) -> Option<T> {
        Some(surrogate.l().probe(x)? / surrogate.g().probe(x)?)
    }
}
