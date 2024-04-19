use std::marker::PhantomData;

use lazy_static::lazy_static;
use ord_subset::OrdSubset;
use statrs::distribution::{Continuous, ContinuousCDF, Normal};
// use rand_distr::{Normal, Distribution};

use crate::{
    dtype, memory::ObservationMaxMin, tpe::TPESurrogate, BayesianSurrogateIO, Memory, SurrogateIO,
};

pub trait AcqFunction<T, S>
where
    T: dtype,
    S: SurrogateIO<T>,
{
    fn probe(&self, surrogate: &S, x: &[T]) -> Option<T>;
}

pub trait AcqJacobian<T, S>: AcqFunction<T, S>
where
    T: dtype,
    S: SurrogateIO<T>,
{
    fn acq_jacobian(&self, surrogate: &S, x: &[T]) -> Option<&[T]>;
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
    pub fn new(xi: T) -> Self {
        Self {
            data_type: PhantomData,
            xi,
        }
    }
}

impl<T, S> AcqFunction<T, S> for EI<T>
where
    T: dtype + OrdSubset,
    S: SurrogateIO<T> + BayesianSurrogateIO<T> + Memory<T, MemType: ObservationMaxMin<T>>,
{
    fn probe(&self, surrogate: &S, x: &[T]) -> Option<T> {
        let mean = surrogate.probe(x)?;
        let sigma = surrogate.probe_variance(x)?.sqrt();
        let min = *surrogate
            .memory()
            .min_y()
            .expect("Obeservations must not be empty!");

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

// impl<T, S> AcqJacobian<T, S> for EI<T>
// where
//     T: dtype + OrdSubset,
//     S: GPSurrogate<T> + Memory<T, MemType: ObservationMaxMin<T>>,
// {
//     fn acq_jacobian(&self, surrogate: &S, x: &[T]) -> Option<&[T]> {
//         let mean = surrogate.probe(x)?;
//         let sigma = surrogate.probe_variance(x)?.sqrt();
//         let min = *surrogate
//             .memory()
//             .min_y()
//             .expect("Obeservations must not be empty!");

//         let z = (min - mean - self.xi) / sigma;

//         let cdf =
//             T::from_f64(NORMAL.cdf(T::to_f64(&z).expect("Converting `T` to f64 must not fail.")))
//                 .expect("Converting f64 to `T` must not fail.");
//         let pdf =
//             T::from_f64(NORMAL.pdf(T::to_f64(&z).expect("Converting `T` to f64 must not fail.")))
//                 .expect("Converting f64 to `T` must not fail.");

//         // let dalpha
//         todo!()
//     }
// }

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

impl<T, S> AcqFunction<T, S> for TPE_EI<T>
where
    T: dtype + OrdSubset,
    S: SurrogateIO<T> + TPESurrogate<T>,
{
    fn probe(&self, surrogate: &S, x: &[T]) -> Option<T> {
        Some(surrogate.l().probe(x)? / surrogate.g().probe(x)?)
    }
}
