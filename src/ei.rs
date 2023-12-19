use std::marker::PhantomData;

use nalgebra::Scalar;

use crate::{BayesianSurrogate, Surrogate};

pub trait AcqFunction<T>
where
    T: Scalar,
{
    fn probe_acq(&self, x: crate::utils::Vector<T>) -> T;
}

pub struct EI<'a, T, S>
where
    T: Scalar,
    S: Surrogate<T> + BayesianSurrogate<T>,
{
    data_type: PhantomData<T>,
    surrogate: &'a S,
}

impl<T, S> AcqFunction<T> for EI<'_, T, S>
where
    T: Scalar,
    S: Surrogate<T> + BayesianSurrogate<T>,
{
    fn probe_acq(&self, x: crate::utils::Vector<T>) -> T {
        let mean = self.surrogate.probe(&x);
        let var = self.surrogate.probe_variance(&x);
        todo!()
    }
}
