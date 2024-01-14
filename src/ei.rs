use std::marker::PhantomData;

use crate::{dtype, BayesianSurrogate, Surrogate};

pub trait AcqFunction<T>
where
    T: dtype,
{
    fn probe_acq(&self, x: &[T]) -> T;
}

pub struct EI<'a, T, S>
where
    T: dtype,
    S: Surrogate<T> + BayesianSurrogate<T>,
{
    data_type: PhantomData<T>,
    surrogate: &'a S,
}

impl<T, S> AcqFunction<T> for EI<'_, T, S>
where
    T: dtype,
    S: Surrogate<T> + BayesianSurrogate<T>,
{
    fn probe_acq(&self, x: &[T]) -> T {
        let mean = self.surrogate.probe(&x);
        let var = self.surrogate.probe_variance(&x);
        todo!()
    }
}
