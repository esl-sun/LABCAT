use anyhow::Result;

use crate::{dtype, SurrogateIO};

pub struct NoTuning {}

impl<T, S> Tune<T, S> for NoTuning
where
    T: dtype,
    S: SurrogateIO<T>,
{
    fn tune(_: &mut S) -> Result<()> {
        Ok(())
    }
}

pub trait Tune<T, S>
where
    T: dtype,
    S: SurrogateIO<T>,
{
    fn tune(sur: &mut S) -> Result<()>;
}

pub trait SurrogateTune<T>
where
    Self: SurrogateIO<T> + Sized,
    T: dtype,
{
    type TuningStrategy: Tune<T, Self> = NoTuning;

    fn tune(&mut self) -> Result<()> {
        Self::TuningStrategy::tune(self)
    }
}
