use anyhow::Result;

use crate::{dtype, SurrogateIO};

pub trait TuningStrategy<T, S>
where
    T: dtype,
    S: SurrogateIO<T>,
{
    type TuningType: SurrogateTuning<T, S>;

    fn tuning_strategy(&self) -> &Self::TuningType;
    fn tuning_strategy_mut(&mut self) -> &mut Self::TuningType;
}

pub trait SurrogateTuning<T, S>
where
    T: dtype,
    S: SurrogateIO<T>,
{
    fn tune(&self, sur: &mut S) -> Result<()>;
}

#[derive(Debug, Clone, Default)]
pub struct NoTuning {}

impl<T, S> SurrogateTuning<T, S> for NoTuning
where
    T: dtype,
    S: SurrogateIO<T>,
{
    fn tune(&self, _: &mut S) -> Result<()> {
        Ok(())
    }
}
