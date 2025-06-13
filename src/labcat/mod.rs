#![allow(non_snake_case)]
use itertools::Itertools;
use ord_subset::{OrdSubset, OrdSubsetIterExt};

use crate::{
    bounds::{Bounds, ContinuousBounds, UpperLowerBounds}, doe::{DoE, DoeIter, RandomSampling}, dtype, ei::{AcqFunction, EI}, gp::GP, kernel::{BayesianKernel, Kernel, ARD}, labcat::{memory::LabcatMemory, tune::LABCAT_GPTune}, lhs::LHS, memory::{
        BaseMemory, Memory, ObservationIO, ObservationInputRecenter, ObservationInputRescale,
        ObservationInputRotate, ObservationOutputRecenter, ObservationOutputRescale,
        ObservationTransform,
    }, sqexp::SqExpARD, tune::{SurrogateTuning, TuningStrategy}, AskTell, Refit, Surrogate, SurrogateIO
};

pub mod memory;
pub mod tune;

pub type LABCAT<T> = GenericLABCAT::<T, GP<T, SqExpARD<T>, LabcatMemory<T>>, LABCAT_GPTune<T>, EI<T>, ContinuousBounds<T>, LHS<T>>;

#[derive(Debug, Clone)]
pub struct GenericLABCAT<T, S, H, A, B, D>
where
    T: dtype,
    S: SurrogateIO<T> + Kernel<T, KernType: ARD<T>> + Memory<T, MemType = LabcatMemory<T>>,
    H: SurrogateTuning<T, S>,
    A: AcqFunction<T, S>,
    B: Bounds<T>,
    D: DoE<T>,
{
    bounds: B,
    tr: ContinuousBounds<T>,
    doe: D,
    doe_iter: DoeIter<D, T>,
    mem: BaseMemory<T>,
    acq: A,
    surrogate: S,
    tuning_strategy: H,
    init_flag: bool,
    f_init: fn(usize) -> usize,
    rho: usize,
}

impl<T, S, H, A, B, D> GenericLABCAT<T, S, H, A, B, D>
where
    T: dtype,
    S: SurrogateIO<T> + Kernel<T, KernType: ARD<T>> + Memory<T, MemType = LabcatMemory<T>>,
    H: SurrogateTuning<T, S> + Default,
    A: AcqFunction<T, S> + Default,
    B: Bounds<T> + UpperLowerBounds<T>,
    D: DoE<T>,
{
    pub fn new(d: usize, beta: T, rho: usize, bounds: B) -> GenericLABCAT<T, S, H, A, B, D> {
        let tr = ContinuousBounds::<T>::scaled_unit(d, beta);

        let f_init = |d| 2 * d + 1;
        let mut doe = D::default();
        doe.build_DoE(f_init(d), &bounds);

        Self {
            bounds,
            tr,
            doe_iter: doe.iter(),
            doe,
            mem: BaseMemory::new(d),
            acq: A::default(),
            surrogate: S::new(d),
            tuning_strategy: H::default(),
            init_flag: false,
            rho,
            f_init,
        }
    }
}

impl<T, S, H, A, B, D> AskTell<T> for GenericLABCAT<T, S, H, A, B, D>
where
    T: dtype + OrdSubset,
    S: SurrogateIO<T>
        + Kernel<T, KernType: ARD<T> + BayesianKernel<T>>
        + Memory<T, MemType = LabcatMemory<T>>
        + Refit<T>,
    H: SurrogateTuning<T, S>,
    A: AcqFunction<T, S>,
    B: Bounds<T> + UpperLowerBounds<T>,
    D: DoE<T>,
{
    type Ask = anyhow::Result<Vec<T>>;
    type Tell = anyhow::Result<()>;
    
    
    fn ask(&mut self) -> Self::Ask {
        if self.doe_iter.len() != 0 {
            return Ok(self
                .doe_iter
                .next()
                .expect("Should always yield next input point!"))
        }

        let mut random_ei_pts = RandomSampling::default();
        random_ei_pts.build_DoE(10 * self.bounds.dim(), &self.tr);

        let a = random_ei_pts
            .DoE()
            .col_iter()
            .enumerate()
            .filter_map(|(i, col)| {
                self.acq
                    .probe(self.surrogate(), col.try_as_col_major().unwrap().as_slice())
                    .and_then(|acq| (i, acq).into())
            })
            .ord_subset_max_by_key(|&(_, ei)| ei)
            .unwrap();

        Ok(random_ei_pts
            .DoE()
            .col(a.0)
            .try_as_col_major()
            .unwrap()
            .as_slice()
            .to_vec())
    }

    fn tell(&mut self, x: &[T], y: &T) -> Self::Tell {
        self.mem.append(x, y);
        self.surrogate.memory_mut().append(x, y);

        // Guard clause: Do not update model if still in init phase
        if self.doe.n() > self.mem.n() {
            return Ok(());
        }

        // If finished with DoE, initialize transforms once
        if !self.init_flag {
            self.surrogate.memory_mut().reset_transform();
            self.surrogate.memory_mut().recenter_X();
            let axis_lens = self.bounds.lb_ub().map(|(lb, ub)| ub.abs_sub(*lb)).collect_vec();
            self.surrogate.memory_mut().rescale_X_with(&axis_lens);
            self.init_flag = true;
        }

        // Normalize Y
        self.surrogate.memory_mut().recenter_Y();
        self.surrogate.memory_mut().rescale_Y();

        // Recenter and rotate X
        self.surrogate.memory_mut().recenter_X();
        self.surrogate.memory_mut().rotate_X();

        // // Refit surrogate
        // self.surrogate.refit().unwrap();

        // Find most likely length-scales
        let _ = self.tuning_strategy.tune(&mut self.surrogate); // 5 fail
        let l = self.surrogate().kernel().l().to_owned();

        // Rescale X
        self.surrogate.memory_mut().rescale_X_with(&l);

        // Discard observations over rho * d
        // TODO: impl m parameter
        self.surrogate
            .memory_mut()
            .tr_discard_with_retain(&self.tr, self.bounds.dim() * self.rho);

        self.surrogate.refit()?; // 1 fail

        Ok(())
    }
}

impl<T, S, H, A, B, D> Surrogate<T> for GenericLABCAT<T, S, H, A, B, D>
where
    T: dtype,
    S: SurrogateIO<T> + Kernel<T, KernType: ARD<T>> + Memory<T, MemType = LabcatMemory<T>>,
    H: SurrogateTuning<T, S>,
    A: AcqFunction<T, S>,
    B: Bounds<T>,
    D: DoE<T>,
{
    type SurType = S;

    fn surrogate(&self) -> &Self::SurType {
        &self.surrogate
    }

    fn surrogate_mut(&mut self) -> &mut Self::SurType {
        &mut self.surrogate
    }
}

impl<T, S, H, A, B, D> Memory<T> for GenericLABCAT<T, S, H, A, B, D>
where
    T: dtype,
    S: SurrogateIO<T> + Kernel<T, KernType: ARD<T>> + Memory<T, MemType = LabcatMemory<T>>,
    H: SurrogateTuning<T, S>,
    A: AcqFunction<T, S>,
    B: Bounds<T>,
    D: DoE<T>,
{
    type MemType = BaseMemory<T>;

    fn memory(&self) -> &Self::MemType {
        &self.mem
    }

    fn memory_mut(&mut self) -> &mut Self::MemType {
        &mut self.mem
    }
}

impl<T, S, H, A, B, D> TuningStrategy<T, S> for GenericLABCAT<T, S, H, A, B, D>
where
    T: dtype,
    S: SurrogateIO<T> + Kernel<T, KernType: ARD<T>> + Memory<T, MemType = LabcatMemory<T>>,
    H: SurrogateTuning<T, S>,
    A: AcqFunction<T, S>,
    B: Bounds<T>,
    D: DoE<T>,
{
    type TuningType = H;

    fn tuning_strategy(&self) -> &Self::TuningType {
        &self.tuning_strategy
    }

    fn tuning_strategy_mut(&mut self) -> &mut Self::TuningType {
        &mut self.tuning_strategy
    }
}
