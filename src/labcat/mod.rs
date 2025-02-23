#![allow(non_snake_case)]
use ord_subset::{OrdSubset, OrdSubsetIterExt};

use crate::{
    bounds::{Bounds, ContinuousBounds, UpperLowerBounds},
    doe::DoE,
    dtype,
    ei::AcqFunction,
    kernel::{BayesianKernel, Kernel, ARD},
    labcat::memory::LabcatMemory,
    lhs::RandomSampling,
    memory::{
        BaseMemory, Memory, ObservationIO, ObservationInputRecenter, ObservationInputRescale,
        ObservationInputRotate, ObservationOutputRecenter, ObservationOutputRescale,
        ObservationTransform,
    },
    tune::{SurrogateTuning, TuningStrategy},
    utils::{ColRefUtils, MatRefUtils},
    AskTell, Refit, Surrogate, SurrogateIO,
};

pub mod memory;
pub mod tune;

#[derive(Debug, Clone)]
pub struct LABCAT<T, S, H, A, B, D>
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
    mem: BaseMemory<T>,
    acq: A,
    surrogate: S,
    tuning_strategy: H,
    f_init: fn(usize) -> usize,
    f_discard: fn(usize) -> usize,
}

impl<T, S, H, A, B, D> LABCAT<T, S, H, A, B, D>
where
    T: dtype,
    S: SurrogateIO<T> + Kernel<T, KernType: ARD<T>> + Memory<T, MemType = LabcatMemory<T>>,
    H: SurrogateTuning<T, S> + Default,
    A: AcqFunction<T, S> + Default,
    B: Bounds<T> + UpperLowerBounds<T>,
    D: DoE<T>,
{
    pub fn new(d: usize, beta: T, bounds: B) -> LABCAT<T, S, H, A, B, D> {
        let tr = ContinuousBounds::<T>::scaled_unit(d, beta);

        let f_init = |d| 2 * d + 1;
        let mut doe = D::default();
        doe.build_DoE(f_init(d), &bounds);

        Self {
            bounds,
            tr,
            doe,
            mem: BaseMemory::new(d),
            acq: A::default(),
            surrogate: S::new(d),
            tuning_strategy: H::default(),
            f_init,
            f_discard: |d| 7 * d,
        }
    }
}

impl<T, S, H, A, B, D> AskTell<T> for LABCAT<T, S, H, A, B, D>
where
    T: dtype + OrdSubset,
    S: SurrogateIO<T>
        + Kernel<T, KernType: ARD<T> + BayesianKernel<T>>
        + Memory<T, MemType = LabcatMemory<T>>
        + Refit<T>,
    H: SurrogateTuning<T, S>,
    A: AcqFunction<T, S>,
    B: Bounds<T>,
    D: DoE<T>,
{
    fn ask(&mut self) -> Vec<T> {
        dbg!("ASK START");

        if let Some(doe_x) = self.doe.get(self.mem.n()) {
            return doe_x.to_vec();
        }

        let mut random_ei_pts = RandomSampling::default();
        random_ei_pts.build_DoE(10 * self.bounds.dim(), &self.tr);

        dbg!("EI INIT");

        let a = random_ei_pts
            .DoE()
            .cols()
            .enumerate()
            .map(|(i, col)| {
                let acq = self.acq.probe(self.surrogate(), col.as_slice()).unwrap();
                (i, acq)
            })
            .ord_subset_max_by_key(|&(_, ei)| ei)
            .unwrap();

        dbg!("EI MAXED");

        // dbg!(a);

        // unsafe {
        //     core::slice::from_raw_parts(random_ei_pts.DoE().col(a.0).as_ptr(), self.bounds.dim())
        // } // UNSAFE UNSAFE UNSAFE

        random_ei_pts.DoE().col(a.0).as_slice().to_vec()
    }

    fn tell(&mut self, x: &[T], y: &T) {
        dbg!("TELL START");
        self.mem.append(x, y);
        self.surrogate.memory_mut().append(x, y);
        dbg!("MEM APPENDED");

        if self.doe.n() > self.mem.n() {
            return;
        }

        self.surrogate.memory_mut().recenter_Y();
        self.surrogate.memory_mut().rescale_Y();
        dbg!("Y RESCALED");
        self.surrogate.memory_mut().recenter_X();
        dbg!("X RECENTERED");
        self.surrogate.memory_mut().rotate_X();
        dbg!("X ROTATED");
        //MAX LOG LIK
        self.tuning_strategy.tune(&mut self.surrogate).unwrap(); // 5 fail
        dbg!("GP TUNED");
        let l = self.surrogate().kernel().l().to_owned();

        self.surrogate.memory_mut().rescale_X_with(&l);
        dbg!("X RESCALED");
        // TODO: impl m parameter
        self.surrogate
            .memory_mut()
            .tr_discard_with_retain(&self.tr, (self.f_discard)(self.bounds.dim()));
        dbg!("MEM DISCARDED");

        self.surrogate.refit().unwrap(); // 1 fail
        dbg!("GP REFITTED");
    }
}

impl<T, S, H, A, B, D> Surrogate<T> for LABCAT<T, S, H, A, B, D>
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

impl<T, S, H, A, B, D> Memory<T> for LABCAT<T, S, H, A, B, D>
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

impl<T, S, H, A, B, D> TuningStrategy<T, S> for LABCAT<T, S, H, A, B, D>
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
