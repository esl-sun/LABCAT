#![allow(non_snake_case)]
use ord_subset::{OrdSubset, OrdSubsetIterExt};

use crate::{
    bounds::{Bounds, ContinuousBounds, UpperLowerBounds},
    doe::DoE,
    dtype,
    ei::AcqFunction,
    kernel::ARD,
    labcat::memory::LabcatMemory,
    lhs::RandomSampling,
    memory::{
        BaseMemory, ObservationIO, ObservationInputRecenter, ObservationInputRescale,
        ObservationInputRotate, ObservationOutputRecenter, ObservationOutputRescale,
        ObservationTransform,
    },
    utils::{ColRefUtils, MatRefUtils},
    AskTell, Kernel, Memory, Refit, Surrogate, SurrogateIO,
};

pub mod memory;
pub mod tune;

#[derive(Debug, Clone)]
pub struct LABCAT<T, S, A, B, D>
where
    T: dtype,
    S: SurrogateIO<T> + Kernel<T, KernType: ARD<T>> + Memory<T, MemType = LabcatMemory<T>>,
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
    f_init: fn(usize) -> usize,
    f_discard: fn(usize) -> usize,
}

impl<T, S, A, B, D> LABCAT<T, S, A, B, D>
where
    T: dtype,
    S: SurrogateIO<T> + Kernel<T, KernType: ARD<T>> + Memory<T, MemType = LabcatMemory<T>>,
    A: AcqFunction<T, S> + Default,
    B: Bounds<T> + UpperLowerBounds<T>,
    D: DoE<T>,
{
    pub fn new(d: usize, beta: T, bounds: B) -> LABCAT<T, S, A, B, D> {
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
            f_init,
            f_discard: |d| 7 * d,
        }
    }

    fn optimize(self) {
        // let doe = self.doe.build_DoE();
        todo!()
    }
}

impl<T, S, A, B, D> AskTell<T> for LABCAT<T, S, A, B, D>
where
    T: dtype + OrdSubset,
    S: SurrogateIO<T>
        + Kernel<T, KernType: ARD<T>>
        + Memory<T, MemType = LabcatMemory<T>>
        + Refit<T>,

    A: AcqFunction<T, S>,
    B: Bounds<T>,
    D: DoE<T>,
{
    fn ask(&mut self) -> &[T] {
        let mut random_ei_pts = RandomSampling::default();
        random_ei_pts.build_DoE(10 * self.bounds.dim(), &self.tr);

        dbg!(self.memory().X());
        dbg!(self.surrogate().memory().X());
        dbg!(self.surrogate().memory().X_prime());

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

        dbg!(a);

        unsafe {
            core::slice::from_raw_parts(random_ei_pts.DoE().col(a.0).as_ptr(), self.bounds.dim())
        } // UNSAFE UNSAFE UNSAFE
    }

    fn tell(&mut self, x: &[T], y: &T) {
        //TODO: Handle DoE init

        self.memory_mut().append(x, y);
        self.surrogate_mut().memory_mut().append(x, y);
        // TODO: abstract LABCAT routine into trait

        self.surrogate_mut().memory_mut().recenter_Y();
        self.surrogate_mut().memory_mut().rescale_Y();

        self.surrogate_mut().memory_mut().recenter_X();

        self.surrogate_mut().memory_mut().rotate_X();

        //MAX LOG LIK
        // self.surrogate_mut().tune().unwrap();

        let l = self.surrogate().kernel().l().to_owned();

        self.surrogate_mut().memory_mut().rescale_X_with(&l);

        // TODO: impl m parameter
        let _idx_discard: Vec<usize> = self
            .surrogate()
            .memory()
            .X()
            .cols()
            .inspect(|x| {
                dbg!(x);
            })
            .enumerate()
            .filter(|(_, col)| !self.tr.inside(col.as_slice()))
            .map(|(i, _)| i)
            .take((self.f_discard)(self.bounds.dim()))
            .collect();

        // self.surrogate_mut().memory_mut().discard_mult(idx_discard);
        dbg!("HERE4");
        //....
        self.surrogate_mut().refit().unwrap();

        // todo!()
    }
}

impl<T, S, A, B, D> Surrogate<T> for LABCAT<T, S, A, B, D>
where
    T: dtype,
    S: SurrogateIO<T> + Kernel<T, KernType: ARD<T>> + Memory<T, MemType = LabcatMemory<T>>,
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

impl<T, S, A, B, D> Memory<T> for LABCAT<T, S, A, B, D>
where
    T: dtype,
    S: SurrogateIO<T> + Kernel<T, KernType: ARD<T>> + Memory<T, MemType = LabcatMemory<T>>,
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
