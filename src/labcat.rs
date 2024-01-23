#![allow(non_snake_case)]

use faer_core::{unzipped, zipped, Col, Mat, MatRef, Row};

use crate::{
    bounds::Bounds, doe::DoE, dtype, ei::AcqFunction, kernel::ARD, memory::{
        BaseMemory, ObservationIO, ObservationInputRescale, ObservationMean, ObservationTransform,
    }, utils::MatMutUtils, AskTell, Kernel, Memory, Refit, Surrogate
};

#[derive(Debug, Clone)]
pub struct LABCAT<T, S, A, B, D>
where
    T: dtype,
    S: Surrogate<T> + Kernel<T, KernType: ARD<T>> + Memory<T, MemType = LabcatMemory<T>>,
    A: AcqFunction<T, S>,
    B: Bounds<T>,
    D: DoE<T>,
{
    bounds: B,
    doe: D,
    mem: BaseMemory<T>,
    acq_func: A,
    surrogate: S,
}

impl<T, S, A, B, D> LABCAT<T, S, A, B, D>
where
    T: dtype,
    S: Surrogate<T> + Kernel<T, KernType: ARD<T>> + Memory<T, MemType = LabcatMemory<T>>,
    A: AcqFunction<T, S>,
    B: Bounds<T>,
    D: DoE<T>,
{
    pub fn new() -> LABCAT<T, S, A, B, D> {
        todo!()
    }

    fn optimize(self) {
        // let doe = self.doe.build_DoE();
        todo!()
    }
}

impl<T, S, A, B, D> AskTell<T> for LABCAT<T, S, A, B, D>
where
    T: dtype,
    S: Surrogate<T> + Kernel<T, KernType: ARD<T>> + Memory<T, MemType = LabcatMemory<T>> + Refit<T>,
    A: AcqFunction<T, S>,
    B: Bounds<T>,
    D: DoE<T>,
{
    fn ask(&mut self) -> Vec<T> {
        todo!()
    }

    fn tell(&mut self, _: &[T], _: &T) {
        
        let l = &self.surrogate.kernel().l();
        let m = &mut self.surrogate.memory();
        m.rescale(&l);
        //....
        self.surrogate.refit();
        
        todo!()
    }
}


#[derive(Clone, Debug)]
pub struct LabcatMemory<T>
where
    T: dtype,
{
    base_mem: BaseMemory<T>,
    R: Mat<T>,
    R_inv: Mat<T>,
    S: Mat<T>,
    S_inv: Mat<T>,
    X_offset: Col<T>,
    y_offset: T,
    y_scale: T,
}

impl<T> ObservationIO<T> for LabcatMemory<T>
where
    Self: ObservationTransform<T>,
    T: dtype,
{
    fn new(d: usize) -> Self {
        Self {
            base_mem: BaseMemory::new(d),
            R: Mat::identity(d, d),
            R_inv: Mat::identity(d, d),
            S: Mat::identity(d, d),
            S_inv: Mat::identity(d, d),
            X_offset: Col::zeros(d),
            y_offset: T::zero(),
            y_scale: T::one(),
        }
    }

    fn dim(&self) -> usize {
        self.base_mem.dim()
    }

    fn n(&self) -> usize {
        self.base_mem.n()
    }

    fn X(&self) -> &Mat<T> {
        self.base_mem.X()
    }

    fn Y(&self) -> &[T] {
        self.base_mem.Y()
    }

    fn append(&mut self, x: &[T], y: T) {
        self.base_mem.append(x, y)
    }

    fn append_mult(&mut self, X: MatRef<T>, Y: &[T]) {
        self.base_mem.append_mult(X, Y)
    }

    fn discard(&mut self, i: usize) {
        self.base_mem.discard(i)
    }

    fn discard_mult(&mut self, idx: Vec<usize>) {
        self.base_mem.discard_mult(idx)
    }

    fn discard_all(&mut self) {
        self.base_mem.discard_all()
    }
}

impl<T> ObservationMean<T> for LabcatMemory<T> where T: dtype {}

impl<T> ObservationTransform<T> for LabcatMemory<T>
where
    T: dtype,
{
    fn x_prime(&self) -> impl Fn(&[T]) -> &[T] {
        // faer_core::col::from_slice(slice)
        // |x| faer_core::col::from_slice(x).as_slice() //TODO: FIX
        |x| x
    }

    fn X_prime(&self) -> Mat<T> {
        // self.R * self.S * self.base_mem.X() + self.X_offset
        todo!()
    }

    fn y_prime(&self) -> impl Fn(&T) -> T {
        |y| (*y * self.y_scale) + self.y_offset
    }

    fn Y_prime(&self) -> Row<T> {
        Row::<T>::from_fn(self.n(), |i| self.y_prime()(&self.Y()[i]))
    }
}

impl<T> ObservationInputRescale<T> for LabcatMemory<T>
where
    T: dtype,
{
    fn rescale(&mut self, l: &[T]) {
        #[cfg(debug_assertions)]
        if l.len() != self.dim() {
            panic!("Dimensions of new rescaling slice and memory do not match!");
        }

        let l = faer_core::col::from_slice::<T>(l);

        //TODO: Avoid ref to private member?
        self.base_mem.X.cols_mut().for_each(|col| {
            zipped!(col, l).for_each(|unzipped!(mut col, l)| col.write(col.read() / l.read()));
        });

        zipped!(
            self.S.as_mut().diagonal_mut().column_vector_mut(),
            self.S_inv.as_mut().diagonal_mut().column_vector_mut(),
            l
        )
        .for_each(|unzipped!(mut S, mut S_inv, l)| {
            S.write(S.read() * l.read());
            S_inv.write(S_inv.read() / l.read())
        });
    }

    fn reset_scaling(&mut self) {
        todo!()
    }
}