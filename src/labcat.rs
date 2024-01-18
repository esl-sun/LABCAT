#![allow(non_snake_case)]

use faer_core::{Col, Mat, MatRef, Row};

use crate::{dtype, utils::MatMutUtils, memory::{BaseMemory, ObservationIO, ObservationTransform, ObservationMean, ObservationInputRescale}};

#[derive(Clone, Debug)]
struct LabcatMemory<T>
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
        self.base_mem
            .X
            .col_chunks_mut(1)
            .for_each(|mut col| col.zip_apply_with_row_slice(l, |val, l| val / l));

        todo!()
    }

    fn reset_scaling(&mut self) {
        todo!()
    }
}