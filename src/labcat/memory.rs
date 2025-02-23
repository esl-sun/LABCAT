use faer::{unzip, zip, Col, Mat, MatMut, MatRef, Row};
use ord_subset::OrdSubset;

use crate::{
    bounds::Bounds,
    dtype,
    memory::{
        BaseMemory, ObservationDiscard, ObservationIO, ObservationInputRecenter,
        ObservationInputRescale, ObservationInputRotate, ObservationMaxMin, ObservationMean,
        ObservationOutputRecenter, ObservationOutputRescale, ObservationTransform,
        ObservationVariance,
    },
    utils::{MatMutUtils, MatRefUtils, ColRefUtils},
};

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

impl<T> LabcatMemory<T>
where
    T: dtype,
{
    pub fn tr_discard_with_retain(&mut self, tr: &impl Bounds<T>, n_retain: usize) {
        let idx_discard = self
            .X()
            .cols()
            .enumerate()
            .filter(|(_, col)| !tr.inside(col.as_slice()))
            .map(|(i, _)| i)
            .take(self.n().saturating_sub(n_retain))
            .collect();

        self.discard_mult(idx_discard);
    }
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

    fn i(&self, i: usize) -> (&[T], &T) {
        self.base_mem.i(i)
    }

    fn X(&self) -> MatRef<T> {
        self.base_mem.X()
    }

    fn X_mut(&mut self) -> MatMut<T> {
        self.base_mem.X_mut()
    }

    fn Y(&self) -> &[T] {
        self.base_mem.Y()
    }

    fn Y_mut(&mut self) -> &mut [T] {
        self.base_mem.Y_mut()
    }

    fn append(&mut self, x: &[T], y: &T) {
        let x = &self.R_inv * &self.S_inv * (faer::ColRef::from_slice(x) - &self.X_offset);
        let y = (*y - self.y_offset) / self.y_scale;
        self.base_mem.append(x.as_ref().as_slice(), &y)
    }

    fn append_mult(&mut self, X: MatRef<T>, Y: &[T]) {
        self.base_mem.append_mult(X, Y);
        // todo!() //TODO: transform
    }
}

impl<T: dtype> ObservationDiscard<T> for LabcatMemory<T> {
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

impl<T: dtype> ObservationMean<T> for LabcatMemory<T> {}

impl<T: dtype + OrdSubset> ObservationMaxMin<T> for LabcatMemory<T> {}

impl<T: dtype> ObservationVariance<T> for LabcatMemory<T> {}

impl<T: dtype> ObservationTransform<T> for LabcatMemory<T> {
    fn x_prime(&self) -> impl Fn(&[T]) -> &[T] {
        // faer_core::col::from_slice(slice)
        // |x| faer_core::col::from_slice(x).as_slice() //TODO: FIX
        todo!();
        |x| x
    }

    fn X_prime(&self) -> Mat<T> {
        let mut X_prime = self.R.as_ref() * self.S.as_ref() * self.base_mem.X().as_ref();
        X_prime.as_mut().cols_mut().for_each(|col| {
            zip!(col, self.X_offset.as_ref())
                .for_each(|unzip!(mut col, off)| *col = *col + *off)
        });

        X_prime
    }

    fn y_prime(&self) -> impl Fn(&T) -> T {
        |y| (*y * self.y_scale) + self.y_offset
    }

    fn Y_prime(&self) -> Row<T> {
        Row::<T>::from_fn(self.n(), |i| self.y_prime()(&self.Y()[i]))
    }

    fn reset_transform(&mut self) {
        self.base_mem.X = self.X_prime();
        self.base_mem.Y = self.Y_prime();

        self.R = Mat::identity(self.dim(), self.dim());
        self.R_inv = Mat::identity(self.dim(), self.dim());
        self.S = Mat::identity(self.dim(), self.dim());
        self.S_inv = Mat::identity(self.dim(), self.dim());
        self.X_offset = Col::zeros(self.dim());
        self.y_offset = T::zero();
        self.y_scale = T::one();
    }
}

impl<T: dtype> ObservationInputRecenter<T> for LabcatMemory<T> {
    fn recenter_X_with(&mut self, cen: &[T]) {
        let cen = faer::ColRef::from_slice(cen);

        self.base_mem.X.as_mut().cols_mut().for_each(|col| {
            zip!(col, cen).for_each(|unzip!(mut col, cen)| *col = *col - *cen )
        });

        self.X_offset += self.R.as_ref() * self.S.as_ref() * cen;
    }
}

impl<T: dtype> ObservationInputRescale<T> for LabcatMemory<T> {
    fn rescale_X_with(&mut self, l: &[T]) {
        #[cfg(debug_assertions)]
        if l.len() != self.dim() {
            panic!("Dimensions of new rescaling slice and memory do not match!");
        }

        let l = faer::ColRef::from_slice(l);

        //TODO: Avoid ref to private member?
        self.base_mem.X.cols_mut().for_each(|col| {
            zip!(col, l).for_each(|unzip!(mut col, l)| *col = (*col) / (*l));
        });

        zip!(
            self.S.as_mut().diagonal_mut().column_vector_mut(),
            self.S_inv.as_mut().diagonal_mut().column_vector_mut(),
            l
        )
        .for_each(|unzip!(mut S, mut S_inv, l)| {
            *S = (*S) * (*l);
            *S_inv = (*S_inv) / (*l)
        });
    }
}

impl<T: dtype> ObservationInputRotate<T> for LabcatMemory<T> {
    fn rotate_X(&mut self) {
        let mut W = Mat::<T>::identity(self.n(), self.n());
        zip!(
            W.as_mut().diagonal_mut().column_vector_mut(),
            faer::ColRef::from_slice(self.Y())
        )
        .for_each(|unzip!(mut W, y)| *W = T::one() - *y);

        let svd = (self.S.as_ref() * self.X().as_ref() * W.as_ref()).svd();
        let u = svd.unwrap().U();

        self.R = self.R.as_ref() * u;
        self.R_inv = u.transpose() * self.R.as_ref();

        self.base_mem.X =
            self.S_inv.as_ref() * u.transpose() * self.S.as_ref() * self.base_mem.X.as_ref();
    }
}

impl<T: dtype> ObservationOutputRecenter<T> for LabcatMemory<T> {
    fn recenter_Y_with(&mut self, cen: &T) {
        zip!(self.base_mem.Y.as_mut()).for_each(|unzip!(mut y)| *y = *y - *cen);

        self.y_offset = self.y_offset + self.y_scale.mul(*cen);
    }
}

impl<T: dtype> ObservationOutputRescale<T> for LabcatMemory<T> {
    fn rescale_Y_with(&mut self, l: &T) {
        // TODO: check cap on l if all equal

        // Cannot divide by zero
        if *l == T::zero() {
            return;
        }

        zip!(self.base_mem.Y.as_mut()).for_each(|unzip!(y)| *y = *y * l.recip());

        self.y_scale = self.y_scale.mul(*l);
    }
}
