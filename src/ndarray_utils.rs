#![allow(non_snake_case)]

use std::{iter::Sum, ops::Neg};

use faer::IntoNdarray;
use faer_core::{ColMut, ColRef, Entity, Mat, RowMut, RowRef, SimpleEntity};
use ndarray::{
    s, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, ArrayViewMut2, Axis, DataMut, DataOwned,
    Dimension, LinalgScalar, ShapeBuilder,
};
use ndarray_linalg::{Scalar, UPLO};
use num_traits::real::Real;
use ord_subset::{OrdSubset, OrdSubsetIterExt};

///////////////////////////////////////////////////////

pub trait IntoOwnedNdarray {
    type Ndarray;
    #[track_caller]
    fn into_ndarray(self) -> Self::Ndarray;
}

impl<T> IntoOwnedNdarray for Mat<T>
where
    T: Entity + SimpleEntity,
{
    type Ndarray = Array2<T>;

    #[track_caller]
    fn into_ndarray(self) -> Self::Ndarray {
        self.as_ref().into_ndarray().into_owned()
    }
}

///////////////////////////////////////////////////////

pub trait RowColIntoNdarray {
    type Ndarray;
    #[track_caller]
    fn into_ndarray(self) -> Self::Ndarray;
}

impl<'a, T> RowColIntoNdarray for ColRef<'a, T>
where
    T: Entity + SimpleEntity,
{
    type Ndarray = ArrayView2<'a, T>;

    #[track_caller]
    fn into_ndarray(self) -> Self::Ndarray {
        let nrows = self.nrows();
        let ptr = self.as_ptr();
        unsafe { ArrayView2::<'_, T>::from_shape_ptr((nrows, 1_usize).into_shape(), ptr) }
    }
}

impl<'a, T> RowColIntoNdarray for ColMut<'a, T>
where
    T: Entity + SimpleEntity,
{
    type Ndarray = ArrayViewMut2<'a, T>;

    #[track_caller]
    fn into_ndarray(self) -> Self::Ndarray {
        let nrows = self.nrows();
        let ptr = self.as_ptr_mut();
        unsafe { ArrayViewMut2::<'_, T>::from_shape_ptr((nrows, 1_usize).into_shape(), ptr) }
    }
}

impl<'a, T> RowColIntoNdarray for RowRef<'a, T>
where
    T: Entity + SimpleEntity,
{
    type Ndarray = ArrayView2<'a, T>;

    #[track_caller]
    fn into_ndarray(self) -> Self::Ndarray {
        let ncols = self.ncols();
        let ptr = self.as_ptr();
        unsafe { ArrayView2::<'_, T>::from_shape_ptr((1_usize, ncols).into_shape(), ptr) }
    }
}

impl<'a, T> RowColIntoNdarray for RowMut<'a, T>
where
    T: Entity + SimpleEntity,
{
    type Ndarray = ArrayViewMut2<'a, T>;

    #[track_caller]
    fn into_ndarray(self) -> Self::Ndarray {
        let ncols = self.ncols();
        let ptr = self.as_ptr_mut();
        unsafe { ArrayViewMut2::<'_, T>::from_shape_ptr((1, ncols).into_shape(), ptr) }
    }
}

///////////////////////////////////////////////////////

pub trait ArrayBaseUtils<D, T, S>
where
    D: Dimension,
    T: Scalar,
    S: DataMut<Elem = T>,
{
    fn ln(self) -> Self
    where
        T: Real;
    fn exp(self) -> Self
    where
        T: Real;
    fn rem_indices_axis(&mut self, indices: Vec<usize>, axis: Axis)
    where
        S: DataOwned + DataMut;
    fn max(&self) -> Option<&T>
    where
        T: OrdSubset;
    fn min(&self) -> Option<&T>
    where
        T: OrdSubset;
    fn indexed_max(&self) -> Option<(D::Pattern, &T)>
    where
        T: OrdSubset;
    fn indexed_min(&self) -> Option<(D::Pattern, &T)>
    where
        T: OrdSubset;
}

/// # Methods For All Array Types
impl<D, T, S> ArrayBaseUtils<D, T, S> for ArrayBase<S, D>
where
    D: Dimension,
    T: Scalar, //+ ndarray_linalg::Scalar + PartialOrd + Float,
    S: DataMut<Elem = T>,
{
    #[inline(always)]
    fn ln(mut self) -> Self
    where
        T: Real,
    {
        self.mapv_inplace(|A| Real::ln(A));
        self
    }

    #[inline(always)]
    fn exp(mut self) -> Self
    where
        T: Real,
    {
        self.mapv_inplace(|A| Real::exp(A));
        self
    }

    #[inline(always)]
    fn rem_indices_axis(&mut self, mut indices: Vec<usize>, axis: Axis)
    where
        S: DataOwned + DataMut,
    {
        indices.sort(); //order so that method works
        indices.dedup(); //remove duplicates

        indices.retain(|i| *i < self.len_of(axis));

        indices
            .iter()
            .rev()
            .for_each(|i| self.remove_index(axis, *i))

        // Array2::from_shape_vec(sh, red_vec).expect("Should never fail.")
    }

    #[inline(always)]
    fn max(&self) -> Option<&T>
    where
        T: OrdSubset,
    {
        // self.iter().max_by(|&x, &y| x.partial_cmp(y)?)
        self.iter().ord_subset_max()
    }

    #[inline(always)]
    fn min(&self) -> Option<&T>
    where
        T: OrdSubset,
    {
        self.iter().ord_subset_min()
    }

    #[inline(always)]
    fn indexed_max(&self) -> Option<(D::Pattern, &T)>
    where
        T: OrdSubset,
    {
        self.indexed_iter().ord_subset_max_by_key(|&(_, x)| x)
    }

    #[inline(always)]
    fn indexed_min(&self) -> Option<(D::Pattern, &T)>
    where
        T: OrdSubset,
    {
        self.indexed_iter().ord_subset_min_by_key(|&(_, x)| x)
    }
}

pub trait Array1Utils<T>
where
    T: Scalar,
{
    fn into_col(self) -> Array2<T>;
    fn rem_at_indices(&mut self, indices: Vec<usize>);
}

impl<T> Array1Utils<T> for Array1<T>
where
    T: Scalar,
{
    #[inline(always)]
    fn into_col(self) -> Array2<T> {
        let len = self.len();
        self.into_shape((len, 1)).expect("Should never fail")
    }

    #[inline(always)]
    fn rem_at_indices(&mut self, mut indices: Vec<usize>) {
        indices.sort(); //order so that method works
        indices.dedup(); //remove duplicates

        indices
            .iter()
            .rev()
            .for_each(|i| self.remove_index(Axis(0), *i))
    }
}

pub trait ArrayView1Utils<T>
where
    T: Scalar,
{
    fn as_col(&self) -> ArrayView2<T>;
}

impl<T> ArrayView1Utils<T> for ArrayView1<'_, T>
where
    T: Scalar,
{
    #[inline(always)]
    fn as_col(&self) -> ArrayView2<T> {
        self.slice(s![.., ndarray::NewAxis])
    }
}

pub trait Array2Utils<T>
where
    T: Scalar,
{
    fn scaled_add_Array1(self, alpha: T, rhs: &Array1<T>, axis: Axis) -> Array2<T>
    where
        T: LinalgScalar;
    fn sub_column(self, rhs: &Array1<T>) -> Array2<T>
    where
        T: LinalgScalar + Neg<Output = T>;
    fn add_column(self, rhs: &Array1<T>) -> Array2<T>
    where
        T: LinalgScalar;

    fn scaled_add_ArrayView1(self, alpha: T, rhs: ArrayView1<T>, axis: Axis) -> Array2<T>
    where
        T: LinalgScalar;
    fn sub_column_view(self, rhs: ArrayView1<T>) -> Array2<T>
    where
        T: LinalgScalar + Neg<Output = T>;
    fn add_column_view(self, rhs: ArrayView1<T>) -> Array2<T>
    where
        T: LinalgScalar;

    // fn scale_Array1(self, rhs: &Array1<f_>, axis: Axis) -> Array2<f_>;
    // fn mul_column(self, rhs: &Array1<f_>) -> Array2<f_>;
    // fn mul_row(self, rhs: &Array1<f_>) -> Array2<f_>;

    fn product_trace(&self, rhs: &Array2<T>) -> T
    where
        T: LinalgScalar + Sum;
    fn fill_with_UPLO(self, uplo: UPLO) -> Self
    where
        T: Copy;
    fn map_UPLO<F>(self, uplo: UPLO, f: F) -> Self
    where
        F: FnMut((usize, usize)) -> T;
    // fn par_map_UPLO(self, uplo: UPLO, func: fn((usize, usize)) -> f_);
    // fn rem_subview_at_index(&mut self, indices: Vec<usize>, axis: Axis);
    fn rem_rows(&mut self, indices: Vec<usize>);
    fn rem_cols(&mut self, indices: Vec<usize>);
}

impl<T> Array2Utils<T> for Array2<T>
where
    T: Scalar, //+ LinalgScalar + std::ops::Neg<Output = T>,
{
    #[inline(always)]
    fn scaled_add_Array1(mut self, alpha: T, rhs: &Array1<T>, axis: Axis) -> Array2<T>
    where
        T: LinalgScalar,
    {
        self.axis_iter_mut(axis)
            .for_each(|mut a| a.scaled_add(alpha, rhs));
        self
    }

    #[inline(always)]
    fn sub_column(self, rhs: &Array1<T>) -> Array2<T>
    where
        T: LinalgScalar + Neg<Output = T>,
    {
        self.scaled_add_Array1(T::neg(T::one()), rhs, Axis(1))
    }

    #[inline(always)]
    fn add_column(self, rhs: &Array1<T>) -> Array2<T>
    where
        T: LinalgScalar,
    {
        self.scaled_add_Array1(T::one(), rhs, Axis(1))
    }

    #[inline(always)]
    fn scaled_add_ArrayView1(mut self, alpha: T, rhs: ArrayView1<T>, axis: Axis) -> Array2<T>
    where
        T: LinalgScalar,
    {
        self.axis_iter_mut(axis)
            .for_each(|mut a| a.scaled_add(alpha, &rhs));
        self
    }

    #[inline(always)]
    fn sub_column_view(self, rhs: ArrayView1<T>) -> Array2<T>
    where
        T: LinalgScalar + Neg<Output = T>,
    {
        self.scaled_add_ArrayView1(T::neg(T::one()), rhs, Axis(1))
    }

    #[inline(always)]
    fn add_column_view(self, rhs: ArrayView1<T>) -> Array2<T>
    where
        T: LinalgScalar,
    {
        self.scaled_add_ArrayView1(T::one(), rhs, Axis(1))
    }

    // #[inline(always)]
    // fn scale_Array1(mut self, rhs: &Array1<T>, axis: Axis) -> Array2<T> {
    //     self.axis_iter_mut(axis).for_each(|mut a| a *= rhs);
    //     self
    // }

    // #[inline(always)]
    // fn mul_column(self, rhs: &Array1<f_>) -> Array2<f_> {
    //     self.scale_Array1(rhs, Axis(1))
    // }

    // #[inline(always)]
    // fn mul_row(self, rhs: &Array1<f_>) -> Array2<f_> {
    //     self.scale_Array1(rhs, Axis(0))
    // }

    #[inline(always)]
    fn product_trace(&self, rhs: &Array2<T>) -> T
    where
        T: LinalgScalar + Sum,
    {
        self.columns()
            .into_iter()
            .zip(rhs.rows())
            .map(|(col, row)| col.dot(&row))
            .sum()
    }

    #[inline(always)]
    fn fill_with_UPLO(mut self, uplo: UPLO) -> Self
    where
        T: Copy,
    {
        if !self.is_square() {
            panic!("Matrix is not square!")
        };

        let dim = self.shape()[0];

        for i in 0..dim {
            for j in 0..i {
                match uplo {
                    UPLO::Upper => self[(i, j)] = self[(j, i)],
                    UPLO::Lower => self[(j, i)] = self[(i, j)],
                }
            }
        }

        self
    }

    #[inline(always)]
    fn map_UPLO<F>(mut self, uplo: UPLO, mut f: F) -> Self
    where
        F: FnMut((usize, usize)) -> T,
    {
        if !self.is_square() {
            panic!("Matrix is not square!")
        };

        let dim = self.shape()[0];

        for i in 0..dim {
            for j in 0..=i {
                match uplo {
                    UPLO::Upper => self[(j, i)] = f((j, i)),
                    UPLO::Lower => self[(i, j)] = f((i, j)),
                }
            }
        }

        self
    }

    // fn par_map_UPLO(mut self, uplo: UPLO, func: fn((usize, usize)) -> f_) {
    //     if !self.is_square() {
    //         panic!("Matrix is not square!")
    //     };

    //     let (i, _) = self.dim();

    //     (0..i).combinations_with_replacement(2)
    //         .into_iter()
    //         .par_bridge()
    //         .map(|v| {
    //             self[(i, 0)] = func((i, 0));
    //             match uplo {
    //                 UPLO::Upper => {let i = v[0]; let j = v[1]; self[(i, j)] = func((i, j))},
    //                 UPLO::Lower => {let i = v[1]; let j = v[0]; self[(i, j)] = func((i, j));},
    //             };

    //         });
    // }

    #[inline(always)]
    fn rem_rows(&mut self, indices: Vec<usize>) {
        self.rem_indices_axis(indices, Axis(0))
    }

    #[inline(always)]
    fn rem_cols(&mut self, indices: Vec<usize>) {
        self.rem_indices_axis(indices, Axis(1))
    }
}

pub trait ArrayView2Utils<T>
where
    T: Scalar,
{
    fn product_trace(&self, rhs: ArrayView2<T>) -> T
    where
        T: LinalgScalar + std::iter::Sum;

    fn as_1D(&self) -> Option<ArrayView1<'_, T>>;
}

impl<T> ArrayView2Utils<T> for ArrayView2<'_, T>
where
    T: Scalar,
{
    #[inline(always)]
    fn product_trace(&self, rhs: ArrayView2<T>) -> T
    where
        T: LinalgScalar + std::iter::Sum,
    {
        self.columns()
            .into_iter()
            .zip(rhs.rows())
            .map(|(col, row)| col.dot(&row))
            .sum()
    }

    fn as_1D(&self) -> Option<ArrayView1<'_, T>> {
        if self.ncols() > 1 && self.nrows() > 1 {
            None
        } else if self.nrows() == 1 {
            Some(self.row(0))
        } else {
            Some(self.column(0))
        }
    }
}

// pub trait Array3Utils {
//     fn outer(&self, index: usize) -> ArrayView2<'_, f_>;
// }

// impl Array3Utils for Array3<f_> {
//     fn outer(&self, index: usize) -> ArrayView2<'_, f_> {
//         self.slice(s![index, .., ..])
//     }
// }

// pub trait Array4Utils {
//     fn outer(&self, row: usize, col: usize) -> ArrayView2<'_, f_>;
// }

// impl Array4Utils for Array4<f_> {
//     fn outer(&self, row: usize, col: usize) -> ArrayView2<'_, f_> {
//         self.slice(s![row, col, .., ..])
//     }
// }
