use std::ops::IndexMut;

use faer::modules::core::{AsColMut, AsColRef, AsMatMut, AsMatRef, AsRowMut, AsRowRef};
use faer::{col, row, unzipped, zipped, Col, ColMut, ColRef, Mat, Row, RowMut, RowRef, Side};
// use ndarray::{Array, Ix2};

use crate::dtype;

#[derive(Debug, Clone)]

pub enum Select {
    Include,
    Exclude,
}

#[derive(Debug, Clone)]

pub enum Axis {
    Col,
    Row,
}

// pub trait IntoOwnedFaer {
//     type Faer;
//     #[track_caller]
//     fn into_faer(self) -> Self::Faer;
// }

// impl<E> IntoOwnedFaer for Array<E, Ix2>
// where
//     E: dtype,
// {
//     type Faer = Mat<E>;

//     #[track_caller]
//     fn into_faer(self) -> Self::Faer {
//         let nrows = self.nrows();
//         let ncols = self.ncols();
//         let strides: [isize; 2] = self.strides().try_into().unwrap();
//         let ptr = { self }.as_ptr();
//         unsafe {
//             faer_core::mat::from_raw_parts::<'_, E>(ptr, nrows, ncols, strides[0], strides[1])
//                 .to_owned_mat()
//         }
//     }
// }

pub trait ColRefUtils<E>
where
    Self: AsColRef<E>,
    E: dtype,
{
    #[inline]
    #[track_caller]
    fn as_slice(&self) -> &[E] {
        let nrows = self.as_col_ref().nrows();
        let ptr = self.as_col_ref().as_ptr();
        E::faer_map(
            ptr,
            #[inline(always)]
            |ptr| unsafe { core::slice::from_raw_parts(ptr, nrows) },
        )
    }

    #[inline]
    #[track_caller]
    fn get_subcol_with_idx(&self, select: Select, mut idx: Vec<usize>) -> Col<E> {
        idx.sort(); //sort vec
        idx.dedup(); //remove duplicates

        #[cfg(debug_assertions)]
        if let Some(max_id) = idx.last() {
            if *max_id >= self.as_col_ref().nrows() {
                panic!("Row index ({}) to {:?} out of bounds!", max_id, select,);
            }
        }

        let nrows = match &select {
            Select::Include => idx.len(),
            Select::Exclude => usize::saturating_sub(self.as_col_ref().nrows(), idx.len()),
        };

        //TODO: Sidestep allocation?
        let mut col_red = Col::<E>::zeros(nrows);

        self.as_col_ref()
            .as_slice()
            .iter()
            .enumerate()
            .filter(|(row_id, _)| match select {
                Select::Include => idx.binary_search(row_id).is_ok(),
                Select::Exclude => idx.binary_search(row_id).is_err(),
            })
            .enumerate()
            .for_each(|(new_row_id, (_, row_val))| {
                col_red[new_row_id] = *row_val;
            });

        col_red
    }
}

impl<E: dtype, M: AsColRef<E>> ColRefUtils<E> for M {}

pub trait ColMutUtils<E>
where
    Self: AsColMut<E>,
    E: dtype,
{
    #[inline]
    #[track_caller]
    fn as_mut_slice(&mut self) -> &mut [E] {
        let nrows = self.as_col_mut().nrows();
        let ptr = self.as_col_mut().as_ptr_mut();
        E::faer_map(
            ptr,
            #[inline(always)]
            |ptr| unsafe { core::slice::from_raw_parts_mut(ptr, nrows) },
        )
    }
}

impl<E: dtype, M: AsColMut<E>> ColMutUtils<E> for M {}

pub trait RowRefUtils<E>
where
    Self: AsRowRef<E>,
    E: dtype,
{
    #[inline]
    #[track_caller]
    fn as_slice(&self) -> &[E] {
        let ncols = self.as_row_ref().ncols();
        let ptr = self.as_row_ref().as_ptr();
        E::faer_map(
            ptr,
            #[inline(always)]
            |ptr| unsafe { core::slice::from_raw_parts(ptr, ncols) },
        )
    }

    #[inline]
    #[track_caller]
    fn get_subrow_with_idx(&self, select: Select, mut idx: Vec<usize>) -> Row<E> {
        idx.sort(); //sort vec
        idx.dedup(); //remove duplicates

        #[cfg(debug_assertions)]
        if let Some(max_id) = idx.last() {
            if *max_id >= self.as_row_ref().nrows() {
                panic!("Row index ({}) to {:?} out of bounds!", max_id, select,);
            }
        }

        let ncols = match &select {
            Select::Include => idx.len(),
            Select::Exclude => usize::saturating_sub(self.as_row_ref().ncols(), idx.len()),
        };

        //TODO: Sidestep allocation?
        let mut row_red = Row::<E>::zeros(ncols);

        self.as_row_ref()
            .as_slice()
            .iter()
            .enumerate()
            .filter(|(col_id, _)| match select {
                Select::Include => idx.binary_search(col_id).is_ok(),
                Select::Exclude => idx.binary_search(col_id).is_err(),
            })
            .enumerate()
            .for_each(|(new_col_id, (_, col_val))| {
                row_red[new_col_id] = *col_val;
            });

        row_red
    }
}

impl<E: dtype, M: AsRowRef<E>> RowRefUtils<E> for M {}

pub trait RowMutUtils<E>
where
    Self: AsRowMut<E>,
    E: dtype,
{
    #[inline]
    #[track_caller]
    fn as_mut_slice(&mut self) -> &mut [E] {
        let ncols = self.as_row_mut().ncols();
        let ptr = self.as_row_mut().as_ptr_mut();
        E::faer_map(
            ptr,
            #[inline(always)]
            |ptr| unsafe { core::slice::from_raw_parts_mut(ptr, ncols) },
        )
    }
}

impl<E: dtype, M: AsRowMut<E>> RowMutUtils<E> for M {}

pub trait MatRefUtils<E>
where
    Self: AsMatRef<E>,
    E: dtype,
{
    fn to_owned_mat(&self) -> Mat<E> {
        let mut mat = Mat::new();
        mat.copy_from(self.as_mat_ref());
        mat
    }

    /// Returns a reference to a slice over the column at the given index.
    #[inline]
    #[track_caller]
    fn col_as_slice(&self, col: usize) -> &[E] {
        assert!(col < self.as_mat_ref().ncols());
        let nrows = self.as_mat_ref().nrows();
        let ptr = self.as_mat_ref().ptr_at(0, col);
        unsafe { core::slice::from_raw_parts(ptr, nrows) }
    }

    #[inline]
    #[track_caller]
    fn cols(&self) -> impl DoubleEndedIterator<Item = ColRef<'_, E>> + '_ {
        let row_stride = self.as_mat_ref().row_stride();

        (0..self.as_mat_ref().ncols()).map(move |col_id| unsafe {
            col::from_raw_parts(
                self.as_mat_ref().ptr_inbounds_at(0, col_id),
                self.as_mat_ref().nrows(),
                row_stride,
            )
        })
    }

    #[inline]
    #[track_caller]
    fn rows(&self) -> impl DoubleEndedIterator<Item = RowRef<'_, E>> + '_ {
        let col_stride = self.as_mat_ref().col_stride();

        (0..self.as_mat_ref().nrows()).map(move |row_id| unsafe {
            row::from_raw_parts(
                self.as_mat_ref().ptr_inbounds_at(row_id, 0),
                self.as_mat_ref().ncols(),
                col_stride,
            )
        })
    }

    fn get_submatrix_with_idx(&self, select: Select, axis: Axis, mut idx: Vec<usize>) -> Mat<E> {
        idx.sort(); //sort vec
        idx.dedup(); //remove duplicates

        #[cfg(debug_assertions)]
        if let Some(max_id) = idx.last() {
            if *max_id
                >= match &axis {
                    Axis::Col => self.as_mat_ref().ncols(),
                    Axis::Row => self.as_mat_ref().nrows(),
                }
            {
                panic!(
                    "{:?} index ({}) to {:?} out of bounds!",
                    axis, max_id, select,
                );
            }
        }

        let (nrows, ncols) = match (&axis, &select) {
            (Axis::Col, Select::Include) => (self.as_mat_ref().nrows(), idx.len()),
            (Axis::Col, Select::Exclude) => (
                self.as_mat_ref().nrows(),
                usize::saturating_sub(self.as_mat_ref().ncols(), idx.len()),
            ),
            (Axis::Row, Select::Include) => (idx.len(), self.as_mat_ref().ncols()),
            (Axis::Row, Select::Exclude) => (
                usize::saturating_sub(self.as_mat_ref().nrows(), idx.len()),
                self.as_mat_ref().ncols(),
            ),
        };

        //TODO: Sidestep allocation?
        let mut mat_red = Mat::<E>::zeros(nrows, ncols);

        match axis {
            Axis::Col => self.as_mat_ref(),
            Axis::Row => self.as_mat_ref().transpose(),
        }
        .cols()
        .enumerate()
        .filter(|(col_id, _)| match select {
            Select::Include => idx.binary_search(col_id).is_ok(),
            Select::Exclude => idx.binary_search(col_id).is_err(),
        })
        .enumerate()
        .for_each(|(new_col_id, (_, col))| {
            zipped!(
                match axis {
                    Axis::Col => mat_red.as_mut(),
                    Axis::Row => mat_red.as_mut().transpose_mut(),
                }
                .col_mut(new_col_id),
                col
            )
            .for_each(|unzipped!(mut old, new)| old.write(new.read()));
        });

        mat_red
    }
}

impl<E: dtype, M: AsMatRef<E>> MatRefUtils<E> for M {}

pub trait MatMutUtils<E>
where
    Self: AsMatMut<E>,
    E: dtype,
{
    //TODO: change zip_apply_* to zipped! from faer crate
    #[inline]
    #[track_caller]
    fn zip_apply_with_col_slice(&mut self, s: &[E], f: fn(E, E) -> E) {
        #[cfg(debug_assertions)]
        if self.as_mat_mut().ncols() != s.len() {
            panic!(
                "Number of rows in self ({}) and column slice length ({}) do not match!",
                self.as_mat_mut().nrows(),
                s.len()
            );
        }

        let nrows = self.as_mat_mut().nrows();
        let ncols = self.as_mat_mut().ncols();

        (0..nrows)
            .flat_map(move |i| (0..ncols).map(move |j| (i, j)))
            .for_each(|(i, j)| {
                *self.as_mat_mut().index_mut((i, j)) = f(*self.as_mat_mut().index_mut((i, j)), s[i])
            });
    }

    #[inline]
    #[track_caller]
    fn zip_apply_with_row_slice(&mut self, s: &[E], f: fn(E, E) -> E) {
        #[cfg(debug_assertions)]
        if self.as_mat_mut().ncols() != s.len() {
            panic!(
                "Number of cols in self ({}) and row slice length ({}) do not match!",
                self.as_mat_mut().ncols(),
                s.len()
            );
        }

        let nrows = self.as_mat_mut().nrows();
        let ncols = self.as_mat_mut().ncols();

        (0..nrows)
            .flat_map(move |i| (0..ncols).map(move |j| (i, j)))
            .for_each(|(i, j)| {
                *self.as_mat_mut().index_mut((i, j)) = f(*self.as_mat_mut().index_mut((i, j)), s[j])
            });
    }

    #[inline]
    #[track_caller]
    fn fill_fn(&mut self, f: fn(usize, usize) -> E) {
        let nrows = self.as_mat_mut().nrows();
        let ncols = self.as_mat_mut().ncols();

        (0..nrows)
            .flat_map(move |i| (0..ncols).map(move |j| (i, j)))
            .for_each(|(i, j)| self.as_mat_mut().write(i, j, f(i, j)));
    }

    #[inline]
    #[track_caller]
    fn apply_fn(&mut self, f: fn((usize, usize), E) -> E) {
        let nrows = self.as_mat_mut().nrows();
        let ncols = self.as_mat_mut().ncols();

        (0..nrows)
            .flat_map(move |i| (0..ncols).map(move |j| (i, j)))
            .for_each(|(i, j)| {
                *self.as_mat_mut().index_mut((i, j)) =
                    f((i, j), *self.as_mat_mut().index_mut((i, j)))
            });
    }

    #[inline]
    #[track_caller]
    fn cols_mut(&mut self) -> impl DoubleEndedIterator<Item = ColMut<'_, E>> + '_ {
        let row_stride = self.as_mat_mut().row_stride();

        (0..self.as_mat_mut().ncols()).map(move |col_id| unsafe {
            col::from_raw_parts_mut(
                self.as_mat_mut().ptr_inbounds_at_mut(0, col_id),
                self.as_mat_mut().nrows(),
                row_stride,
            )
        })
    }

    #[inline]
    #[track_caller]
    fn rows_mut(&mut self) -> impl DoubleEndedIterator<Item = RowMut<'_, E>> + '_ {
        let col_stride = self.as_mat_mut().col_stride();

        (0..self.as_mat_mut().nrows()).map(move |row_id| unsafe {
            row::from_raw_parts_mut(
                self.as_mat_mut().ptr_inbounds_at_mut(row_id, 0),
                self.as_mat_mut().ncols(),
                col_stride,
            )
        })
    }

    #[inline]
    #[track_caller]
    fn fill_with_side(&mut self, side: Side) {
        if self.as_mat_mut().nrows() != self.as_mat_mut().ncols() {
            panic!("Matrix is not square!")
        };

        let dim = self.as_mat_mut().nrows();

        for i in 0..dim {
            for j in 0..i {
                match side {
                    Side::Upper => self.as_mat_mut()[(i, j)] = self.as_mat_mut()[(j, i)],
                    Side::Lower => self.as_mat_mut()[(j, i)] = self.as_mat_mut()[(i, j)],
                }
            }
        }
    }
}

impl<E: dtype, M: AsMatMut<E>> MatMutUtils<E> for M {}

pub trait MatUtils<E>
where
    E: dtype,
{
    fn indexed_iter(&self) -> impl Iterator<Item = ((usize, usize), &E)>;
    fn fill_fn(&mut self, f: fn(usize, usize) -> E);
    fn apply_fn<F>(&mut self, f: F)
    where
        F: FnMut((usize, usize), E) -> E;
    fn fill_row_with_slice(&mut self, row: usize, s: &[E]);
    fn fill_col_with_slice(&mut self, col: usize, s: &[E]);
}

impl<E: dtype> MatUtils<E> for Mat<E> {
    fn indexed_iter(&self) -> impl Iterator<Item = ((usize, usize), &E)> {
        (0..self.nrows())
            .flat_map(|i| (0..self.ncols()).map(move |j| (i, j)))
            .map(|(i, j)| ((i, j), self.get(i, j)))
    }

    fn fill_fn(&mut self, f: fn(usize, usize) -> E) {
        let nrows = self.nrows();
        let ncols = self.ncols();

        (0..nrows)
            .flat_map(move |i| (0..ncols).map(move |j| (i, j)))
            .for_each(|(i, j)| self.write(i, j, f(i, j)));
    }

    fn apply_fn<F>(&mut self, mut f: F)
    where
        F: FnMut((usize, usize), E) -> E,
    {
        let nrows = self.nrows();
        let ncols = self.ncols();

        (0..nrows)
            .flat_map(move |i| (0..ncols).map(move |j| (i, j)))
            .for_each(|(i, j)| self.write(i, j, f((i, j), *self.get(i, j))));
    }

    fn fill_row_with_slice(&mut self, row: usize, s: &[E]) {
        #[cfg(debug_assertions)]
        if row >= self.nrows() {
            panic!(
                "Row index ({}) out of bounds for matrix with {} rows!",
                row,
                self.nrows()
            );
        }

        #[cfg(debug_assertions)]
        if s.len() != self.ncols() {
            panic!(
                "Length of slice ({}) does not match number of columns in matrix ({})!",
                s.len(),
                self.ncols()
            );
        }

        (0..self.ncols()).for_each(|j| self.write(row, j, s[j]))
    }

    fn fill_col_with_slice(&mut self, col: usize, s: &[E]) {
        #[cfg(debug_assertions)]
        if col >= self.ncols() {
            panic!(
                "Col index ({}) out of bounds for matrix with {} cols!",
                col,
                self.ncols()
            );
        }

        #[cfg(debug_assertions)]
        if s.len() != self.nrows() {
            panic!(
                "Length of slice ({}) does not match number of rows in matrix ({})!",
                s.len(),
                self.nrows()
            );
        }

        (0..self.nrows()).for_each(|i| self.write(i, col, s[i]))
    }
}
