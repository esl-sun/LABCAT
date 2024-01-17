use std::ops::IndexMut;

use faer_core::{col, row, ColRef, Mat, MatMut, MatRef, RowRef};
use ndarray::{Array, Ix2};

use crate::dtype;

pub trait IntoOwnedFaer {
    type Faer;
    #[track_caller]
    fn into_faer(self) -> Self::Faer;
}

impl<E> IntoOwnedFaer for Array<E, Ix2>
where
    E: dtype,
{
    type Faer = Mat<E>;

    #[track_caller]
    fn into_faer(self) -> Self::Faer {
        let nrows = self.nrows();
        let ncols = self.ncols();
        let strides: [isize; 2] = self.strides().try_into().unwrap();
        let ptr = { self }.as_ptr();
        unsafe {
            faer_core::mat::from_raw_parts::<'_, E>(ptr, nrows, ncols, strides[0], strides[1])
                .to_owned_mat()
        }
    }
}

pub trait MatRefUtils<E>
where
    E: dtype,
{
    fn to_owned_mat(self) -> Mat<E>;
    fn col_as_slice(&self, col: usize) -> &[E];
    fn row_as_slice(&self, row: usize) -> &[E];
    fn cols(&self) -> impl DoubleEndedIterator<Item = ColRef<'_, E>> + '_;
    fn rows(&self) -> impl DoubleEndedIterator<Item = RowRef<'_, E>> + '_;
}

impl<'a, E: dtype> MatRefUtils<E> for MatRef<'a, E> {
    fn to_owned_mat(self) -> Mat<E> {
        let mut mat = Mat::new();
        mat.copy_from(self);
        mat
    }

    /// Returns a reference to a slice over the column at the given index.
    #[inline]
    #[track_caller]
    fn col_as_slice(&self, col: usize) -> &[E] {
        assert!(col < self.ncols());
        let nrows = self.nrows();
        let ptr = self.as_ref().ptr_at(0, col);
        unsafe { core::slice::from_raw_parts(ptr, nrows) }
    }

    /// Returns a reference to a slice over the row at the given index.
    #[inline]
    #[track_caller]
    fn row_as_slice(&self, row: usize) -> &[E] {
        assert!(row < self.nrows());
        let nrows = self.nrows();
        let ptr = self.as_ref().transpose().ptr_at(0, row);
        unsafe { core::slice::from_raw_parts(ptr, nrows) }
    }

    #[inline]
    #[track_caller]
    fn cols(&self) -> impl DoubleEndedIterator<Item = ColRef<'_, E>> + '_ {
        let row_stride = self.row_stride();

        (0..self.ncols()).map(move |col_id| unsafe {
            col::from_raw_parts(
                self.as_ref().ptr_inbounds_at(0, col_id),
                self.nrows(),
                row_stride,
            )
        })
    }

    fn rows(&self) -> impl DoubleEndedIterator<Item = RowRef<'_, E>> + '_ {
        let col_stride = self.col_stride();

        (0..self.nrows()).map(move |row_id| unsafe {
            row::from_raw_parts(
                self.as_ref().ptr_inbounds_at(row_id, 0),
                self.ncols(),
                col_stride,
            )
        })
    }
}

pub trait MatMutUtils<E>
where
    E: dtype,
{
    fn zip_apply_with_col_slice(&mut self, s: &[E], f: fn(E, E) -> E);
    fn zip_apply_with_row_slice(&mut self, s: &[E], f: fn(E, E) -> E);
    fn fill_fn(&mut self, f: fn(usize, usize) -> E);
    fn apply_fn(&mut self, f: fn((usize, usize), E) -> E);
}

impl<'a, E: dtype> MatMutUtils<E> for MatMut<'a, E> {
    fn zip_apply_with_col_slice(&mut self, s: &[E], f: fn(E, E) -> E) {
        #[cfg(debug_assertions)]
        if self.ncols() != s.len() {
            panic!(
                "Number of rows in self ({}) and column slice length ({}) do not match!",
                self.nrows(),
                s.len()
            );
        }

        let nrows = self.nrows();
        let ncols = self.ncols();

        (0..nrows)
            .flat_map(move |i| (0..ncols).map(move |j| (i, j)))
            .for_each(|(i, j)| *self.index_mut((i, j)) = f(*self.index_mut((i, j)), s[i]));
    }

    fn zip_apply_with_row_slice(&mut self, s: &[E], f: fn(E, E) -> E) {
        #[cfg(debug_assertions)]
        if self.ncols() != s.len() {
            panic!(
                "Number of cols in self ({}) and row slice length ({}) do not match!",
                self.ncols(),
                s.len()
            );
        }

        let nrows = self.nrows();
        let ncols = self.ncols();

        (0..nrows)
            .flat_map(move |i| (0..ncols).map(move |j| (i, j)))
            .for_each(|(i, j)| *self.index_mut((i, j)) = f(*self.index_mut((i, j)), s[j]));
    }

    fn fill_fn(&mut self, f: fn(usize, usize) -> E) {
        let nrows = self.nrows();
        let ncols = self.ncols();

        (0..nrows)
            .flat_map(move |i| (0..ncols).map(move |j| (i, j)))
            .for_each(|(i, j)| self.write(i, j, f(i, j)));
    }

    fn apply_fn(&mut self, f: fn((usize, usize), E) -> E) {
        let nrows = self.nrows();
        let ncols = self.ncols();

        (0..nrows)
            .flat_map(move |i| (0..ncols).map(move |j| (i, j)))
            .for_each(|(i, j)| *self.index_mut((i, j)) = f((i, j), *self.index_mut((i, j))));
    }
}

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
    fn remove_cols(&self, idx: Vec<usize>) -> Mat<E>;
    // fn remove_rows(&self, idx: Vec<usize>) -> Mat<E>;
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

    fn remove_cols(&self, mut idx: Vec<usize>) -> Mat<E> {
        idx.sort(); //sort vec
        idx.dedup(); //remove duplicates

        #[cfg(debug_assertions)]
        if let Some(max_id) = idx.last() {
            if *max_id >= self.ncols() {
                panic!(
                    "Col index to remove ({}) out of bounds for matrix with {} cols!",
                    max_id,
                    self.ncols()
                );
            }
        }

        let nrows = self.nrows();
        let mut mat_red = Mat::<E>::zeros(nrows, usize::saturating_sub(self.ncols(), idx.len()));

        self.as_ref()
            .cols()
            .enumerate()
            .filter(|(col_id, _)| idx.binary_search(col_id).is_err())
            .enumerate()
            .for_each(|(new_col_id, (_, col))| unsafe {
                let s = core::slice::from_raw_parts(col.as_ptr(), nrows);
                mat_red.fill_col_with_slice(new_col_id, s);
            });

        mat_red
    }

    //DOES NOT WORK, possibly due to not being able to take row slice of column-major matrix, Maybe use slower implementation with manual read/write
    // fn remove_rows(&self, mut idx: Vec<usize>) -> Mat<E> {

    //     idx.sort(); //sort vec
    //     idx.dedup(); //remove duplicates

    //     #[cfg(debug_assertions)]
    //     if let Some(max_id) = idx.last() {
    //         if *max_id >= self.nrows() {
    //             panic!(
    //                 "Row index to remove ({}) out of bounds for matrix with {} rows!",
    //                 max_id,
    //                 self.nrows()
    //             );
    //         }
    //     }

    //     let ncols = self.ncols();
    //     let mut mat_red = Mat::<E>::zeros(usize::saturating_sub(self.nrows(), idx.len()), ncols);

    //     self.as_ref()
    //         .transpose()
    //         .cols()
    //         .enumerate()
    //         .filter(|(row_id, row)| !idx.binary_search(row_id).is_ok())
    //         .enumerate()
    //         .for_each(|(new_row_id, (row_id, row))| unsafe {
    //             dbg!(&row);
    //             let s = core::slice::from_raw_parts(row.as_ptr(), ncols); // FIX
    //             dbg!(s);
    //             mat_red.fill_row_with_slice(new_row_id, s);
    //         });

    //     mat_red
    // }

    // fn remove_rows(&self, idx: Vec<usize>) -> Mat<E> {
    //     todo!()
    // }
}
