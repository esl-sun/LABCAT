use std::ops::IndexMut;

use faer_core::{Mat, MatMut, MatRef};
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
}

pub trait MatMutUtils<E>
where
    E: dtype,
{
    // fn zip_apply_with_col_slice(&mut self, s: &[E], f: fn(E, E) -> E)
    fn zip_apply_with_row_slice(&mut self, s: &[E], f: fn(E, E) -> E);
    fn fill_fn(&mut self, f: fn(usize, usize) -> E);
    fn apply_fn(&mut self, f: fn((usize, usize), E) -> E);
}

impl<'a, E: dtype> MatMutUtils<E> for MatMut<'a, E> {
    // fn zip_apply_with_col_slice(&mut self, s: &[E], f: fn(E, E) -> E)
    // {
    //     #[cfg(debug_assertions)]
    //     if self.nrows() != s.len() {
    //         panic!(
    //             "Number of rows in self ({}) and column slice length ({}) do not match!",
    //             self.nrows(),
    //             s.len()
    //         );
    //     }

    //     let nrows = self.nrows();
    //     let ncols = self.ncols();

    //     (0..nrows)
    //         .flat_map(move |i| (0..ncols).map(move |j| (i, j)))
    //         .for_each(|(i, j)| self.write(i, j, f(*self.get_mut(i, j), s[i])));
    // }

    fn zip_apply_with_row_slice(&mut self, s: &[E], f: fn(E, E) -> E) {
        #[cfg(debug_assertions)]
        if self.ncols() != s.len() {
            panic!(
                "Number of cols in self ({}) and column slice length ({}) do not match!",
                self.ncols(),
                s.len()
            );
        }

        let nrows = self.nrows();
        let ncols = self.ncols();

        (0..nrows)
            .flat_map(move |i| (0..ncols).map(move |j| (i, j)))
            // .for_each(|(i, j)| self.write(i, j, f(*self.get_mut(i, j), s[j])));
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

        // for i in 0..nrows {
        //     for j in 0..ncols {
        //         *self.index_mut((i, j)) = f((i, j), *self.index_mut((i, j)))
        //     }
        // }

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
    // fn fill_iter<'a>(&mut self, iter: impl Iterator<Item = ((usize, usize), &'a E)>);
    fn fill_fn(&mut self, f: fn(usize, usize) -> E);
    fn apply_fn<F>(&mut self, f: F)
    where
        F: FnMut((usize, usize), E) -> E;
    fn fill_row_with_slice(&mut self, row: usize, s: &[E]);
    fn fill_col_with_slice(&mut self, col: usize, s: &[E]);
    // fn remove_cols_rows(&self, col_idx: Vec<usize>, row_idx: Vec<usize>) -> Mat<T>;
    // fn remove_cols(self, idx: Vec<usize>) -> Mat<E>;
    // fn remove_rows(&self, idx: Vec<usize>) -> Mat<T>;
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

    // fn remove_cols_rows(&self, col_idx: Vec<usize>, row_idx: Vec<usize>) -> Mat<E> {
    //     todo!()
    // }

    // fn remove_cols(self, idx: Vec<usize>) {

    //     self

    //     // idx.dedup(); //remove duplicates

    //     // let mut mat_red = Mat::<E>::new();
    //     // self.col_chunks(1)
    //     //     .enumerate()
    //     //     .filter(|&(i, _)| !idx.contains(&i))
    //     //     .map(|(i, col)| col)
    //     //     .for_each(|col|)
    //     // mat_red

    //     todo!()
    // }

    // fn remove_rows(&self, idx: Vec<usize>) -> Mat<E> {
    //     todo!()
    // }
}
