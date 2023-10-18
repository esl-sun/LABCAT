use ndarray::{
    s, Array1, Array2, Array3, Array4, ArrayBase, ArrayView1, ArrayView2, Axis, DataMut, Dimension,
    NewAxis,
};
use ndarray_linalg::{Scalar, UPLO};

use crate::f_;

pub trait ArrayBaseUtils<A, D>
where
    D: Dimension,
{
    fn apply(self, func: fn(A) -> A) -> Self;
    fn apply_mut(&mut self, func: fn(A) -> A);
    fn ln(self) -> Self;
    fn exp(self) -> Self;
}

/// # Methods For All Array Types
impl<A, S, D> ArrayBaseUtils<A, D> for ArrayBase<S, D>
where
    A: Scalar + Send + Sync,
    S: DataMut<Elem = A>,
    D: Dimension,
{
    fn apply(mut self, func: fn(A) -> A) -> Self {
        self.par_mapv_inplace(func);
        self
    }

    fn apply_mut(&mut self, func: fn(A) -> A) {
        self.par_mapv_inplace(func);
    }

    fn ln(self) -> Self {
        self.apply(|A| A.ln())
    }

    fn exp(self) -> Self {
        self.apply(|A| A.exp())
    }
}

pub trait ArrayBaseFloatUtils<D>
where
    D: Dimension,
{
    fn max(&self) -> Option<&f_>;
    fn min(&self) -> Option<&f_>;
    fn indexed_max(&self) -> Option<(D::Pattern, &f_)>;
    fn indexed_min(&self) -> Option<(D::Pattern, &f_)>;
}

impl<S, D> ArrayBaseFloatUtils<D> for ArrayBase<S, D>
where
    S: DataMut<Elem = f_>,
    D: Dimension,
{
    #[inline(always)]
    fn max(&self) -> Option<&f_> {
        let max = self.into_iter().max_by(|x, y| x.total_cmp(y));

        if let Some(max) = max {
            if max.is_infinite() || max.is_nan() {
                return None;
            }
        }
        max
    }

    #[inline(always)]
    fn min(&self) -> Option<&f_> {
        let min = self.into_iter().min_by(|x, y| x.total_cmp(y));

        if let Some(min) = min {
            if min.is_infinite() || min.is_nan() {
                return None;
            }
        }
        min
    }

    #[inline(always)]
    fn indexed_max(&self) -> Option<(D::Pattern, &f_)> {
        let max = self.indexed_iter().max_by(|(_, x), (_, y)| x.total_cmp(y));

        if let Some((_, max)) = max {
            if max.is_infinite() || max.is_nan() {
                return None;
            }
        }
        max
    }

    #[inline(always)]
    fn indexed_min(&self) -> Option<(D::Pattern, &f_)> {
        let min = self.indexed_iter().min_by(|(_, x), (_, y)| x.total_cmp(y));

        if let Some((_, min)) = min {
            if min.is_infinite() || min.is_nan() {
                return None;
            }
        }
        min
    }
}

pub trait Array1Utils {
    fn into_col(self) -> Array2<f_>;
    fn rem_at_index(self, indices: Vec<usize>) -> Array1<f_>;
}

impl Array1Utils for Array1<f_> {
    #[inline(always)]
    fn into_col(self) -> Array2<f_> {
        let len = self.len();
        self.into_shape((len, 1)).expect("Should never fail")
    }

    #[inline(always)]
    fn rem_at_index(mut self, mut indices: Vec<usize>) -> Array1<f_> {
        indices.sort(); //order so that method works
        indices.dedup(); //remove duplicates

        indices.iter().enumerate().for_each(|(offset, rem_index)| {
            self.remove_index(Axis(0), rem_index.saturating_sub(offset))
        });

        self
    }
}

pub trait ArrayView1Utils {
    fn into_col(&self) -> ArrayView2<f_>;
}

impl ArrayView1Utils for ArrayView1<'_, f_> {
    #[inline(always)]
    fn into_col(&self) -> ArrayView2<f_> {
        self.slice(s![.., NewAxis])
    }
}

pub trait Array2Utils {
    fn scaled_add_Array1(self, alpha: f_, rhs: &Array1<f_>, axis: Axis) -> Array2<f_>;
    fn sub_column(self, rhs: &Array1<f_>) -> Array2<f_>;
    fn add_column(self, rhs: &Array1<f_>) -> Array2<f_>;

    fn scaled_add_ArrayView1(self, alpha: f_, rhs: &ArrayView1<f_>, axis: Axis) -> Array2<f_>;
    fn sub_column_view(self, rhs: &ArrayView1<f_>) -> Array2<f_>;
    fn add_column_view(self, rhs: &ArrayView1<f_>) -> Array2<f_>;

    fn scale_Array1(self, rhs: &Array1<f_>, axis: Axis) -> Array2<f_>;
    fn mul_column(self, rhs: &Array1<f_>) -> Array2<f_>;
    fn mul_row(self, rhs: &Array1<f_>) -> Array2<f_>;

    fn product_trace(&self, rhs: &Array2<f_>) -> f_;
    fn fill_with_UPLO(self, uplo: UPLO) -> Self;
    // fn map_UPLO(self, uplo: UPLO, f: fn((usize, usize)) -> f_) -> Self;
    fn map_UPLO<F>(self, uplo: UPLO, f: F) -> Self
    where
        F: FnMut((usize, usize)) -> f_;
    // fn par_map_UPLO(self, uplo: UPLO, func: fn((usize, usize)) -> f_);

    fn rem_subview_at_index(self, indices: Vec<usize>, axis: Axis) -> Array2<f_>;
    fn rem_rows(self, indices: Vec<usize>) -> Array2<f_>;
    fn rem_cols(self, indices: Vec<usize>) -> Array2<f_>;
}

impl Array2Utils for Array2<f_> {
    #[inline(always)]
    fn scaled_add_Array1(mut self, alpha: f_, rhs: &Array1<f_>, axis: Axis) -> Array2<f_> {
        self.axis_iter_mut(axis)
            .for_each(|mut a| a.scaled_add(alpha, rhs));
        self
    }

    #[inline(always)]
    fn sub_column(self, rhs: &Array1<f_>) -> Array2<f_> {
        self.scaled_add_Array1(-1.0, rhs, Axis(1))
    }

    #[inline(always)]
    fn add_column(self, rhs: &Array1<f_>) -> Array2<f_> {
        self.scaled_add_Array1(1.0, rhs, Axis(1))
    }

    #[inline(always)]
    fn scaled_add_ArrayView1(mut self, alpha: f_, rhs: &ArrayView1<f_>, axis: Axis) -> Array2<f_> {
        self.axis_iter_mut(axis)
            .for_each(|mut a| a.scaled_add(alpha, rhs));
        self
    }

    #[inline(always)]
    fn sub_column_view(self, rhs: &ArrayView1<f_>) -> Array2<f_> {
        self.scaled_add_ArrayView1(-1.0, rhs, Axis(1))
    }

    #[inline(always)]
    fn add_column_view(self, rhs: &ArrayView1<f_>) -> Array2<f_> {
        self.scaled_add_ArrayView1(1.0, rhs, Axis(1))
    }

    #[inline(always)]
    fn scale_Array1(mut self, rhs: &Array1<f_>, axis: Axis) -> Array2<f_> {
        self.axis_iter_mut(axis).for_each(|mut a| a *= rhs);
        self
    }

    #[inline(always)]
    fn mul_column(self, rhs: &Array1<f_>) -> Array2<f_> {
        self.scale_Array1(rhs, Axis(1))
    }

    #[inline(always)]
    fn mul_row(self, rhs: &Array1<f_>) -> Array2<f_> {
        self.scale_Array1(rhs, Axis(0))
    }

    #[inline(always)]
    fn product_trace(&self, rhs: &Array2<f_>) -> f_ {
        self.columns()
            .into_iter()
            .zip(rhs.rows().into_iter())
            .map(|(col, row)| col.dot(&row))
            .sum()
    }

    #[inline(always)]
    fn fill_with_UPLO(mut self, uplo: UPLO) -> Self {
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

    fn map_UPLO<F>(mut self, uplo: UPLO, mut f: F) -> Self
    where
        F: FnMut((usize, usize)) -> f_,
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
    fn rem_subview_at_index(self, mut indices: Vec<usize>, axis: Axis) -> Array2<f_> {
        indices.sort(); //order so that method works
        indices.dedup(); //remove duplicates

        match axis {
            Axis(i) if i == 0 => indices.retain(|i| *i < self.nrows()),
            Axis(i) if i == 1 => indices.retain(|i| *i < self.ncols()),
            _ => panic!("Invalid Axis index!"),
        };

        let sh = match axis {
            Axis(i) if i == 0 => (self.nrows() - indices.len(), self.ncols()),
            Axis(i) if i == 1 => (self.nrows(), self.ncols() - indices.len()),
            _ => panic!("Invalid Axis index!"),
        };

        let red_vec: Vec<f_> = match axis {
            Axis(i) if i == 0 => self
                .indexed_iter()
                .filter(|((i, _), _)| !indices.contains(i))
                .map(|((_, _), val)| *val)
                .collect(),
            Axis(i) if i == 1 => self
                .indexed_iter()
                .filter(|((_, j), _)| !indices.contains(j))
                .map(|((_, _), val)| *val)
                .collect(),
            _ => panic!("Invalid Axis index!"),
        };

        Array2::from_shape_vec(sh, red_vec).expect("Should never fail.")
    }

    #[inline(always)]
    fn rem_rows(self, indices: Vec<usize>) -> Array2<f_> {
        self.rem_subview_at_index(indices, Axis(0))
    }

    #[inline(always)]
    fn rem_cols(self, indices: Vec<usize>) -> Array2<f_> {
        self.rem_subview_at_index(indices, Axis(1))
    }
}

pub trait ArrayView2Utils {
    fn product_trace(&self, rhs: &ArrayView2<f_>) -> f_;
}

impl ArrayView2Utils for ArrayView2<'_, f_> {
    fn product_trace(&self, rhs: &ArrayView2<f_>) -> f_ {
        self.columns()
            .into_iter()
            .zip(rhs.rows().into_iter())
            .map(|(col, row)| col.dot(&row))
            .sum()
    }
}

pub trait Array3Utils {
    fn outer(&self, index: usize) -> ArrayView2<'_, f_>;
}

impl Array3Utils for Array3<f_> {
    fn outer(&self, index: usize) -> ArrayView2<'_, f_> {
        self.slice(s![index, .., ..])
    }
}

pub trait Array4Utils {
    fn outer(&self, row: usize, col: usize) -> ArrayView2<'_, f_>;
}

impl Array4Utils for Array4<f_> {
    fn outer(&self, row: usize, col: usize) -> ArrayView2<'_, f_> {
        self.slice(s![row, col, .., ..])
    }
}
