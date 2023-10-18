use anyhow::Result;
use ndarray::{array, Array1, Array2, ArrayView1, Axis};
use ndarray_linalg::{Determinant, Eig, Norm, SVD};
use rand;

use crate::{
    bounds_array::ArrayBounds,
    f_,
    utils::{Array1Utils, Array2Utils, ArrayBaseFloatUtils},
};

#[derive(Debug, Clone)]
pub enum MemoryState {
    Fitted,
    Unfitted,
}

#[derive(Debug, Clone)]
pub struct Memory {
    state: MemoryState,
    pub X: Array2<f_>,
    X_offset: Array1<f_>,
    // X_trans: Array2<f_>,     //down
    // X_trans_inv: Array2<f_>, //up

    // Rot * Scale * X + x_offset = X_true
    // Scale_inv * Rot_inv * (X_true - x_offset) = X

    // Rot * Scale * X = X_true
    // Scale_inv * Rot_inv * X_true = X
    X_scale: Array2<f_>,      // up
    X_rotate: Array2<f_>,     // up
    X_scale_inv: Array2<f_>,  // down
    X_rotate_inv: Array2<f_>, // down

    pub y: Array1<f_>,
    y_offset: f_,
    y_scaling: f_,
}

impl Memory {
    pub fn new(d: usize) -> Memory {
        Memory {
            state: MemoryState::Unfitted,
            X: Array2::zeros((d, 0)),
            X_offset: Array1::zeros((d,)),
            // X_trans: Array2::eye(d),
            // X_trans_inv: Array2::eye(d),
            X_scale: Array2::eye(d),
            X_rotate: Array2::eye(d),
            X_scale_inv: Array2::eye(d),
            X_rotate_inv: Array2::eye(d),

            y: Array1::zeros((0,)),
            y_offset: 0.0,
            y_scaling: 1.0,
        }
    }

    pub fn state(&self) -> &MemoryState {
        &self.state
    }

    pub fn set_fitted(&mut self) {
        self.state = MemoryState::Fitted
    }

    fn set_unfitted(&mut self) {
        self.state = MemoryState::Unfitted
    }

    pub fn reset_transform(&mut self) {
        self.X = self.X();

        self.X_offset = Array1::zeros((self.X.nrows(),));
        // self.X_trans = Array2::eye(self.X.nrows());
        // self.X_trans_inv = Array2::eye(self.X.nrows());

        self.X_scale = Array2::eye(self.X.nrows());
        self.X_rotate = Array2::eye(self.X.nrows());
        self.X_scale_inv = Array2::eye(self.X.nrows());
        self.X_rotate_inv = Array2::eye(self.X.nrows());

        self.y = self.y();

        self.y_offset = 0.0;
        self.y_scaling = 1.0;

        self.set_unfitted();
    }

    pub fn recenter_X(&mut self, x: ArrayView1<f_>) {
        self.X = self.X.clone().sub_column_view(&x); // TODO: REWORK UTIL FUNCS TO WORK WITH ARRAYBASE, ADD INPLACE METHODS

        // self.X_offset = self.X_offset.clone() + self.X_trans_inv.dot(&x); //TODO: SPURIOUS CLONE

        self.X_offset = self.X_offset.clone() + self.X_rotate.dot(&self.X_scale).dot(&x);

        self.set_unfitted();
    }

    pub fn rescale_X(&mut self, l: ArrayView1<f_>, prior_sigma_cap: Option<f_>) {
        fn cap_l(l: &f_, prior_sigma: &f_) -> f_ {
            match l {
                l if *l >= 1.0 + 3.0 * prior_sigma => 1.0 + 3.0 * prior_sigma,
                l if *l <= 1.0 - 3.0 * prior_sigma => 1.0 - 3.0 * prior_sigma,
                _ => *l,
            }
        }

        let l = match prior_sigma_cap {
            Some(sig) => l.map(|l| cap_l(l, &sig)),
            None => l.to_owned(),
        };

        let l_recip = Array2::from_diag(&l.map(|val| 1.0 / val));
        let l = Array2::from_diag(&l);

        self.X_scale_inv = l_recip.dot(&self.X_scale_inv);
        self.X_scale = l.dot(&self.X_scale);

        self.X = l_recip.dot(&self.X);

        self.set_unfitted();
    }

    pub fn rotate_X(&mut self) -> Result<()> {
        let W = Array2::from_diag(&self.y.map(|y_i| 1.0 - y_i));

        let (u, _, _) = self.X_scale.dot(&self.X).dot(&W).svd(true, false)?;
        // let (u, _, _) = self.X.svd(true, false)?;
        let u = u.expect("Unwrap of U should not fail");

        self.X_rotate_inv = u.t().dot(&self.X_rotate_inv);
        self.X_rotate = self.X_rotate.dot(&u);
        // self.X_rotate = u.dot(&self.X_rotate);

        // self.X = self
        //     .X_scale_inv
        //     .dot(&self.X_rotate_inv)
        //     .dot(&X_prime.sub_column(&self.X_offset));
        // dbg!(&self.X());

        self.X = self.X_scale_inv.dot(&u.t()).dot(&self.X_scale).dot(&self.X);

        self.set_unfitted();

        Ok(())
    }

    pub fn rescale_y(&mut self) {
        let min = self.y_prime_min();

        self.y -= min;
        self.y_offset += self.y_scaling * min;

        let max = match self.y_prime_max() {
            // Avoid rescaling if vals are all equal, i.e. max = min = 0.0
            x if x == 0.0 => 1.0,
            x => x,
        };

        self.y_scaling *= max;
        self.y /= max;

        self.set_unfitted();
    }

    pub fn rescale_X_bounds(&mut self, bounds: &ArrayBounds) {
        self.reset_transform();

        let b_midpnt = bounds.midpoint();
        let b_lens = bounds.axes_len();
        self.recenter_X(b_midpnt.view());
        self.rescale_X(b_lens.view(), None);

        self.set_unfitted();
    }

    pub fn append(&mut self, mut X: Array2<f_>, y: Array1<f_>) {
        if !X.shape().ends_with(&[y.len()]) {
            panic!("Number of inputs and outputs differ");
        }

        // X = self.X_trans.dot(&X.sub_column(&self.X_offset));
        X = self
            .X_scale_inv
            .dot(&self.X_rotate_inv)
            .dot(&X.sub_column(&self.X_offset));

        let mut y = y;
        y -= self.y_offset;
        y /= self.y_scaling; //MOVE TO OWN FN?

        self.X
            .append(Axis(1), X.view())
            .expect("append should never fail");
        self.y
            .append(Axis(0), y.view())
            .expect("append should never fail");

        self.set_unfitted();
    }

    pub fn forget(&mut self, search_dom: &ArrayBounds, min: usize) {
        let n_to_forget = self.X.ncols().saturating_sub(min);

        if n_to_forget == 0 {
            return;
        }

        let mut forget_indexes: Vec<usize> = self
            .X
            .columns()
            .into_iter()
            .enumerate()
            .filter(|(_, col)| !search_dom.inside(*col))
            .map(|(i, _)| i)
            .collect();

        forget_indexes.truncate(n_to_forget);

        self.y = self.y.clone().rem_at_index(forget_indexes.clone());
        self.X = self.X.clone().rem_cols(forget_indexes);

        self.set_unfitted();
    }

    #[inline(always)]
    pub fn X(&self) -> Array2<f_> {
        // self.X_trans_inv.dot(&self.X).add_column(&self.X_offset)
        (self.X_rotate.dot(&self.X_scale).dot(&self.X)).add_column(&self.X_offset)
    }

    #[inline(always)]
    pub fn x_test(&self, x_test: ArrayView1<f_>) -> Array1<f_> {
        // self.X_trans_inv.dot(&x_test) + &self.X_offset
        self.X_rotate.dot(&self.X_scale).dot(&x_test) + &self.X_offset
    }

    #[inline(always)]
    fn X_prime_min(&self) -> ArrayView1<f_> {
        self.X.column(self.min_index())
    }

    #[inline(always)]
    pub fn X_min(&self) -> Array1<f_> {
        self.x_test(self.X_prime_min())
    }

    #[inline(always)]
    pub fn y(&self) -> Array1<f_> {
        self.y.mapv(|val| val * self.y_scaling) + self.y_offset
    }

    #[inline(always)]
    pub fn y_test(&self, y_test: f_) -> f_ {
        (y_test * self.y_scaling) + self.y_offset
    }

    #[inline(always)]
    pub fn y_m(&self) -> Array1<f_> {
        &self.y - self.y_prime_mean()
    }

    #[inline(always)]
    pub fn y_prime_mean(&self) -> f_ {
        self.y.mean().expect("Cannot get mean of empty y array!")
    }

    #[inline(always)]
    pub fn y_prime_min(&self) -> f_ {
        self.y
            .min()
            .copied()
            .expect("Cannot get min of empty array or array with invalid values!")
    }

    #[inline(always)]
    pub fn y_prime_max(&self) -> f_ {
        self.y
            .max()
            .copied()
            .expect("Cannot get max of empty array or array with invalid values!")
    }

    #[inline(always)]
    pub fn y_min(&self) -> f_ {
        self.y_test(self.y_prime_min())
    }

    #[inline(always)]
    pub fn y_scaling(&self) -> f_ {
        self.y_scaling
    }

    #[inline(always)]
    pub fn min_index(&self) -> usize {
        self.y
            .indexed_min()
            .expect("Cannot get min of empty array or array with invalid values!")
            .0
    }

    #[inline(always)]
    pub fn y_prime_std_dev(&self) -> f_ {
        self.y.std(0.0)
    }

    #[inline(always)]
    pub fn n(&self) -> usize {
        self.X.ncols()
    }

    pub fn in_memory(&self, x: ArrayView1<f_>) -> bool {
        self.X
            .columns()
            .into_iter()
            .any(|col| col.abs_diff_eq(&x, 1e-12))
    }

    pub fn corners_2D(&self) -> Array2<f_> {
        if self.X.nrows() != 2 {
            panic!("corners_2D is not implemented for problems with dim != 2!")
        }

        let mut corners = array![[1.0, -1.0, -1.0, 1.0], [1.0, 1.0, -1.0, -1.0]];
        // let mut corners = array![[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        // let mut corners = Array2::<f_>::eye(2);

        corners.columns_mut().into_iter().for_each(|mut col| {
            let new_col = self.x_test(col.view());
            col.assign(&new_col)
        });

        // println!("prod {}", &self.X_trans.dot(&self.X_trans_inv));

        // corners.columns_mut().into_iter().for_each(|mut col| {
        //     let new_col = self.X_trans_inv.dot(&col.view());
        //     col.assign(&new_col)
        // });

        corners
    }

    pub fn next_pcs(&self) -> Array2<f_> {
        let mut pcs = array![
            [2.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
        ];

        let W = Array2::from_diag(&self.y.map(|y_i| 1.0 - y_i));
        // let W = Array2::from_diag(&self.y.map(|y_i| 1.0));

        if self.X.ncols() > 10 {
            let (u, _, _) = self.X_scale.dot(&self.X).dot(&W).svd(true, false).unwrap();
            // let (u, _, _) = self.X.svd(true, false)?;
            let u = u.expect("Unwrap of U should not fail");
            // let u = array![[0.984807753, -0.1736481777], [0.1736481777, 0.984807753]];
            pcs = u.dot(&pcs);
        }

        pcs.columns_mut().into_iter().for_each(|mut col| {
            let new_col = self.X_rotate.dot(&col) + &self.X_offset; //self.x_test(col.view());
            col.assign(&new_col)
        });

        pcs

        // if self.X.ncols() > 10 {
        //     dbg!("SVD");
        //     if let Ok(res) = self.X.dot(&W).svd(true, false) {
        //         dbg!("SVD OK");

        //         X.columns_mut().into_iter().for_each(|mut col| {
        //             let new_col = self.x_test(col.view());
        //             col.assign(&new_col)
        //         });

        //         let (u, _, _) = res;
        //         let u = u.unwrap();

        //         let X_rotate_inv = u.t().dot(&self.X_rotate_inv);
        //         self.X_scale_inv.dot(&X_rotate_inv).dot(&X.sub_column(&self.X_offset))
        //     } else {
        //         array![[2.0, 0.0, 0.0], [0.0, 0.0, 2.0]]
        //     }
        // } else {
        //     array![[2.0, 0.0, 0.0], [0.0, 0.0, 2.0]]
        // }

        // X.columns_mut().into_iter().for_each(|mut col| {
        //     let new_col = self.x_test(col.view());
        //     col.assign(&new_col)
        // });

        // if let Ok(res) = self.X.dot(&W).svd(true, false) {
        //     let (u, _, _) = res;
        //     let u = u.unwrap();
        //     self.X_scale_inv.dot(&u.t()).dot(&self.X_rotate_inv).dot(&X.sub_column(&self.X_offset))
        // } else {
        //     array![[2.0, 0.0, 0.0], [0.0, 0.0, 2.0]]
        // }

        // let (u, _, _) = self.X.svd(true, false)?;
        // match u {
        //     Some(u) => self.X_scale_inv.dot(&u.t()).dot(&self.X_rotate_inv).dot(&X.sub_column(&self.X_offset)),
        //     None => array![[2.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
        // }
    }

    pub fn angle(&self) -> f_ {
        let mut corners = array![[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];

        corners.columns_mut().into_iter().for_each(|mut col| {
            let new_col = self.X_rotate.dot(&self.X_scale).dot(&col.view());
            col.assign(&new_col)
        });

        let a = corners.column(0);
        let b = corners.column(2);

        (a.dot(&b)) / (a.norm_l2() * b.norm_l2())
    }
}
