use std::ops::Mul;

use anyhow::{bail, Result};
use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, ArrayView1, Axis};
use ndarray_linalg::SolveC;
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use oxidized_l_bfgs_b_c::Lbfgsb;
use rand::Rng;
use sobol::params::JoeKuoD6;
use sobol::Sobol;
use statrs::distribution::{Continuous, ContinuousCDF};

// use cobyla::{fmin_cobyla, CstrFn};

use crate::{
    f_,
    gp::GP,
    kernel::{Kernel, SquaredExponential},
    utils::{Array2Utils, ArrayView1Utils},
};

pub trait ExpectedImprovement {
    fn ei(&self, x: ArrayView1<f_>) -> f_;
    fn ei_vec(&self, x: &Vec<f_>) -> f_;
    fn ei_slice(&self, x: &[f_]) -> f_;
    // fn ei_barrier(&self, x: &Vec<f_>) -> f_;
    fn ei_jac(&self, x: ArrayView1<f_>) -> Array1<f_>;
    fn ei_jac_vec(&self, x: &Vec<f_>) -> Vec<f_>;
    fn optimize_ei(&self, x: ArrayView1<f_>) -> Result<(f_, Array1<f_>)>;
    fn optimize_ei_par(&self, n: usize) -> Result<(f_, Array1<f_>)>;
    fn random_pt(&self, n: usize) -> Result<(f_, Array1<f_>)>;
}

impl<kern: Kernel> ExpectedImprovement for GP<kern> {
    //checked
    fn ei(&self, x: ArrayView1<f_>) -> f_ {
        let (mean, sigma) = self.predict_single(x).unwrap();

        let z = (self.mem.y_prime_min() - mean) / sigma;

        sigma * (z * self.n.cdf(z) + self.n.pdf(z))
    }

    // Worked better with wrong jac? THIS ONE IS RIGHT
    fn ei_jac(&self, x: ArrayView1<f_>) -> Array1<f_> {
        //TODO: Change back to results?

        let (mean, sigma) = self.predict_single(x).unwrap();
        let z = (self.mem.y_prime_min() - mean) / sigma;

        let dkT = self.kernel.obs_jac(&self.mem.X, x);

        let dalpha = self
            .L
            .solvec(&self.kernel.k_diag(self.mem.X.view(), x))
            .unwrap();

        let ds = dkT.dot(&dalpha).mul(sigma.recip());

        let dz = dkT
            .dot(&self.alpha)
            .scaled_add_Array1(-1.0 * z, &ds, Axis(1));

        (ds.mul(z * self.n.cdf(z) + self.n.pdf(z)) - dz.mul(self.n.cdf(z))).remove_axis(Axis(1))
        //should be +
    }

    fn ei_vec(&self, x: &Vec<f64>) -> f_ {
        // let x = Array1::from_vec(x.to_owned()); //unnessesary clone
        // -1.0 * self.ei(x.view())

        unsafe {
            // dangerous direct cast from vec to ArrayView1
            let view = ndarray::ArrayView::from_shape_ptr((x.len(),), x.as_ptr());
            -1.0 * self.ei(view)
        }
    }

    fn ei_slice(&self, x: &[f64]) -> f_ {
        // let x = Array1::from_vec(x.to_owned()); //unnessesary clone
        // -1.0 * self.ei(x.view())

        unsafe {
            // dangerous direct cast from vec to ArrayView1
            let view = ndarray::ArrayView::from_shape_ptr((x.len(),), x.as_ptr());
            // println!("EI {}", -1.0 * self.ei(view));
            -1.0 * self.ei(view)
        }
    }

    fn ei_jac_vec(&self, x: &Vec<f_>) -> Vec<f_> {
        // let x = Array1::from_vec(x.to_owned()); //unnessesary clone
        // self.ei_jac(x.view()).mul(-1.0).to_vec()

        unsafe {
            // dangerous direct cast from vec to ArrayView1
            let view = ndarray::ArrayView::from_shape_ptr((x.len(),), x.as_ptr());
            self.ei_jac(view).mul(-1.0).to_vec()
        }
    }

    // L-BFGS-B
    fn optimize_ei(&self, x: ArrayView1<f_>) -> Result<(f_, Array1<f_>)> {
        // let startinner = Instant::now();

        let mut x = x.to_vec();

        let f = |x: &Vec<f64>| self.ei_vec(x);
        let g = |x: &Vec<f64>| self.ei_jac_vec(x);

        let mut min = Lbfgsb::new(&mut x, &f, &g);

        // set bounds
        for d in 0..self.dim {
            min.set_lower_bound(d, -self.beta);
            min.set_upper_bound(d, self.beta);
        }

        //solver options TODO: parametrize?
        min.set_verbosity(-1);
        min.set_termination_tolerance(1e3);
        if let None = min.minimize() {
            bail!("EI minimization failed!")
        }
        let x_final = Array1::from_vec(x);

        let final_ei = self.ei(x_final.view());

        if final_ei.is_nan() {
            bail!("EI minimization failed!")
        }

        // println!("(EI) sing. opt Elapsed time: {:.6?}", startinner.elapsed());
        Ok((final_ei, x_final))
    }

    // COBYLA
    // fn optimize_ei(&self, x: ArrayView1<f_>) -> Result<(f_, Array1<f_>)> {
    //     // let startinner = Instant::now();

    //     let mut x = x.to_vec();

    //     let f = |x: &[f64], data: &mut ()| self.ei_slice(x);
    //     let g = |x: &Vec<f64>| self.ei_jac_vec(x);

    //     // let mut min = Lbfgsb::new(&mut x, &f, &g);

    //     let mut cons: Vec<&dyn CstrFn> = vec![];
    //     let binding = |x: &[f64]| self.beta - x[0];
    //     cons.push(&binding);
    //     let binding = |x: &[f64]| x[0] - self.beta;
    //     cons.push(&binding);

    //     let binding = |x: &[f64]| self.beta - x[1];
    //     cons.push(&binding);
    //     let binding = |x: &[f64]| x[1] - self.beta;
    //     cons.push(&binding);

    //     // set bounds
    //     for d in 0..self.dim {

    //         // cons.push(&|x: &[f64]| 0.5 - x[d]);
    //         // cons.push(&|x: &[f64]| x[d] - 0.5);
    //     }

    //     let (status, x_opt) = fmin_cobyla(f, &mut x, &cons, (), 0.5, 1e-4, 200, 0);
    //     println!("status = {}", status);
    //     println!("x = {:?}", x_opt);

    //     if status == 0 {
    //         let x_final = Array1::from_vec(x);

    //         return Ok((self.ei(x_final.view()), x_final));
    //     }
    //     else
    //     {
    //         bail!("EI minimization failed!")
    //     }
    // }

    // seems to be working
    fn optimize_ei_par(&self, n: usize) -> Result<(f_, Array1<f_>)> {
        // let start = Instant::now();

        // let mut X = self.search_dom.LHS_sample(20 - 1);

        // let start = Instant::now();
        // let mut X = self.search_dom_LHS.clone();

        //////////////////////////////////

        let params = JoeKuoD6::minimal();
        let seq = Sobol::<f_>::new(self.dim, &params);
        let sob: Vec<f_> = seq
            .take(n - 1)
            .flatten()
            .map(|val| val / self.beta)
            .collect(); // sobol seq accross [-beta, beta]^d
        let mut X = Array2::from_shape_vec((self.dim, n - 1), sob).unwrap();

        X.append(
            Axis(1),
            self.mem.X.column(self.mem.min_index()).clone().into_col(),
        )
        .expect("append should never fail");

        //////////////////////////

        // let start = Instant::now();

        // let ei = |x: &DVector<f_>| self.ei_barrier(x.data.as_vec());

        // let strategy = RestartStrategy::BIPOP(Default::default());
        // let restarter = RestartOptions::new(self.dim, -self.beta..=self.beta, strategy)
        //     .max_function_evals_per_run(200)
        //     .enable_printing(true)
        //     .build()
        //     .unwrap();

        // let results = restarter.run_parallel(|| ei);
        // let best = results.best.unwrap();

        // println!("(EI) cma Elapsed time: {:.6?}", start.elapsed());
        // let start = Instant::now();

        // Ok((best.value, Array1::from_vec(best.point.data.as_vec().to_vec())))

        // let min = self.mem.X.column(self.mem.min_index()).to_owned();
        // let mut cmaes_state = CMAESOptions::new(min.to_vec(), 1.0)
        //     .fun_target(1e-8)
        //     .enable_printing(200)
        //     .max_generations(20000)
        //     .build(sphere)
        //     .unwrap();

        // let results = cmaes_state.run();

        //////////////////////////

        // dbg!(&self.mem.X);
        // dbg!(&self.mem.X());
        //gives heap corruption error

        // let res: Vec<Result<(f_, Array1<f_>)>> = (0..X.ncols())
        //     .into_par_iter()
        //     .map(|i| -> Result<(f_, Array1<f_>)> { optimize_ei_cloned(self.force_clone(), X.column(i)) })
        //     .collect();

        // let res: Vec<Result<(f_, Array1<f_>)>> = (0..X.ncols())
        //     .chunks(10).into_iter().flat_map(|r| )
        //     .into_par_iter()
        //     .map(|i| -> Result<(f_, Array1<f_>)> { self.optimize_ei(X.column(i)) })
        //     .collect(); //unfinished

        ///////////////////////// works with chunk_size > 200
        // // let chunk_size = (X.ncols()) / 15;
        // let chunk_size = 40;

        // let res: Vec<Result<(f_, Array1<f_>)>> = (0..X.ncols())
        //     .into_par_iter()
        //     .chunks(chunk_size)
        //     .flat_map_iter(|range| range.iter().map(|i| self.optimize_ei(X.column(*i))).collect::<Vec<Result<(f_, Array1<f_>)>>>()).collect();
        // .flat_map_iter(|range| range.iter().map(|i| optimize_ei_cloned(self.force_clone(), X.column(*i))).collect::<Vec<Result<(f_, Array1<f_>)>>>()).collect();
        // dbg!(&res);

        ////////////////////////////////
        // println!("(EI) setup Elapsed time: {:.6?}", start.elapsed());
        // let start = Instant::now();

        let mut res = vec![]; //working
        res.extend(X.columns().into_iter().map(|col| self.optimize_ei(col)));
        // res.extend(X.columns().into_iter().map(|col| optimize_ei_cloned(self.force_clone(), col) ));

        // println!("(EI) opt Elapsed time: {:.6?}", start.elapsed());
        // let start = Instant::now();
        let max = res
            .into_iter()
            .filter_map(|res| res.ok())
            .filter_map(|tup| match tup.0.is_finite() {
                true => Some(tup),
                false => None,
            })
            .filter_map(|tup| match !self.mem.in_memory(tup.1.view()) {
                true => Some(tup),
                false => None,
            })
            .filter_map(
                |tup| match self.bounds.inside((self.mem.x_test(tup.1.view())).view()) {
                    // Rejection sampling for target f bounds
                    true => Some(tup),
                    false => None,
                },
            )
            .max_by(|(a, _), (b, _)| (a).total_cmp(b));
        // println!("(EI) max Elapsed time: {:.6?}", start.elapsed());
        match max {
            Some(max) => Ok(max),
            None => self.random_pt(self.dim * 100),
        }

        ////////////////////////////////

        // let mut rng = rand::thread_rng();

        // let mut best_ei = 0.0;
        // let mut best_ei_pnt = self.mem.X.column(self.mem.min_index()).to_owned();

        // let mut x = self.mem.X.column(self.mem.min_index()).to_owned();
        // let x_res = self.optimize_ei(x.view());

        // (best_ei, best_ei_pnt) = match x_res {
        //     Ok((start_ei, start_ei_pnt)) => {
        //         if self
        //             .bounds
        //             .inside((self.mem.x_test(start_ei_pnt.view())).view())
        //         {
        //             (start_ei, start_ei_pnt)
        //         } else {
        //             (best_ei, best_ei_pnt)
        //         }
        //     }
        //     Err(_) => (best_ei, best_ei_pnt),
        // };

        // for _ in 0..n {
        //     x.iter_mut()
        //         .for_each(|i| *i = rng.gen::<f64>() * 2.0f64 * self.beta - 1.0f64 * self.beta);

        //     let x_res = self.optimize_ei(x.view());

        //     let (x_ei, x_pnt) = match x_res {
        //         Ok(x_ok) => x_ok,
        //         Err(_) => continue,
        //     };

        //     if x_ei > best_ei && self.bounds.inside((self.mem.x_test(x_pnt.view())).view()) {
        //         best_ei = x_ei;
        //         best_ei_pnt = x_pnt;
        //     }
        // }

        // if best_ei_pnt.iter().all(|val| *val == 0.0) {
        //     self.random_pt(100)
        // } else {
        //     Ok((best_ei, best_ei_pnt))
        // }

        ////////////////////////////////
    }

    fn random_pt(&self, n: usize) -> Result<(f_, Array1<f_>)> {
        // println!("RAND");
        for _ in 0..n {
            let x = Array1::random((self.dim,), Uniform::new(-self.beta, self.beta));

            if self.mem.in_memory(x.view()) {
                continue;
            }

            if !self.bounds.inside((self.mem.x_test(x.view())).view()) {
                continue;
            }

            return Ok((self.ei(x.view()), x));
        }

        bail!("EI point not found!")
    }
}
