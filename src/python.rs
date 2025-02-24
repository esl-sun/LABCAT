use anyhow::Result;
use ndarray::prelude::*;
use numpy::ToPyArray;
use std::marker::PhantomData;
use std::time::{Duration, Instant};

#[cfg(feature = "python")]
use numpy::IntoPyArray;
#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::acq::ExpectedImprovement;
use crate::bounds::{Bounds, Ready};
use crate::gp::GP;
use crate::hyp_opt::HyperparameterOptimizer;
use crate::kernel::Kernel;
use crate::utils::Array1Utils;
use crate::{Auto, Config, GPState, LABCAT, LABCATReadyState, Manual, f_};
use crate::{LABCATConfig, OptimizationSummary};

#[cfg(feature = "python")]
#[derive(Clone)]
pub struct pyConfig {
    py_callable_target_fn: Option<PyObject>,
    py_callable_init_fn: Option<PyObject>,
    py_callable_forget_fn: Option<PyObject>,
}

impl pyConfig {
    pub fn new() -> pyConfig {
        pyConfig {
            py_callable_target_fn: None,
            py_callable_init_fn: None,
            py_callable_forget_fn: None,
        }
    }

    pub fn target_fn<'py>(&self, x: &Array2<f_>, py: Python<'py>) -> Array1<f_> {
        let x = x.to_pyarray(py);

        match &self.py_callable_target_fn {
            Some(f) => match f.call1(py, (x,)) {
                Ok(ret) => match ret.extract::<Vec<f_>>(py) {
                    Ok(vec) => Array1::from_vec(vec),
                    Err(_) => panic!(
                        "Python target function must return a value that can be parsed to vector!"
                    ),
                },
                Err(_) => panic!("Failed to call python target function!"),
            },
            None => panic!("Python target function not set!!"),
        }
    }

    pub fn init_pts_fn<'py>(&self, d: usize, py: Python<'py>) -> usize {
        match &self.py_callable_init_fn {
            Some(f) => match f.call1(py, (d,)) {
                Ok(ret) => match ret.extract::<usize>(py) {
                    Ok(val) => val,
                    Err(_) => panic!(
                        "Python init points number function must return a value that can be parsed to usize!"
                    ),
                },
                Err(_) => panic!("Failed to call python initial points number function!"),
            },
            None => d + 1,
        }
    }

    pub fn forget_fn<'py>(&self, d: usize, py: Python<'py>) -> usize {
        match &self.py_callable_forget_fn {
            Some(f) => match f.call1(py, (d,)) {
                Ok(ret) => match ret.extract::<usize>(py) {
                    Ok(val) => val,
                    Err(_) => panic!(
                        "Python forget number function must return a value that can be parsed to usize!"
                    ),
                },
                Err(_) => panic!("Failed to call python forget number function!"),
            },
            None => d * 10,
        }
    }
}

#[cfg(feature = "python")]
impl LABCAT {
    pub fn new(bounds: Bounds<Ready>) -> LABCAT<Config> {
        let gp = GP::new(bounds.bounds_arr().to_owned(), 0.5, 0.15);

        #[cfg(feature = "LHS")]
        let init_points = bounds.bounds_arr().LHS_sample(bounds.dim() + 1);
        #[cfg(not(feature = "LHS"))]
        let init_points = bounds.bounds_arr().random_sample(bounds.dim() + 1);

        let gp_state = GPState::Init(init_points);
        let config = LABCATConfig::new();
        let py_config = pyConfig::new();

        LABCAT {
            gp,
            gp_state,
            bounds,
            config,
            py_config,
            config_state: PhantomData,
        }
    }

    pub fn new_preconfigured<'py>(
        bounds: Bounds<Ready>,
        config: LABCATConfig,
        py_config: pyConfig,
        py: Python<'py>,
    ) -> LABCAT<Manual> {
        let gp = GP::new(
            bounds.bounds_arr().to_owned(),
            config.beta.into(),
            config.prior_sigma.into(),
        );
        let init_n = py_config.init_pts_fn(bounds.dim(), py);

        #[cfg(feature = "LHS")]
        let init_points = bounds.bounds_arr().LHS_sample(init_n);
        #[cfg(not(feature = "LHS"))]
        let init_points = bounds.bounds_arr().random_sample(init_n);

        let gp_state = GPState::Init(init_points);

        LABCAT {
            gp,
            gp_state,
            bounds,
            config,
            py_config,
            config_state: PhantomData,
        }
    }

    pub fn bounds(&self) -> &Bounds<Ready> {
        &self.bounds
    }

    pub fn config(&self) -> &LABCATConfig {
        &self.config
    }

    pub fn pyConfig(&self) -> &pyConfig {
        &self.py_config
    }
}

#[cfg(feature = "python")]
impl LABCAT<Config> {
    pub fn beta(&mut self, beta: f_) {
        self.config.beta = beta;
    }

    pub fn prior_sigma(&mut self, sigma: f_) {
        self.config.prior_sigma = sigma;
    }

    pub fn init_pts_fn(&mut self, f: PyObject) {
        self.py_config.py_callable_init_fn = Some(f);
    }

    pub fn forget_fn(&mut self, f: PyObject) {
        self.py_config.py_callable_forget_fn = Some(f);
    }

    pub fn target_tol(&mut self, tol: f_) {
        self.config.target_tol = tol.into();
    }

    pub fn target_val(&mut self, val: f_) {
        self.config.target_val = Some(val);
    }

    pub fn restarts(&mut self, restarts: bool) {
        self.config.restarts = restarts;
    }

    pub fn max_samples(&mut self, n: usize) {
        self.config.max_samples = Some(n)
    }

    pub fn max_time(&mut self, dur: Duration) {
        self.config.max_time = Some((dur, Instant::now()));
    }

    pub fn build<'py>(self, py: Python<'py>) -> LABCAT<Manual> {
        let gp = GP::new(
            self.bounds.bounds_arr().to_owned(),
            self.config.beta.into(),
            self.config.prior_sigma.into(),
        );

        let init_n = self.py_config.init_pts_fn(self.bounds.dim(), py);

        #[cfg(feature = "LHS")]
        let init_points = self.bounds.bounds_arr().LHS_sample(init_n);
        #[cfg(not(feature = "LHS"))]
        let init_points = self.bounds.bounds_arr().random_sample(init_n);

        let gp_state = GPState::Init(init_points);

        LABCAT {
            gp,
            gp_state,
            bounds: self.bounds,
            config: self.config,
            py_config: self.py_config,
            config_state: PhantomData,
        }
    }
}

#[cfg(feature = "python")]
impl<S: LABCATReadyState> LABCAT<S> {
    fn restart<'py>(&mut self, _err: anyhow::Error, py: Python<'py>) {
        let mut gp = GP::new(
            self.bounds.bounds_arr().to_owned(),
            self.config.beta.into(),
            self.config.prior_sigma.into(),
        );

        let init_n = self.py_config.init_pts_fn(self.bounds.dim(), py);

        #[cfg(feature = "LHS")]
        let init_points = self.bounds.bounds_arr().LHS_sample(init_n - 1);
        #[cfg(not(feature = "LHS"))]
        let init_points = self.bounds.bounds_arr().random_sample(init_n - 1);

        let gp_state = GPState::Init(init_points);

        gp.mem.append(
            self.gp.mem.X_min().into_col(),
            Array1::from_elem((1,), self.gp.mem.y_min()),
        );
        self.gp = gp;
        self.gp_state = gp_state;
    }

    fn _suggest<'py>(&mut self, py: Python<'py>) -> Array2<f_> {
        self.state_transition();
        match self.gp_state.clone() {
            GPState::Init(init_pts) => init_pts,
            GPState::Nominal => match self.step_alogrithm(py) {
                Ok(Arr) => Arr,
                Err(err) => {
                    self.restart(err, py);
                    self._suggest(py)
                }
            },
        }
    }

    fn step_alogrithm<'py>(&mut self, py: Python<'py>) -> Result<Array2<f_>> {
        let min = self.gp.mem.X.column(self.gp.mem.min_index()).to_owned();
        self.gp.mem.recenter_X(min.view());

        self.gp.mem.rescale_y();

        #[cfg(feature = "PCA")]
        self.gp.mem.rotate_X()?;

        self.gp.fit()?;

        match self.gp.optimize_thetas() {
            Ok(_) => assert!(true),
            Err(_) => {
                self.gp.kernel.whiten_l();
                self.gp.fit()?;
            }
        };

        self.gp
            .mem
            .rescale_X(self.gp.kernel.l(), Some(self.config.prior_sigma.into()));
        self.gp.kernel.whiten_l();

        let min_n = self.py_config.forget_fn(self.bounds.dim(), py);
        self.gp.mem.forget(&self.gp.search_dom, min_n);

        self.gp.fit()?;
        let (_, ei_pt) = self.gp.optimize_ei(10 * self.bounds.dim())?;
        let scaled_ei_pt = self.gp.mem.x_test(ei_pt.view()).into_col();
        Ok(scaled_ei_pt)
    }
}

#[cfg(feature = "python")]
impl LABCAT<Manual> {
    pub fn set_target_fn(self, f: PyObject) -> LABCAT<Auto> {
        let py_config = pyConfig {
            py_callable_init_fn: self.py_config.py_callable_init_fn,
            py_callable_forget_fn: self.py_config.py_callable_forget_fn,
            py_callable_target_fn: Some(f),
        };

        LABCAT {
            gp: self.gp,
            gp_state: self.gp_state,
            bounds: self.bounds,
            config: self.config,
            py_config,
            config_state: PhantomData,
        }
    }

    pub fn suggest<'py>(&mut self, py: Python<'py>) -> Array2<f_> {
        self._suggest(py)
    }

    pub fn observe(&mut self, X: Array2<f_>, y: Array1<f_>) {
        self._observe(X, y)
    }

    pub fn thetas(&self) -> &Array1<f_> {
        self.gp.kernel.thetas()
    }

    pub fn predict(&self, mut x_prime: Array2<f_>) -> (Array2<f_>, Array2<f_>) {
        let y_prime = match self.gp.predict(x_prime.clone()) {
            Ok(res) => res.0,
            Err(_) => Array2::from_elem((x_prime.ncols(), 1), 0.0),
        };
        let y = y_prime.map(|y_prime| self.gp.mem.y_test(*y_prime));
        x_prime.columns_mut().into_iter().for_each(|mut col| {
            let new_col = self.gp.mem.x_test(col.view()); //self.x_test(col.view());
            col.assign(&new_col)
        });

        (x_prime, y)
    }

    pub fn observations(&self) -> (Array2<f_>, Array1<f_>) {
        (self.gp.mem.X(), self.gp.mem.y())
    }

    pub fn bounds(&self) -> &Bounds<Ready> {
        &self.bounds
    }

    pub fn config(&self) -> &LABCATConfig {
        &self.config
    }

    pub fn pyConfig(&self) -> &pyConfig {
        &self.py_config
    }
}

#[cfg(feature = "python")]
impl LABCAT<Auto> {
    pub fn print_interval(&mut self, interval: usize) {
        self.config.auto_print = Some(interval);
    }

    pub fn run<'py>(&mut self, py: Python<'py>) -> OptimizationSummary {
        let print = self.config.auto_print.is_some();
        if print {
            println!("{}", self.title());
            println!("{}", self.mid_border());
        }
        loop {
            let suggest = self._suggest(py);
            let samples = self.py_config.target_fn(&suggest, py);
            self._observe(suggest, samples);

            if print {
                if self.config.n_samples % self.config.auto_print.expect("Already checked option")
                    == 0
                {
                    println!("{}", self.iter_summary());
                }
            }

            if let Some(term) = self._check_converged() {
                if print {
                    println!("{}", self.bottom_border());
                }

                return OptimizationSummary {
                    term_reason: term,
                    n_samples: self.config.n_samples,
                    min_x: self.gp.mem.X().column(self.gp.mem.min_index()).to_owned(),
                    min_y: self.gp.mem.y()[self.gp.mem.min_index()],
                };
            };
        }
    }
}
