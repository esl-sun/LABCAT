#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(unused_imports)]
#![allow(dead_code)]

use anyhow::Result;
use std::fmt::Display;
use std::marker::PhantomData;
use std::time::{Duration, Instant};

use acq::ExpectedImprovement;
use hyp_opt::HyperparameterOptimizer;
// use fallible_option::Fallible::{self, Fail, Success};
use ndarray::Slice;
use ndarray::prelude::*;

#[cfg(feature = "python")]
use pyo3::prelude::*;

pub mod acq;
pub mod bound_types;
pub mod bounds;
pub mod bounds_array;
pub mod bounds_transforms;
pub mod gp;
pub mod hyp_opt;
pub mod kernel;
pub mod memory;
pub mod utils;

#[cfg(feature = "python")]
pub mod python;

use bounds::{Bounds, Ready};
use gp::GP;
use kernel::{Kernel, SquaredExponential};
use utils::Array1Utils;

#[cfg(feature = "python")]
use python::pyConfig;

#[cfg(not(feature = "f64"))]
pub type f_ = f32;
#[cfg(feature = "f64")]
pub type f_ = f64;

#[cfg(not(feature = "f64"))]
pub type i_ = i32;
#[cfg(feature = "f64")]
pub type i_ = i64;

#[derive(Debug, Clone, PartialEq)]
pub enum GPState {
    Init(Array2<f_>),
    Nominal,
    // Restart(Array2<f_>),
}

trait LABCATConfigState {}
pub trait LABCATReadyState {}

pub struct Config {}
pub struct Manual {}
pub struct Auto {}

impl LABCATConfigState for Config {}
impl LABCATConfigState for Manual {}
impl LABCATConfigState for Auto {}

impl LABCATReadyState for Manual {}
impl LABCATReadyState for Auto {}

#[derive(Debug)]
pub enum TermCond {
    MachineEpsilonReached,
    TargetTolReached,
    TargetValReached,
    MaxItersReached,
    MaxTimeReached,
    DidNotConverge,
}

#[derive(Debug)]
pub struct OptimizationSummary {
    term_reason: TermCond,
    n_samples: usize,
    min_x: Array1<f_>,
    min_y: f_,
}

#[derive(Debug, Clone)]
pub struct LABCATConfig {
    beta: f_, // TODO: back to f_?
    prior_sigma: f_,
    restarts: bool,
    target_tol: f_,
    target_val: Option<f_>,
    n_samples: usize,
    max_samples: Option<usize>,
    max_time: Option<(Duration, Instant)>,
    auto_print: Option<usize>,
}

impl LABCATConfig {
    pub fn new() -> LABCATConfig {
        LABCATConfig {
            beta: 0.5,
            prior_sigma: 0.15,
            restarts: false,
            target_tol: f_::EPSILON,
            target_val: None,
            n_samples: 0,
            max_samples: None,
            max_time: None,
            auto_print: None,
        }
    }
}

impl OptimizationSummary {
    pub fn reason(&self) -> String {
        match self.term_reason {
            TermCond::MachineEpsilonReached => "Machine Epsilon reached!".into(),
            TermCond::TargetTolReached => "Target tolerance reached!".into(),
            TermCond::TargetValReached => "Target value reached!".into(),
            TermCond::MaxItersReached => "Maximum sampling iterations reached!".into(),
            TermCond::MaxTimeReached => "Maximum wall-time reached!".into(),
            TermCond::DidNotConverge => "Did not converge!".into(),
        }
    }

    pub fn n_samples(&self) -> &usize {
        &self.n_samples
    }

    pub fn min_x(&self) -> &Array1<f_> {
        &self.min_x
    }

    pub fn min_y(&self) -> &f_ {
        &self.min_y
    }
}

impl Display for OptimizationSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "--------------------",)?;
        writeln!(f, "{}", self.reason())?;
        writeln!(
            f,
            "iter: {} y_min: {:.6e}, x_min: {:.3}",
            self.n_samples, self.min_y, self.min_x
        )?;
        writeln!(f, "--------------------",)?;
        Ok(())
    }
}

pub struct LABCAT<LABCATConfigState = Config> {
    gp: GP<SquaredExponential>,
    gp_state: GPState,
    bounds: Bounds<Ready>,
    config: LABCATConfig,
    config_state: PhantomData<LABCATConfigState>,

    #[cfg(not(feature = "python"))]
    target_fn: Option<fn(&Array2<f_>) -> Array1<f_>>,

    #[cfg(not(feature = "python"))]
    init_pts_fn: fn(usize) -> usize,

    #[cfg(not(feature = "python"))]
    forget_fn: fn(usize) -> usize,
    #[cfg(feature = "python")]
    py_config: pyConfig,
}

#[cfg(not(feature = "python"))]
impl LABCAT {
    pub fn new(bounds: Bounds<Ready>) -> LABCAT<Config> {
        let gp = GP::new(bounds.bounds_arr().to_owned(), 0.5, 0.1);
        let init_pts_fn = |d: usize| d + 1;
        let forget_fn = |d: usize| 10 * d;
        #[cfg(feature = "LHS")]
        let init_points = bounds.bounds_arr().LHS_sample(init_pts_fn(bounds.dim()));
        #[cfg(not(feature = "LHS"))]
        let init_points = bounds.bounds_arr().random_sample(init_pts_fn(bounds.dim()));
        let gp_state = GPState::Init(init_points);
        let config = LABCATConfig::new();

        LABCAT {
            gp,
            gp_state,
            target_fn: None,
            bounds,
            config,
            init_pts_fn,
            forget_fn,
            config_state: PhantomData,
        }
    }
}

#[cfg(not(feature = "python"))]
impl LABCAT<Config> {
    pub fn beta(mut self, beta: f_) -> Self {
        self.config.beta = beta;
        self
    }

    pub fn prior_sigma(mut self, sigma: f_) -> Self {
        self.config.prior_sigma = sigma;
        self
    }

    pub fn init_pts_fn(mut self, f: fn(usize) -> usize) -> Self {
        self.init_pts_fn = f;
        self
    }

    pub fn forget_fn(mut self, f: fn(usize) -> usize) -> Self {
        self.forget_fn = f;
        self
    }

    pub fn target_tol(mut self, tol: f_) -> Self {
        self.config.target_tol = tol.into();
        self
    }

    pub fn target_val(mut self, val: f_) -> Self {
        self.config.target_val = Some(val);
        self
    }

    pub fn restarts(mut self, restarts: bool) -> Self {
        self.config.restarts = restarts;
        self
    }

    pub fn max_samples(mut self, n: usize) -> Self {
        self.config.max_samples = Some(n);
        self
    }

    pub fn max_time(mut self, dur: Duration) -> Self {
        self.config.max_time = Some((dur, Instant::now()));
        self
    }

    pub fn build(self) -> LABCAT<Manual> {
        let gp = GP::new(
            self.bounds.bounds_arr().to_owned(),
            self.config.beta.into(),
            self.config.prior_sigma.into(),
        );

        #[cfg(feature = "LHS")]
        let init_points = self
            .bounds
            .bounds_arr()
            .LHS_sample((self.init_pts_fn)(self.bounds.dim()));
        #[cfg(not(feature = "LHS"))]
        let init_points = self
            .bounds
            .bounds_arr()
            .random_sample((self.init_pts_fn)(self.bounds.dim()));
        let gp_state = GPState::Init(init_points);

        LABCAT {
            gp,
            gp_state,
            bounds: self.bounds,
            config: self.config,
            config_state: PhantomData,

            target_fn: self.target_fn,
            init_pts_fn: self.init_pts_fn,
            forget_fn: self.forget_fn,
        }
    }
}

impl<S: LABCATReadyState> LABCAT<S> {
    pub fn n(&self) -> usize {
        self.gp.mem.n()
    }

    pub fn X_min(&self) -> Array1<f_> {
        self.gp.mem.X_min()
    }

    pub fn y_min(&self) -> f_ {
        self.gp.mem.y_min()
    }

    pub fn state(&self) -> &GPState {
        &self.gp_state
    }

    fn x_fill(&self) -> String {
        "\u{02500}"
            .chars()
            .cycle()
            .take(self.bounds.dim() * 11 + 2)
            .collect()
    }

    fn x_fill_title(&self) -> String {
        let offset = "\u{02500}".len() * 11;
        let mut title = self.x_fill();
        title.replace_range(..offset, &format!("{:^11}", "x_min"));
        title
    }

    fn title(&self) -> String {
        format!(
            "\u{0256D} {:^6} \u{0252C} {:^14} \u{0252C}{}\u{0256E}",
            "Iter",
            "y_min",
            self.x_fill_title()
        )
    }

    fn mid_border(&self) -> String {
        format!(
            "\u{0251C}{:\u{02500}^8}\u{0253C}{:\u{02500}^16}\u{0253C}{}\u{02524}",
            "\u{02500}",
            "\u{02500}",
            self.x_fill()
        )
    }

    fn bottom_border(&self) -> String {
        format!(
            "\u{02570}{:\u{02500}^8}\u{02534}{:\u{02500}^16}\u{02534}{}\u{0256F}",
            "\u{02500}",
            "\u{02500}",
            self.x_fill()
        )
    }

    fn iter_summary(&self) -> String {
        format!(
            "\u{02502} {:^6} \u{02502} {:^14} \u{02502} {:>9.5} \u{02502}",
            self.config.n_samples,
            format!("{:.6e}", self.gp.mem.y_min()),
            self.gp.mem.X_min()
        )
    }

    fn state_transition(&mut self) {
        match &self.gp_state {
            GPState::Init(init_pts) => {
                if init_pts.ncols() == 0 {
                    self.gp.mem.reset_transform();
                    self.gp.mem.rescale_X_bounds(&self.gp.bounds);
                    let min = self.gp.mem.X.column(self.gp.mem.min_index()).to_owned();
                    self.gp.mem.recenter_X(min.view());
                    self.gp.mem.rescale_y();
                    self.gp_state = GPState::Nominal;
                }
            }

            GPState::Nominal => assert!(true),
        }
    }

    #[cfg(not(feature = "python"))]
    fn restart(&mut self, _err: anyhow::Error) {
        println!("RESTARTING");
        println!("{}", _err);

        let mut gp = GP::new(
            self.bounds.bounds_arr().to_owned(),
            self.config.beta.into(),
            self.config.prior_sigma.into(),
        );

        #[cfg(feature = "LHS")]
        let init_points = self
            .bounds
            .bounds_arr()
            .LHS_sample((self.init_pts_fn)(self.bounds.dim() - 1));
        #[cfg(not(feature = "LHS"))]
        let init_points = self
            .bounds
            .bounds_arr()
            .random_sample((self.init_pts_fn)(self.bounds.dim() - 1));

        let gp_state = GPState::Init(init_points);

        gp.mem.append(
            self.gp.mem.X_min().into_col(),
            Array1::from_elem((1,), self.gp.mem.y_min()),
        );
        self.gp = gp;
        self.gp_state = gp_state;
    }

    #[cfg(not(feature = "python"))]
    fn _suggest(&mut self) -> Array2<f_> {
        self.state_transition();
        match self.gp_state.clone() {
            GPState::Init(init_pts) => init_pts,
            GPState::Nominal => match self.step_alogrithm() {
                Ok(Arr) => Arr,
                Err(err) => {
                    self.restart(err);
                    self._suggest()
                }
            },
        }
    }

    #[cfg(not(feature = "python"))]
    fn step_alogrithm(&mut self) -> Result<Array2<f_>> {
        let min = self.gp.mem.X.column(self.gp.mem.min_index()).to_owned();
        self.gp.mem.recenter_X(min.view());

        self.gp.mem.rescale_y();

        #[cfg(feature = "PCA")]
        self.gp.mem.rotate_X()?;

        self.gp.fit()?;

        // self.gp.optimize_thetas()?;
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

        self.gp
            .mem
            .forget(&self.gp.search_dom, (self.forget_fn)(self.bounds.dim()));

        self.gp.fit()?;
        let (_, ei_pt) = self.gp.optimize_ei(10 * self.bounds.dim())?;
        let scaled_ei_pt = self.gp.mem.x_test(ei_pt.view()).into_col();
        Ok(scaled_ei_pt)
    }

    fn _observe(&mut self, X: Array2<f_>, y: Array1<f_>) {
        self.config.n_samples += X.ncols();

        self.state_transition();
        match self.gp_state.clone() {
            GPState::Init(mut init_pts) => {
                let s = Slice::new(0, Some(-(X.ncols() as isize)), 1);
                self.gp.mem.append(X, y);
                init_pts.slice_axis_inplace(Axis(1), s);
                self.gp_state = GPState::Init(init_pts);
            }
            GPState::Nominal => {
                self.gp.mem.append(X, y);
            } // GPState::Restart(_) => panic!("Should never trigger"),
        }
    }

    pub fn _check_converged(&self) -> Option<TermCond> {
        if self.gp.mem.y_scaling() < self.config.target_tol.into() {
            if self.config.target_tol == f_::EPSILON {
                return Some(TermCond::MachineEpsilonReached);
            } else {
                return Some(TermCond::TargetTolReached);
            }
        };

        if let Some(val) = self.config.target_val {
            if self.gp.mem.y()[self.gp.mem.min_index()] <= val.into() {
                return Some(TermCond::TargetValReached);
            }
        }

        if let Some(max) = self.config.max_samples {
            if self.config.n_samples >= max {
                return Some(TermCond::MaxItersReached);
            }
        }

        if let Some((dur, start)) = self.config.max_time {
            if Instant::now().duration_since(start) > dur {
                return Some(TermCond::MaxTimeReached);
            }
        }

        None
    }
}

#[cfg(not(feature = "python"))]
impl LABCAT<Manual> {
    pub fn set_target_fn(self, f: fn(&Array2<f_>) -> Array1<f_>) -> LABCAT<Auto> {
        LABCAT {
            gp: self.gp,
            gp_state: self.gp_state,
            bounds: self.bounds,
            target_fn: Some(f),
            init_pts_fn: self.init_pts_fn,
            forget_fn: self.forget_fn,
            config: self.config,
            config_state: PhantomData,
        }
    }

    pub fn observations(&self) -> (Array2<f_>, Array1<f_>) {
        (self.gp.mem.X(), self.gp.mem.y())
    }

    pub fn thetas(&self) -> &Array1<f_> {
        self.gp.kernel.thetas()
    }

    pub fn suggest(&mut self) -> Array2<f_> {
        self._suggest()
    }

    pub fn observe(&mut self, X: Array2<f_>, y: Array1<f_>) {
        self._observe(X, y)
    }

    pub fn check_converged(&self) -> Option<TermCond> {
        self._check_converged()
    }
}

#[cfg(not(feature = "python"))]
impl LABCAT<Auto> {
    pub fn print_interval(mut self, interval: usize) -> Self {
        self.config.auto_print = Some(interval);
        self
    }

    pub fn run(mut self) -> OptimizationSummary {
        let print = self.config.auto_print.is_some();
        if print {
            println!("{}", self.title());
            println!("{}", self.mid_border());
        }
        loop {
            let suggest = self._suggest();
            let samples = (self.target_fn.unwrap())(&suggest);
            self._observe(suggest, samples);

            if print {
                if self.config.n_samples % self.config.auto_print.expect("Already checked option")
                    == 0
                {
                    println!("{}", self.iter_summary());
                }
            }

            if let Some(term) = self._check_converged() {
                if self.config.restarts == true {
                    match term {
                        TermCond::MaxItersReached => {
                            println!("{}", self.bottom_border());
                            return OptimizationSummary {
                                term_reason: term,
                                n_samples: self.config.n_samples,
                                min_x: self.gp.mem.X_min(),
                                min_y: self.gp.mem.y_min(),
                            };
                        }
                        TermCond::MaxTimeReached => {
                            println!("{}", self.bottom_border());
                            return OptimizationSummary {
                                term_reason: term,
                                n_samples: self.config.n_samples,
                                min_x: self.gp.mem.X_min(),
                                min_y: self.gp.mem.y_min(),
                            };
                        }
                        _ => self.restart(anyhow::format_err!("Restart after converge!")),
                    };
                } else {
                    if print {
                        println!("{}", self.bottom_border());
                    }

                    return OptimizationSummary {
                        term_reason: term,
                        n_samples: self.config.n_samples,
                        min_x: self.gp.mem.X_min(),
                        min_y: self.gp.mem.y_min(),
                    };
                }
            };
        }
    }
}
