use labcat::bound_types::{BoundRepr, BoundTrait, BoundType};
use labcat::bounds::{Bounds, Ready, BoundReprs};
use labcat::bounds_transforms::BoundTransform;
use labcat::{Auto, Config, Manual, LABCAT};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use std::fmt::Display;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

// fn matrix_to_numpy<'py, A, S, D>(py: Python<'py>, matrix: ArrayBase<S, D>) -> PyObject
// where
//     A: Scalar + Send + Sync + numpy::Element,
//     S: DataMut<Elem = A>,
//     D: Dimension,
// {
//     matrix.to_owned().into_pyarray(py).into()
// }

#[pyclass(name = "BoundsConfig")]
struct PyBoundsConfig {
    bounds: Bounds,
}

#[pymethods]
impl PyBoundsConfig {
    #[new]
    fn new() -> PyBoundsConfig {
        PyBoundsConfig {
            bounds: Bounds::new(),
        }
    }

    fn parse_config(&mut self, dict: &PyDict) {

        for (k, v) in dict.iter() {
            let key = k
                .extract::<String>()
                .expect("Could not parse key string in bounds config dict!");
            let val = v
                .extract::<&PyDict>()
                .expect("Could not value dict string in bounds config dict!");

            let bound_type = val
                .get_item("type")
                .expect(&format!(
                    "Type of bound not specified for bound \"{}\"!",
                    &key
                ))
                .extract::<String>()
                .expect(&format!(
                    "Type specification for bound \"{}\" could not be parsed to string!",
                    &key
                ));

            match bound_type.as_str() {
                "int" => self.parse_discrete(&key, &val),
                "real" => self.parse_continuous(&key, &val),
                "cat" => self.parse_categorical(&key, &val),
                "bool" => self.add_boolean(&key),
                &_ => panic!("Bound type \"{}\" is not supported!", bound_type),
            }
        }

    }

    fn add_categorical(&mut self, label: &str, categories: Vec<&str>) {
        self.bounds = self.bounds.clone().add_categorical(label, categories)
    }

    fn add_boolean(&mut self, label: &str) {
        self.bounds = self.bounds.clone().add_boolean(label);
    }

    fn add_discrete(&mut self, label: &str, upper: i64, lower: i64) {
        self.bounds = self
            .bounds
            .clone()
            .add_discrete(label, upper.into(), lower.into())
    }

    fn add_continuous(&mut self, label: &str, upper: f64, lower: f64) {
        self.bounds = self
            .bounds
            .clone()
            .add_continuous(label, upper.into(), lower.into())
    }

    fn add_discrete_with_transform(
        &mut self,
        label: &str,
        upper: i64,
        lower: i64,
        transform: &str,
    ) {
        let trans = BoundTransform::parse_transform(transform).expect(&format!(
            "Bound transform \"{}\" not recognized!",
            transform
        ));

        self.bounds = self.bounds.clone().add_discrete_with_transform(
            label,
            upper.into(),
            lower.into(),
            trans,
        )
    }

    fn add_continuous_with_transform(
        &mut self,
        label: &str,
        upper: f64,
        lower: f64,
        transform: &str,
    ) {
        let trans = BoundTransform::parse_transform(transform).expect(&format!(
            "Bound transform \"{}\" not recognized!",
            transform
        ));

        self.bounds = self.bounds.clone().add_continuous_with_transform(
            label,
            upper.into(),
            lower.into(),
            trans,
        )
    }

    fn parse_continuous(&mut self, label: &str, dict: &PyDict) {
        let space = dict.get_item("space");

        let transform = match space {
            Some(val) => {
                let trans_str = val.extract().expect(&format!(
                    "Type specification for bound \"{}\" could not be parsed to string!",
                    &label
                ));
                BoundTransform::parse_transform(trans_str).unwrap_or(BoundTransform::Linear)
            }
            None => BoundTransform::Linear,
        };

        let range = dict.get_item("range").expect(&format!(
            "Range specification for bound \"{}\" could not be found!",
            &label
        ));

        let range = range.extract::<(f64, f64)>().expect(&format!(
            "Range for bound \"{}\" could not be parsed to (f64, f64)!",
            &label
        ));

        self.add_continuous_with_transform(label, range.1, range.0, transform.to_string())
        
    }

    fn parse_discrete(&mut self, label: &str, dict: &PyDict) {
        let space = dict.get_item("space");

        let transform = match space {
            Some(val) => {
                let trans_str = val.extract().expect(&format!(
                    "Type specification for bound \"{}\" could not be parsed to string!",
                    &label
                ));
                BoundTransform::parse_transform(trans_str).unwrap_or(BoundTransform::Linear)
            }
            None => BoundTransform::Linear,
        };

        let range = dict.get_item("range").expect(&format!(
            "Range specification for bound \"{}\" could not be found!",
            &label
        ));

        let range = range.extract::<(i64, i64)>().expect(&format!(
            "Range for bound \"{}\" could not be parsed to (i64, i64)!",
            &label
        ));

        self.add_discrete_with_transform(label, range.1, range.0, transform.to_string())
        
    }

    fn parse_categorical(&mut self, label: &str, dict: &PyDict) {
        let categories = dict.get_item("categories").expect(&format!(
            "Categories for categorical bound \"{}\" could not be found!",
            &label
        ));

        let categories = categories.extract::<&PyList>().expect(&format!(
            "Categories for categorical bound \"{}\" could not be parsed to list!",
            &label
        ));

        let categories = categories.iter()
            .map(|item| 
                item.extract::<&str>().expect(&format!(
                    "Category in categorical bound \"{}\" could not be parsed to string!",
                    &label
                ))
            )
            .collect();

        self.add_categorical(label, categories);
        
    }

    pub fn build(&self) -> PyBounds {
        
        PyBounds {
            bounds: self.bounds.clone().build(),
        }
    }
}

#[pyclass(name = "Bounds")]
struct PyBounds {
    bounds: Bounds<Ready>,
}

#[pymethods]
impl PyBounds {
    
    #[new]
    fn new(dim: usize, upper: f64, lower: f64) -> PyBounds {
        PyBounds {
            bounds: Bounds::new_continuous(dim, upper, lower),
        }
    }
    
    fn dim(&self) -> usize {
        self.bounds.dim()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.bounds)
    }

    pub fn arr<'py>(&mut self, py: Python<'py>) -> PyObject {
        self.bounds.bounds_arr().bounds_arr().clone().into_pyarray(py).into()
    }

    fn repr<'py>(&self, x: &PyArray1<f64>, py: Python<'py>) -> Option<&'py PyList> {
        let x = unsafe { x.as_array() };

        let reprs = self.bounds.repr(x)?;
        let map = PyDict::new(py);
        for repr in reprs.iter() {
            match repr {
                BoundRepr::Continuous((str, val)) => map.set_item(str.clone(), *val).ok()?,
                BoundRepr::Discrete((str, val)) => map.set_item(str, *val).ok()?,
                BoundRepr::Categorical((str, cat)) => map.set_item(str, cat.clone()).ok()?,
                BoundRepr::Boolean((str, b)) => map.set_item(str, *b).ok()?,
            };
        }

        let list = PyList::empty(py);
        list.append(map).ok()?;

        Some(list)
    }

    fn parse<'py>(&self, x: &PyList, py: Python<'py>) -> Option<&'py PyList> {

        let py_list = PyList::empty(py);

        for x in x.iter() {
            
            let x_dict = x.extract::<&PyDict>().expect("Could not parse items in list to PyDict!");
            let mut repr_vec = vec![];

            for bound in self.bounds.iter_bounds() {
                let item = x_dict.get_item(bound.label()) //try to get boundrepr from python dict
                    .expect(&format!("Bound {} could not be found in python dict!", bound.label()));
                    
                let repr = match bound {
                    BoundType::Continuous(_) => {
                        let val = item.extract::<f64>()
                            .expect(&format!("Python dict key for bound {} could not be parsed to f64!", bound.label()));
                        BoundRepr::Continuous((bound.label().into(), val)) 
                    },
                    BoundType::Discrete(_) => {
                        let val = item.extract::<i64>().ok()
                            .expect(&format!("Python dict key for bound {} could not be parsed to i64!", bound.label()));
                        BoundRepr::Discrete((bound.label().into(), val)) 
                    },
                    BoundType::Categorical(_) => {
                        let cat = item.extract::<String>().ok()
                            .expect(&format!("Python dict key for bound {} could not be parsed to String!", bound.label()));
                        BoundRepr::Categorical((bound.label().into(), cat)) 
                    },
                    BoundType::Boolean(_) => {
                        let bool = item.extract::<bool>()
                            .expect(&format!("Python dict key for bound {} could not be parsed to bool!", bound.label()));
                        BoundRepr::Boolean((bound.label().into(), bool)) 
                    },
                };

                repr_vec.push(repr);
            }

            let reprs = BoundReprs::new(repr_vec);

            py_list.append(self.bounds.parse(reprs).into_pyarray(py)).ok()?;
        }

        Some(py_list)
    }
}

#[pyclass(name = "LABCATConfig")]
struct PyLABCATConfig {
    labcat: LABCAT<Config>,
}

#[pymethods]
impl PyLABCATConfig {
    #[new]
    fn new(bounds: &PyBounds) -> PyLABCATConfig {
        // let bounds = Bounds::new_continuous(2, 3.0, 2.0);
        PyLABCATConfig {
            labcat: LABCAT::new(bounds.bounds.clone()),
        }
    }

    fn beta(&mut self, beta: f64) {
        self.labcat.beta(beta);
    }

    fn prior_sigma(&mut self, prior_sigma: f64) {
        self.labcat.prior_sigma(prior_sigma);
    }

    pub fn init_fn(&mut self, f: PyObject) {
        self.labcat.init_pts_fn(f);
    }

    pub fn forget_fn(&mut self, f: PyObject) {
        self.labcat.forget_fn(f);
    }

    pub fn build<'py>(&mut self, py: Python<'py>) -> PyLABCATManual {
        PyLABCATManual {
            labcat: LABCAT::new_preconfigured(
                self.labcat.bounds().clone(),
                self.labcat.config().clone(),
                self.labcat.pyConfig().clone(),
                py
            ),
        }
    }
}

#[pyclass(name = "LABCATManual")]
struct PyLABCATManual {
    labcat: LABCAT<Manual>,
}

#[pymethods]
impl PyLABCATManual {
    pub fn deb(&self) {
        println!("{:?}", self.labcat.config())
    }

    pub fn observations<'py>(&mut self, py: Python<'py>) -> (PyObject, PyObject) {
        let (x, y) = self.labcat.observations();

        (x.into_pyarray(py).into(), y.into_pyarray(py).into())
    }

    pub fn thetas<'py>(&mut self, py: Python<'py>) -> PyObject {
        let thetas = self.labcat.thetas().to_owned();

        thetas.into_pyarray(py).into()
    }

    pub fn suggest<'py>(&mut self, py: Python<'py>) -> PyObject {
        self.labcat.suggest(py).into_pyarray(py).into()
    }

    pub fn observe(&mut self, x: &PyArray2<f64>, y: &PyArray1<f64>) {
        unsafe {
            self.labcat
                .observe(x.as_array().to_owned(), y.as_array().to_owned())
        }
    }

    pub fn predict<'py>(&mut self, x: &PyArray2<f64>, py: Python<'py>) -> PyObject {
        unsafe{
            self.labcat.predict(x.as_array().to_owned()).0.into_pyarray(py).into()
        }
    }

    pub fn set_target_fn<'py>(&mut self, f: PyObject, py: Python<'py>) -> PyLABCATAuto {
        PyLABCATAuto {
            labcat: LABCAT::new_preconfigured(
                self.labcat.bounds().clone(),
                self.labcat.config().clone(),
                self.labcat.pyConfig().clone(),
                py
            )
            .set_target_fn(f),
        }
    }
}

#[pyclass(name = "LABCATAuto")]
struct PyLABCATAuto {
    labcat: LABCAT<Auto>,
}

#[pymethods]
impl PyLABCATAuto {
    fn target_tol(&mut self, tol: f64) {
        self.labcat.target_tol(tol);
    }

    fn target_val(&mut self, val: f64) {
        self.labcat.target_val(val);
    }

    fn max_samples(&mut self, n: usize) {
        self.labcat.max_samples(n);
    }

    pub fn max_time(&mut self, seconds: u64) {
        self.labcat.max_time(std::time::Duration::new(seconds, 0));
    }
    
    pub fn print_interval(&mut self, interval: usize) {
        self.labcat.print_interval(interval);
    }

    pub fn run<'py>(&mut self, py: Python<'py>) -> OptimizationResult {
        let res = self.labcat.run(py);

        OptimizationResult {
            term_reason: res.reason(),
            n_samples: *res.n_samples(),
            min_x: res.min_x().to_owned().into_pyarray(py).into(),
            min_y: *res.min_y(),
        }
    }
}

#[pyclass(name = "LABCATResult")]
struct OptimizationResult {
    term_reason: String,
    n_samples: usize,
    min_x: PyObject,
    min_y: f64,
}

#[pymethods]
impl OptimizationResult {
    fn __str__(&self) -> String {
        format!("{}", self)
    }
}

impl Display for OptimizationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "--------------------",)?;
        writeln!(f, "{}", self.term_reason)?;
        writeln!(
            f,
            "iter: {} y_min: {:.6e}, x_min {:.3}",
            self.n_samples, self.min_y, self.min_x
        )?;
        writeln!(f, "--------------------",)?;
        Ok(())
    }
}

#[pymodule]
fn labcat(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyBoundsConfig>()?;
    m.add_class::<PyBounds>()?;
    m.add_class::<PyLABCATConfig>()?;
    m.add_class::<PyLABCATManual>()?;
    m.add_class::<PyLABCATAuto>()?;
    m.add_class::<OptimizationResult>()?;

    Ok(())
}
