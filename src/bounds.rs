use std::{fmt::Display, marker::PhantomData, ops::Deref, slice::Iter};

use ndarray::{Array1, ArrayView1};

use crate::{
    bound_types::{Boolean, BoundRepr, BoundTrait, BoundType, Categorical, Continuous, Discrete},
    bounds_array::ArrayBounds,
    bounds_transforms::BoundTransform,
    f_, i_,
};

trait BoundsConfig {}

#[derive(Debug, Clone)]
pub struct Config {}
#[derive(Debug, Clone)]
pub struct Ready {}

impl BoundsConfig for Config {}
impl BoundsConfig for Ready {}

#[derive(Debug, Clone)]
pub struct Bounds<BoundsConfig = Config> {
    bounds: Vec<BoundType>,
    bounds_arr: ArrayBounds,
    config_state: PhantomData<BoundsConfig>,
}

impl Default for Bounds {
    fn default() -> Self {
        Self::new()
    }
}

impl Bounds {
    pub fn new() -> Bounds<Config> {
        let bounds: Vec<BoundType> = vec![];

        let bounds_arr = ArrayBounds::new(bounds.clone());

        Bounds {
            bounds,
            bounds_arr,
            config_state: PhantomData,
        }
    }

    pub fn new_continuous(d: usize, upper: f_, lower: f_) -> Bounds<Ready> {
        if d == 0 {
            panic!("Dimension of bounds must be non-zero!")
        }

        let bounds: Vec<BoundType> = (0..d)
            .map(|i| Continuous::new(&format!("d{}", i + 1), upper, lower).enum_var())
            .collect();

        let bounds_arr = ArrayBounds::new(bounds.clone());

        Bounds {
            bounds,
            bounds_arr,
            config_state: PhantomData,
        }
    }
}

impl Bounds<Config> {
    fn push_bound(&mut self, bound: BoundType) {
        if self.bounds.iter().any(|b| b.label() == bound.label()) {
            panic!(
                "Cannot have bounds with duplicate \"{}\" labels!",
                bound.label()
            )
        }

        self.bounds.push(bound);
        self.bounds_arr = ArrayBounds::new(self.bounds.clone());
    }

    pub fn add_categorical(mut self, label: &str, categories: Vec<&str>) -> Bounds<Config> {
        if categories.is_empty() {
            panic!("Amount of categories in bound must be non-zero!")
        }

        let mut dup_check = categories.clone();
        dup_check.sort();
        dup_check.dedup();
        if dup_check.len() != categories.len() {
            panic!("Categories must be unique!")
        }

        self.push_bound(Categorical::new(label, categories).enum_var());
        self
    }

    pub fn add_boolean(mut self, label: &str) -> Bounds<Config> {
        self.push_bound(Boolean::new(label).enum_var());
        self
    }

    pub fn add_discrete(mut self, label: &str, upper: i_, lower: i_) -> Bounds<Config> {
        if upper <= lower {
            panic!("Upper value for discrete bound cannot be <= lower value! ")
        }

        self.push_bound(Discrete::new(label, upper, lower).enum_var());
        self
    }

    pub fn add_discrete_with_transform(
        mut self,
        label: &str,
        upper: i_,
        lower: i_,
        transform: BoundTransform,
    ) -> Bounds<Config> {
        if upper <= lower {
            panic!("Upper value for discrete bound cannot be <= lower value! ")
        }

        self.push_bound(Discrete::new_with_transform(label, upper, lower, transform).enum_var());
        self
    }

    pub fn add_continuous(mut self, label: &str, upper: f_, lower: f_) -> Bounds<Config> {
        if upper <= lower {
            panic!("Upper value for continuous bound cannot be <= lower value! ")
        }

        self.push_bound(Continuous::new(label, upper, lower).enum_var());
        self
    }

    pub fn add_continuous_with_transform(
        mut self,
        label: &str,
        upper: f_,
        lower: f_,
        transform: BoundTransform,
    ) -> Bounds<Config> {
        if upper <= lower {
            panic!("Upper value for continuous bound cannot be <= lower value! ")
        }

        self.push_bound(Continuous::new_with_transform(label, upper, lower, transform).enum_var());
        self
    }

    pub fn build(self) -> Bounds<Ready> {
        Bounds {
            bounds: self.bounds,
            bounds_arr: self.bounds_arr,
            config_state: PhantomData,
        }
    }
}

impl Bounds<Ready> {
    pub fn dim(&self) -> usize {
        self.bounds.len()
    }

    pub fn iter_bounds(&self) -> Iter<BoundType> {
        self.bounds.iter()
    }

    pub fn inside(&self, x: ArrayView1<f_>) -> bool {
        if x.len() != self.bounds.len() {
            panic!("Input point dim does not match bounds dim!")
        };

        self.bounds
            .iter()
            .zip(x.iter())
            .all(|(bound, x)| bound.inside(x))
    }

    pub fn repr(&self, x: ArrayView1<f_>) -> Option<BoundReprs> {
        if x.len() != self.bounds.len() {
            panic!("Input point dim does not match bounds dim!")
        };

        let reprs: Option<Vec<BoundRepr>> = self
            .bounds
            .iter()
            .zip(x.iter())
            .map(|(bound, x)| bound.repr(x))
            .collect();

        reprs.map(|reprs| BoundReprs { reprs })
    }

    pub fn parse(&self, x: BoundReprs) -> Array1<f_> {
        let mut res_vec = vec![];

        for bound in self.bounds.iter() {
            // iter through self bounds
            let x_match = x
                .iter() // try to find matching boundrepr in x
                .find(|&bound_repr| bound_repr.label() == bound.label())
                .unwrap_or_else(|| {
                    panic!("Bound {} could not be found during parsing!", bound.label())
                });
            res_vec.push(bound.parse(x_match)); // parse matching bound_repr and push into res
        }

        Array1::from_vec(res_vec)
    }

    pub fn bounds_arr(&self) -> &ArrayBounds {
        &self.bounds_arr
    }
}

#[derive(Debug, Clone)]
pub struct BoundReprs {
    reprs: Vec<BoundRepr>,
}

impl BoundReprs {
    pub fn new(reprs: Vec<BoundRepr>) -> BoundReprs {
        BoundReprs { reprs }
    }

    pub fn iter(&self) -> Iter<BoundRepr> {
        self.reprs.iter()
    }
}

// impl Deref for BoundReprs {
//     type Target = Vec<BoundRepr>;

//     fn deref(&self) -> &Self::Target {
//         &self.reprs
//     }
// }

impl Display for BoundReprs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{ ")?;

        self.reprs
            .iter()
            .try_fold((), |_, repr| write!(f, "{}, ", repr))?;

        write!(f, "}}")?;

        Ok(())
    }
}
