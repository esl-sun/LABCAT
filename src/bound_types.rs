use std::fmt::Display;

use enum_dispatch::enum_dispatch;
use ndarray::{arr1, Array1, AssignElem};
use rand::{thread_rng, Rng};

use crate::{
    bounds_transforms::{BoundTransform, BoundTransformTrait, BoundTransformType},
    f_, i_,
};

#[enum_dispatch(BoundType)]
pub trait BoundTrait {
    fn label(&self) -> &str;
    fn inside(&self, x: &f_) -> bool;
    fn repr(&self, x: &f_) -> Option<BoundRepr>;
    fn parse(&self, x: &BoundRepr) -> f_;
    fn enum_var(&self) -> BoundType;
    fn bound_arr(&self) -> Array1<f_>;
}

#[derive(Debug, Clone)]
pub struct Continuous {
    label: String,
    upper: f_,
    lower: f_,
    transform: BoundTransformType,
}

impl Continuous {
    pub fn new(label: &str, upper: f_, lower: f_) -> Self {
        Continuous {
            label: label.into(),
            upper,
            lower,
            transform: BoundTransform::new_transform(BoundTransform::Linear),
        }
    }

    pub fn new_with_transform(
        label: &str,
        upper: f_,
        lower: f_,
        transform: BoundTransform,
    ) -> Self {
        let transform = BoundTransform::new_transform(transform);
        let upper = transform.transform(upper);
        let lower = transform.transform(lower);

        Continuous {
            label: label.into(),
            upper,
            lower,
            transform,
        }
    }
}

impl BoundTrait for Continuous {
    fn label(&self) -> &str {
        &self.label
    }

    fn inside(&self, x: &f_) -> bool {
        x <= &self.upper && x >= &self.lower
    }

    fn repr(&self, x: &f_) -> Option<BoundRepr> {
        if self.inside(x) {
            Some(BoundRepr::Continuous((
                self.label.clone(),
                self.transform.inv_transform(*x),
            )))
        } else {
            None
        }
    }

    fn parse(&self, x: &BoundRepr) -> f_ {
        match x {
            BoundRepr::Continuous((label, val)) => {
                if label != self.label() {
                    panic!(
                        "Label of bound representaion {} does not match label of bound {}!",
                        x.label(),
                        label
                    )
                };
                let parsed_val = self.transform.transform(*val);

                if !self.inside(&parsed_val) {
                    panic!(
                        "Parsed value for bound representation {} does not satisfy bound {}!",
                        x.label(),
                        label
                    )
                };

                parsed_val
            }
            _ => panic!(
                "Bound representation {} cannot be parsed to continuous bound type!",
                x
            ),
        }
    }

    fn enum_var(&self) -> BoundType {
        BoundType::Continuous(self.clone())
    }

    fn bound_arr(&self) -> Array1<f_> {
        arr1(&[self.lower, self.upper])
    }
}

#[derive(Debug, Clone)]
pub struct Discrete {
    label: String,
    upper: i_,
    lower: i_,
    transform: BoundTransformType,
}

impl Discrete {
    pub fn new(label: &str, upper: i_, lower: i_) -> Self {
        Discrete {
            label: label.into(),
            upper,
            lower,
            transform: BoundTransform::new_transform(BoundTransform::Linear),
        }
    }

    pub fn new_with_transform(
        label: &str,
        upper: i_,
        lower: i_,
        transform: BoundTransform,
    ) -> Self {
        match transform {
            BoundTransform::Logistic => {
                panic!("Logistic transformation not supported for discrete bound!")
            }
            _ => assert!(true),
        };

        Discrete {
            label: label.into(),
            upper,
            lower,
            transform: BoundTransform::new_transform(transform),
        }
    }
}

impl BoundTrait for Discrete {
    fn label(&self) -> &str {
        &self.label
    }

    fn inside(&self, x: &f_) -> bool {
        x <= &self.transform.transform(self.upper as f_)
            && x >= &self.transform.transform(self.lower as f_)
    }

    fn repr(&self, x: &f_) -> Option<BoundRepr> {
        if self.inside(x) {
            Some(BoundRepr::Discrete((
                self.label.clone(),
                self.transform.inv_transform(*x).floor() as i_,
            )))
        } else {
            None
        }
    }

    fn parse(&self, x: &BoundRepr) -> f_ {
        match x {
            BoundRepr::Discrete((label, val)) => {
                if label != self.label() {
                    panic!(
                        "Label of bound representaion {} does not match label of bound {}",
                        x.label(),
                        label
                    )
                };
                let parsed_val = self.transform.transform(*val as f_);

                if !self.inside(&parsed_val) {
                    panic!(
                        "Parsed value for bound representation {} does not satisfy bound {}!",
                        x.label(),
                        label
                    )
                };

                parsed_val
            }
            _ => panic!(
                "Bound representation {} cannot be parsed to discrete bound type!",
                x
            ),
        }
    }

    fn enum_var(&self) -> BoundType {
        BoundType::Discrete(self.clone())
    }

    fn bound_arr(&self) -> Array1<f_> {
        arr1(&[
            self.transform.transform(self.lower as f_),
            self.transform.transform(self.upper as f_),
        ])
    }
}

#[derive(Debug, Clone)]
pub struct Categorical {
    label: String,
    categories: Vec<String>,
}

impl Categorical {
    pub fn new(label: &str, categories: Vec<&str>) -> Self {
        if categories.len() == 0 {
            panic!("Number of categories in categorical bound must be non-zero!");
        }

        let categories = categories
            .iter()
            .map(|str| Into::<String>::into(*str))
            .collect();

        Categorical {
            label: label.into(),
            categories,
        }
    }
}

impl BoundTrait for Categorical {
    fn label(&self) -> &str {
        &self.label
    }

    fn inside(&self, x: &f_) -> bool {
        ((x.floor() as i_) <= (self.categories.len() as i_ - 1) && (x.floor() as i_) >= 0)
            || (x.ceil() as i_) == self.categories.len() as i_
    }

    fn repr(&self, x: &f_) -> Option<BoundRepr> {
        if self.inside(x) {
            let mut index = x.floor() as usize;
            if index == self.categories.len() {
                index.assign_elem(index - 1);
            }

            Some(BoundRepr::Categorical((
                self.label.clone(),
                self.categories[index].clone(),
            )))
        } else {
            None
        }
    }

    fn parse(&self, x: &BoundRepr) -> f_ {
        match x {
            BoundRepr::Categorical((label, key_cat)) => {
                if label != self.label() {
                    panic!(
                        "Label of bound representaion {} does not match label of bound {}",
                        x.label(),
                        label
                    )
                };
                let (index, _) = self
                    .categories
                    .iter()
                    .enumerate()
                    .find(|cat| cat.1 == key_cat)
                    .expect(&format!(
                        "Category {} could not be found in bound {} during parsing!",
                        key_cat,
                        self.label()
                    ));

                rand::thread_rng().gen::<f_>() + index as f_ // TODO: randomize in interval instead of fixed index?
            }
            _ => panic!(
                "Bound representation {} cannot be parsed to categorical bound type!",
                x
            ),
        }
    }

    fn enum_var(&self) -> BoundType {
        BoundType::Categorical(self.clone())
    }

    fn bound_arr(&self) -> Array1<f_> {
        arr1(&[0.0, self.categories.len() as f_])
    }
}

#[derive(Debug, Clone)]
pub struct Boolean {
    label: String,
}

impl Boolean {
    pub fn new(label: &str) -> Self {
        Boolean {
            label: label.into(),
        }
    }
}

impl BoundTrait for Boolean {
    fn label(&self) -> &str {
        &self.label
    }

    fn inside(&self, x: &f_) -> bool {
        ((x.floor() as i_) <= 1 && (x.floor() as i_) >= 0) || (x.ceil() as i_) == 2
    }

    fn repr(&self, x: &f_) -> Option<BoundRepr> {
        if self.inside(x) {
            let b = if x <= &1.0 { true } else { false };

            Some(BoundRepr::Boolean((self.label.clone(), b)))
        } else {
            None
        }
    }

    fn parse(&self, x: &BoundRepr) -> f_ {
        match x {
            BoundRepr::Boolean((label, bool)) => {
                if label != self.label() {
                    panic!(
                        "Label of bound representaion {} does not match label of bound {}",
                        x.label(),
                        label
                    )
                };
                match bool {
                    true => rand::thread_rng().gen(), //TODO: randomize in interval instead of fixed vals?
                    false => rand::thread_rng().gen::<f_>() + 1.0,
                }
            }
            _ => panic!(
                "Bound representation {} cannot be parsed to boolean bound type!",
                x
            ),
        }
    }

    fn enum_var(&self) -> BoundType {
        BoundType::Boolean(self.clone())
    }

    fn bound_arr(&self) -> Array1<f_> {
        arr1(&[0.0, 2.0])
    }
}

#[enum_dispatch]
#[derive(Debug, Clone)]
pub enum BoundType {
    Continuous,
    Discrete,
    Categorical,
    Boolean,
}

#[derive(Debug, Clone)]
pub enum BoundRepr {
    Continuous((String, f_)),
    Discrete((String, i_)),
    Categorical((String, String)),
    Boolean((String, bool)),
}

impl BoundRepr {
    pub fn label(&self) -> &str {
        match self {
            BoundRepr::Continuous((label, _)) => label,
            BoundRepr::Discrete((label, _)) => label,
            BoundRepr::Categorical((label, _)) => label,
            BoundRepr::Boolean((label, _)) => label,
        }
    }
}

impl Display for BoundRepr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BoundRepr::Continuous((label, val)) => write!(f, "{}: {}", label, val)?,
            BoundRepr::Discrete((label, int)) => write!(f, "{}: {}", label, int)?,
            BoundRepr::Categorical((label, cat)) => write!(f, "{}: {}", label, cat)?,
            BoundRepr::Boolean((label, b)) => write!(f, "{}: {}", label, b)?,
        };
        Ok(())
    }
}
