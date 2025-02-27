use std::ops::Neg;

use enum_dispatch::enum_dispatch;

use crate::f_;

pub enum BoundTransform {
    Linear,
    Log,
    Logistic,
    BiLog,
}

impl BoundTransform {
    pub fn new_transform(self) -> BoundTransformType {
        match self {
            BoundTransform::Linear => BoundTransformType::Linear(Linear {}),
            BoundTransform::Log => BoundTransformType::Log(Log {}),
            BoundTransform::Logistic => BoundTransformType::Logistic(Logistic {}),
            BoundTransform::BiLog => BoundTransformType::BiLog(BiLog {}),
        }
    }

    pub fn parse_transform(data: &str) -> Option<BoundTransform> {
        match data {
            "linear" => Some(BoundTransform::Linear),
            "log" => Some(BoundTransform::Log),
            "bilog" => Some(BoundTransform::BiLog),
            "logistic" | "logit" => Some(BoundTransform::Logistic),
            &_ => None,
        }
    }

    pub fn to_string(&self) -> &str {
        match self {
            BoundTransform::Linear => "linear",
            BoundTransform::Log => "log",
            BoundTransform::Logistic => "logistic",
            BoundTransform::BiLog => "bilog",
        }
    }
}

#[enum_dispatch(BoundTransformType)]
pub trait BoundTransformTrait {
    fn transform(&self, x: f_) -> f_;
    fn inv_transform(&self, x: f_) -> f_;
}

#[enum_dispatch]
#[derive(Debug, Clone)]
pub enum BoundTransformType {
    Linear,
    Log,
    Logistic,
    BiLog,
}

impl std::fmt::Display for BoundTransformType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BoundTransformType::Linear(_) => write!(f, "None"),
            BoundTransformType::Log(_) => write!(f, "Log"),
            BoundTransformType::Logistic(_) => write!(f, "Logistic"),
            BoundTransformType::BiLog(_) => write!(f, "BiLog"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Linear {}

impl BoundTransformTrait for Linear {
    fn transform(&self, x: f_) -> f_ {
        x
    }

    fn inv_transform(&self, x: f_) -> f_ {
        x
    }
}

#[derive(Debug, Clone)]
pub struct Log {}

impl BoundTransformTrait for Log {
    fn transform(&self, x: f_) -> f_ {
        if x <= 0.0 {
            panic!("Bound cannot be set to <= 0 for bound with log transformation!");
        }
        x.ln()
    }

    fn inv_transform(&self, x: f_) -> f_ {
        x.exp()
    }
}

#[derive(Debug, Clone)]
pub struct BiLog {}

impl BoundTransformTrait for BiLog {
    fn transform(&self, x: f_) -> f_ {
        x.signum() * (x.abs() + 1.0).ln()
    }

    fn inv_transform(&self, x: f_) -> f_ {
        x.signum() * (x.abs() - 1.0).exp()
    }
}

#[derive(Debug, Clone)]
pub struct Logistic {}

impl BoundTransformTrait for Logistic {
    fn transform(&self, x: f_) -> f_ {
        if x >= 1.0 || x <= 0.0 {
            panic!("Bound must be set between 1 and 0 for bound with logistic transformation!");
        }

        (x / (1.0 - x)).ln()
    }

    fn inv_transform(&self, x: f_) -> f_ {
        1.0 / (1.0 + x.neg().exp())
    }
}
