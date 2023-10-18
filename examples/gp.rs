use algpa_lib::{bounds::Bounds, f_, LABCAT};
// use gplib::{ALGPA, GP, ALGPAtrait};
use ndarray::{Array1, Array2,};
#[allow(dead_code, non_snake_case)]
fn true_fn(X: &Array2<f_>) -> Array1<f_> {

    let mut y = Array1::zeros((X.ncols(),));
    y.iter_mut()
        .zip(X.columns())
        .for_each(|(y, col)| *y = rosenbrock_vec(col.to_vec()));
    y
}

pub fn rosenbrock_vec(x: Vec<f64>) -> f64 {
    let a = 1.0f64;
    let b = 100.0f64;

    let mut total = 0.0;
    for i in 0..(x.len() - 1) {
        total += b * (x[i + 1] - x[i].powi(2)).powi(2) + (a - x[i]).powi(2)
    }
    return total;
}

#[allow(dead_code, non_snake_case)]
fn main() {
    
    // let b = Bounds::new()
    //     .add_categorical("c1", vec!["cat1", "cat2"])
    //     .add_boolean("c2")
    //     .add_continuous_with_transform(
    //         "c3",
    //         0.99,
    //         0.01,
    //         gplib::bounds_transforms::BoundTransform::Logistic,
    //     )
    //     .build();


    let bounds = Bounds::new_continuous(3, 5.0, -5.0);
    let alg = LABCAT::new(bounds)
        .beta(0.5)
        .max_samples(500)
        // .restarts(true)
        // .target_tol(1e-6)
        .prior_sigma(0.1)
        // .target_val(0.1)
        .init_pts_fn(|d| 2 * d + 1)
        .forget_fn(|d| d * 7)
        .build();

    let res = alg.set_target_fn(true_fn).print_interval(25).run();
    println!("{}", res);
}
