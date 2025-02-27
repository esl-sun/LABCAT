use labcat::{LABCAT, bounds::Bounds, f_};
// use gplib::{ALGPA, GP, ALGPAtrait};
use ndarray::{Array1, Array2};

// Rosenbrock bjective function
pub fn rosenbrock_vec(x: Vec<f64>) -> f64 {
    let a = 1.0f64;
    let b = 100.0f64;

    let mut total = 0.0;
    for i in 0..(x.len() - 1) {
        total += b * (x[i + 1] - x[i].powi(2)).powi(2) + (a - x[i]).powi(2)
    }
    return total;
}

// Helper function to apply the Rosenbrock function to columns of the X matrix, required to run the LABCAT struct automatically.
#[allow(non_snake_case)]
fn true_fn(X: &Array2<f_>) -> Array1<f_> {
    let mut y = Array1::zeros((X.ncols(),));
    y.iter_mut()
        .zip(X.columns())
        .for_each(|(y, col)| *y = rosenbrock_vec(col.to_vec()));
    y
}

fn main() {
    // For more information on the Bounds struct, refer to the bounds.rs example
    let bounds = Bounds::new_continuous(2, 5.0, -5.0);

    // The LABCAT struct is constructed using the builder pattern. The LABCAT struct also uses the type-state pattern to indicate whether the struct is using an ask-tell interface or queries the objective function automatically.
    let mut ask_tell_alg = LABCAT::new(bounds.clone())
        // Default algorithm parameters
        // .restarts(true)
        // .beta(1.0 / bounds.dim())
        // .prior_sigma(0.1)
        // .init_pts_fn(|d| 2 * d + 1)
        // .forget_fn(|d| d * 7)
        .build();

    for i in 0..5 {
        let x = ask_tell_alg.suggest();
        let y = true_fn(&x);
        println!("Iter {}: X:\n{:.3}, \ny:{:.3}\n", i , &x, &y);
        ask_tell_alg.observe(x, y);
    }


    let auto_alg = LABCAT::new(bounds)
        .build()
        // Setting the objective function changes the struct from an ask-tell interface to a struct that automatically calls the objective functions as needed when run() is called
        .set_target_fn(true_fn);

    let res = auto_alg
        // Termination conditions can also be set for LABCAT<Auto>. Multiple condtions are allowed. Possible termination conditions:
        // Objective function sample budget
        .max_samples(150)
        // Target value of the output dynamic range (y_max - y_min)
        // .target_tol(1e-3)
        // Target output value of the objective function
        // .target_val(0.1)
        // Maximum wall-clock time of execution
        // .max_time(std::time::Duration::new(1, 0))

        // Interval to print run summary to terminal if set, does not print to terminal if not specified
        .print_interval(25)

        // Call run to execute the algorithm automatically
        .run();
    
    // Display summary of the result of the optimization run
    println!("{}", res);
}
