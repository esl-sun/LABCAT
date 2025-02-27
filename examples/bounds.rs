use std::vec;

use labcat::bounds::{BoundTransform, Bounds};

fn main() {
    // New bounds are constructed using builder notation. This struct also uses the type state pattern to ensure that only fully configured Bounds can be used.
    let manual_bounds_cfg = Bounds::new()
        .add_continuous("d1", 5.0, 0.0)
        .add_continuous("d2", 1.0, -1.0);

    println!("{}", &manual_bounds_cfg);

    let manual_bounds = manual_bounds_cfg
        // Continuous and discrete bounds support Log, BiLog and Logistic transformations of the bounding values.
        .add_continuous_with_transform("d3", 0.95, 0.05, BoundTransform::Logistic)
        // Currently, LABCAT lacks explicit handling of categorical and boolean values. These bounds are currently discretized based on the number of valid states (i.e., 3 categories are mapped to the interval (0, 3)) cast to a continuous axis.
        .add_categorical("d4", vec!["c1", "c2", "c3"])
        .add_boolean("d5")
        // Use the build command to finalize the Bounds struct
        .build();

    println!("{}", &manual_bounds);
    let transformed_point = ndarray::array![1.0, 0.0, -2.0, 1.5, 0.5];
    println!(
        "{} -> {}",
        &manual_bounds.repr(transformed_point.view()).unwrap(),
        &transformed_point
    );

    assert!(manual_bounds.inside(ndarray::array![1.0, 0.0, 0.5, 1.0, 0.5].view()));
    assert!(!manual_bounds.inside(ndarray::array![-1.0, -10.0, 1.0, 4.0, 3.0].view()));

    // This constructor is also provided as a shortcut for building continuous Bounds with the same bounds for each dimension
    let auto_bounds = Bounds::new_continuous(3, 5.0, -5.0);

    println!("{}", &auto_bounds);
}
