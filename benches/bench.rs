use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::DVector;
use ndarray::Array1;

use pikaso_lib::ToVector;

fn baseline_vec_convert(vec: DVector<f64>) -> Vec<f64> {
    vec.into_vec_test()
}

fn criterion_baseline_vec_convert(c: &mut Criterion) {
    let v = DVector::from_vec(vec![0.0; 50]);
    c.bench_function("baseline_vec_convert", |b| {
        b.iter(|| baseline_vec_convert(black_box(v.clone())))
    });
}

fn nalgebra_vec_convert(vec: DVector<f64>) -> Vec<f64> {
    vec.into_vec_test()
}

fn criterion_nalgebra_vec_convert(c: &mut Criterion) {
    let v = DVector::from_vec(vec![0.0; 50]);
    c.bench_function("nalgebra_vec_convert", |b| {
        b.iter(|| nalgebra_vec_convert(black_box(v.clone())))
    });
}

fn ndarray_vec_convert(vec: Array1<f64>) -> Vec<f64> {
    vec.into_vec_test()
}

fn criterion_ndarray_vec_convert(c: &mut Criterion) {
    let v = Array1::from_vec(vec![0.0; 50]);
    c.bench_function("ndarray_vec_convert", |b| {
        b.iter(|| ndarray_vec_convert(black_box(v.clone())))
    });
}

criterion_group!(
    benches,
    criterion_baseline_vec_convert,
    criterion_nalgebra_vec_convert,
    criterion_ndarray_vec_convert,
);
criterion_main!(benches);
