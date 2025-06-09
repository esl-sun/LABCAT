use criterion::{criterion_group, criterion_main, Criterion};
use faer::{Mat, MatRef};

use std::hint::black_box;

fn dummy<T>(_: MatRef<'_, T>) {}

fn criterion_temp(c: &mut Criterion) {
    let m = Mat::<f64>::identity(10, 10);
    c.bench_function("baseline_vec_convert", |b| {
        b.iter(|| dummy(black_box(m.as_ref())))
    });
}

criterion_group!(benches, criterion_temp,);
criterion_main!(benches);
