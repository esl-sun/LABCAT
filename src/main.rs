use faer::{mat, Mat};
use faer_core::zip::{ViewMut, MaybeContiguous};
use faer_core::{Col, Row};
use labcat::kernel::{Kernel, KernelSum};
use labcat::lhs::LHS;
use labcat::memory::{BaseMemory, ObservationIO};
use labcat::{ei::EI, gp::GP, kde::KDE, sqexp::SqExpARD, SMBO};
use ndarray::s;
// use faer_core::ColIndex;
// use faer_core::Mat;

// use labcat::ToVector;
use labcat::utils::{MatMutUtils, MatRefUtils, MatUtils};

fn main() {
    let kern = SqExpARD::<f32>::new(5);
    let kern2 = SqExpARD::<f32>::new(5);
    let kern3 = SqExpARD::<f32>::new(5);
    let sum = kern.sum(kern2);
    let sum2 = sum.sum(kern3);

    let kde = KDE::<f32, KernelSum<f32, SqExpARD<f32>, SqExpARD<f32>>>::new(5);
    dbg!(kde);

    let mem = BaseMemory::<f64>::default();
    dbg!(mem);
    let mem = BaseMemory::<f64>::new(5);
    dbg!(mem);

    let mut i = Mat::<f64>::identity(2, 3);
    let mut j = Mat::<f64>::identity(3, 2);
    // let k = i * j;
    // i.resize_with(new_nrows, new_ncols, f);
    // dbg!(&i);
    let mut v = i.as_mut();
    // dbg!(v.subcols(1, 2).get(1, 0));
    // v.subcols(0, 2);
    // v.zip_apply_with_row_slice();
    // v.fill_fn();
    v.zip_apply_with_row_slice(&[10.0, 100.0, 1000.0], |old, new| old + new);
    dbg!(&i.as_ref().col_as_slice(1));

    for item in i.indexed_iter() {
        dbg!(item);
    }

    for col in i.as_ref().cols() {
        dbg!(col);
    }

    for row in i.as_ref().rows() {
        dbg!(row);
    }

    dbg!(i.remove_rows(vec![0,]));

    // let mut mu = i.get_mut(0, 0..i.ncols());

    // mu.fill(2.0);

    // dbg!(i.col_capacity());
    // dbg!(i);
    let c = Col::<f64>::zeros(5);
    let r = Row::<f64>::zeros(6);
    c.as_ref().as_ptr();

    dbg!(c);
    dbg!(&r);

    let r2 = r.clone();
    dbg!(r2);

    let a = ndarray::Array2::<f64>::eye(4);

    let s = &[0.0, 1.0, 2.0];

    // dbg!(s.faer_add(&[10.0, 100.0, 1000.0]));

    // let X = mem.X().into();
    // dbg!(test(X.into()));

    // let smbo = SMBO::<
    //     f64,
    //     LHS<f64>,
    //     BaseMemory<f64>,
    //     GP<f64, SqExpARD<f64>>,
    //     EI<'_, f64, GP<f64, SqExpARD<f64>>>,
    // >::new();
}
