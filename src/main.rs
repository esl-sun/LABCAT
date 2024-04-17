use faer::{Mat, Col, Row};
use faer_ext::{IntoFaer, IntoNdarray};

use labcat::bounds::ContinuousBounds;
use labcat::kernel::{BaseKernel, KernelSum};
use labcat::labcat::LabcatMemory;
use labcat::lhs::LHS;
use labcat::memory::{
    BaseMemory, ObservationIO, ObservationInputRecenter, ObservationInputRescale,
    ObservationInputRotate, ObservationOutputRecenter, ObservationOutputRescale,
    ObservationTransform,
};
use labcat::sqexp::SqExp;
use labcat::{AskTell, RefitWith, SurrogateIO};
use labcat::{ei::EI, gp::GP, kde::KDE, sqexp::SqExpARD, SMBO};
use labcat::utils::{MatMutUtils, MatRefUtils, MatUtils};

fn main() {
    let kern = SqExpARD::<f32>::new(5);
    let kern2 = SqExpARD::<f32>::new(5);
    let kern3 = SqExpARD::<f32>::new(5);
    let sum = kern.sum(kern2);
    let sum = sum.sum(kern3);
    let _ = sum;

    let kde = KDE::<f32, KernelSum<f32, SqExpARD<f32>, SqExpARD<f32>>>::new(5);
    dbg!(kde);

    let mem = BaseMemory::<f64>::default();
    dbg!(mem);
    let mem = BaseMemory::<f64>::new(5);
    dbg!(mem);

    let mut i = Mat::<f64>::identity(4, 4);
    // i.apply_fn(|(i, j), val| i as f64);

    // let k = i * j;
    // i.resize_with(new_nrows, new_ncols, f);
    // dbg!(&i);
    // dbg!(v.subcols(1, 2).get(1, 0));
    // v.subcols(0, 2);
    // v.zip_apply_with_row_slice();
    // v.fill_fn();
    // v.zip_apply_with_row_slice(&[10.0, 100.0, 1000.0], |old, new| old + new);
    dbg!(&i);
    // dbg!(&i.as_ref().row_as_slice(1));

    for item in i.indexed_iter() {
        dbg!(item);
    }

    for col in i.as_ref().cols() {
        dbg!(col);
    }

    for row in i.as_ref().rows() {
        dbg!(row);
    }

    dbg!(&i);
    dbg!(i.remove_cols(vec![1, 2]));
    dbg!(i.remove_rows(vec![0, 2]));
    // dbg!(i.remove_rows(vec![0,]));

    // let mut mu = i.get_mut(0, 0..i.ncols());

    // mu.fill(2.0);

    // dbg!(i.col_capacity());
    // dbg!(i);
    let c = Col::<f64>::zeros(5);
    let r = Row::<f64>::zeros(6);

    c.as_ref().as_ptr();

    dbg!(&c);
    dbg!(&r);

    let r2 = r.clone();
    dbg!(r2);

    let mut a = ndarray::Array2::<f64>::eye(4);
    dbg!(&a);
    dbg!(a.view().into_faer());
    for col in a.view_mut().into_faer().cols_mut() {
        dbg!(col[0]);
    }

    let l = faer::row::from_slice::<f64>(&[1.0, 2.0, 3.0, 50.0]);

    // faer_core::zipped!(
    //     i.as_mut().diagonal_mut().column_vector_mut(),
    //     l
    // )
    // .for_each(|faer_core::unzipped!(mut i, l)| {
    //     i.write(i.read() * l.read());
    // });

    i.rows_mut().for_each(|col| {
        faer::zipped!(col, l)
            .for_each(|faer::unzipped!(mut col, l)| col.write(col.read() + l.read()));
    });

    dbg!(i.as_ref().into_ndarray());

    // let s = &[0.0, 1.0, 2.0];

    // dbg!(s.faer_add(&[10.0, 100.0, 1000.0]));

    // let X = mem.X().into();
    // dbg!(test(X.into()));

    // let mut smbo = SMBO::<
    //     f64,
    //     ContinuousBounds,
    //     LHS<f64>,
    //     GP<f64, SqExpARD<f64>, BaseMemory<f64>>,
    //     EI<f64>,
    // >::new();

    // smbo.ask();

    let mut mem = LabcatMemory::<f64>::new(2);
    mem.append(&[1.0, 2.0], &1.0);
    mem.append(&[3.0, 4.0], &2.0);
    mem.append(&[5.0, 6.0], &3.0);
    mem.append(&[7.0, 8.0], &4.0);

    mem.recenter_X(&[1.0, 2.0]);
    mem.rescale_X(&[2.0, 4.0]);
    mem.rotate_X();

    dbg!(mem.X());
    dbg!(mem.X_prime());

    mem.recenter_Y(&1.0);
    mem.rescale_Y(&3.0);

    dbg!(mem.Y());
    dbg!(mem.Y_prime());

    let mut kde = KDE::<f64, SqExpARD<f64>>::new(2);
    kde.refit_from(&mem).unwrap();
    dbg!(kde.probe(&[0.0, 0.0]));
}
