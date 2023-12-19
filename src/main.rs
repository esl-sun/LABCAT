use nalgebra::{DMatrix, DMatrixView, DVector, Dyn, Vector3};
use ndarray::{Array1, Array2, ArrayView2};
use pikaso_lib::fallible::{Fallible, Fallible::{Fail, Success}};
use pikaso_lib::kernel::Kernel;
use pikaso_lib::lhs::{DoE, RandomSampling};
use pikaso_lib::memory::{BaseMemory, ObservationMemory};
use pikaso_lib::ndarray_utils::{Array2Utils, ArrayBaseUtils, ArrayView2Utils};
use pikaso_lib::utils::{Matrix, MatrixView};
use pikaso_lib::{ei::EI, gp::GP, lhs::LHS, sqexp::SqExpARD, utils::Vector, SMBO};

// use pikaso_lib::ToVector;

fn main() {
    let i = 3_f32.exp();
    dbg!(i);
    let i = 3_f32.exp() as i32;
    dbg!(i);
    // let v = DVector::from_vec(vec![0.0, 1.0, 2.0]);
    // let v = Vector3::from_vec(vec![0.0, 1.0, 2.0]);
    // dbg!(v.shape());
    // dbg!(v);

    // let v = Array1::from_vec(vec![6.0, 7.0, 8.0]);
    // dbg!(v.into_vec_test());
    // println!("Hello, world!");
    // let mut a = ndarray::arr2(&[[0.0_f32, 1.0], [5.0, 10.0] , [-5.0, -10.0]]);
    let mut a = Array2::<f32>::eye(5);
    // a.view_mut().remove_index(axis, index);
    dbg!(a.indexed_max());
    let x: MatrixView<f32> = a.view().into();
    // test(x.into());
    // let view = a.view();
    // dbg!(a);
    // dbg!(view.product_trace(a.view()));

    // let v: Vector<f64> = vec![0.0, 1.0, 2.0].into();
    // let r: &Vec<f64> = &v;

    // let gp = GP::<f64, SqExpARD<f64>>::new();
    // let lhs = LHS::default();
    // dbg!(lhs.build_DoE(10, a.clone().into()).into());

    // let rand = RandomSampling::default();
    // dbg!(rand.build_DoE(10, a.into()).into());

    let kern = SqExpARD::<f32>::new(5);

    let mem = BaseMemory::<f64>::default();

    let X = mem.X().into();
    dbg!(test(X.into()));

    let smbo = SMBO::<
        f64,
        LHS<f64>,
        BaseMemory<f64>,
        GP<f64, SqExpARD<f64>>,
        EI<'_, f64, GP<f64, SqExpARD<f64>>>,
    >::new();
}

fn test(a: DMatrix<f64>) -> Fallible<f64>{
    dbg!("1");
    test_fallible()?;
    dbg!("2");
    Success
}

fn test_fallible() -> Fallible<f64> {
    // Fail(42.0)
    Success
}