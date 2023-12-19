use nalgebra::Scalar;
use nalgebra::{DMatrix, DMatrixView, DVector, Dyn};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ShapeBuilder};

//////////////////////////

#[derive(Debug, Clone)]
pub struct Vector<T>
where
    T: Scalar,
{
    data: Vec<T>,
}

// impl<T> AsRef<T> for Vector<T>
// where
//     T: ?Sized + Scalar,
//     <Vector<T> as Deref>::Target: AsRef<T>,
// {
//     fn as_ref(&self) -> &T {
//         self.deref().as_ref()
//     }
// }

// impl<T> Deref for Vector<T>
// where
//     T: Scalar
// {
//     type Target = Vec<T>;

//     fn deref(&self) -> &Self::Target {
//         &self.data
//     }
// }

//////////////////////////
impl<T> From<Vector<T>> for Vec<T>
where
    T: Scalar,
{
    fn from(value: Vector<T>) -> Self {
        value.data
    }
}

impl<T> From<Vec<T>> for Vector<T>
where
    T: Scalar,
{
    fn from(value: Vec<T>) -> Self {
        Vector { data: value }
    }
}

//////////////////////////
impl<T> From<Vector<T>> for Array1<T>
where
    T: Scalar,
{
    fn from(value: Vector<T>) -> Self {
        Array1::from_vec(value.into())
    }
}

impl<T> From<Array1<T>> for Vector<T>
where
    T: Scalar,
{
    fn from(value: Array1<T>) -> Self {
        Vector {
            data: value.into_raw_vec(),
        }
    }
}

//////////////////////////
impl<T> From<Vector<T>> for DVector<T>
where
    T: Scalar,
{
    fn from(value: Vector<T>) -> Self {
        DVector::from_vec(value.into())
    }
}

impl<T> From<DVector<T>> for Vector<T>
where
    T: Scalar,
{
    fn from(value: DVector<T>) -> Self {
        Vector {
            data: value.data.into(),
        }
    }
}

///////////////////////////////

#[derive(Debug, Clone)]
pub struct VectorView<'a, T>
where
    T: Scalar,
{
    view: ArrayView1<'a, T>,
}

impl<'a, T> From<&'a Vec<T>> for VectorView<'a, T>
where
    T: Scalar,
{
    fn from(value: &'a Vec<T>) -> Self {
        unsafe {
            VectorView {
                view: ArrayView1::from_shape_ptr((value.len(),), value.as_ptr()),
            }
        }
    }
}

// impl<'a, T> From<VectorView<'a, T>> for &'a Vec<T>
// where
//     T: Scalar,
// {
//     fn from(value: VectorView<'a, T>) -> Self {
//         unsafe { value.view.as_slice().unwrap().into() }
//     }
// }

impl<'a, T> From<ArrayView1<'a, T>> for VectorView<'a, T>
where
    T: Scalar,
{
    fn from(value: ArrayView1<'a, T>) -> Self {
        VectorView { view: value }
    }
}

impl<'a, T> From<VectorView<'a, T>> for ArrayView1<'a, T>
where
    T: Scalar,
{
    fn from(value: VectorView<'a, T>) -> Self {
        value.view
    }
}

///////////////////////////////

#[derive(Debug, Clone)]
pub struct Matrix<T>
where
    T: Scalar,
{
    data: Array2<T>,
}

///////////////////////////////

impl<T> From<Matrix<T>> for Array2<T>
where
    T: Scalar,
{
    fn from(value: Matrix<T>) -> Self {
        value.data
    }
}

impl<T> From<Array2<T>> for Matrix<T>
where
    T: Scalar,
{
    fn from(value: Array2<T>) -> Self {
        Matrix { data: value }
    }
}

////////////////////////////////

impl<T> From<Matrix<T>> for DMatrix<T>
where
    T: Scalar,
{
    fn from(value: Matrix<T>) -> Self {
        let std_layout = value.data.is_standard_layout();
        let nrows = Dyn(value.data.nrows());
        let ncols = Dyn(value.data.ncols());
        let mut res = DMatrix::from_vec_generic(nrows, ncols, value.data.into_raw_vec());
        if std_layout {
            // This can be expensive, but we have no choice since nalgebra VecStorage is always
            // column-based.
            res.transpose_mut();
        }
        res
    }
}

impl<T> From<DMatrix<T>> for Matrix<T>
where
    T: Scalar,
{
    fn from(value: DMatrix<T>) -> Self {
        unsafe {
            Matrix {
                data: Array2::from_shape_vec_unchecked(
                    value.shape().strides(value.strides()),
                    value.data.into(),
                ),
            }
        }
    }
}

//////////////////////////////////////

#[derive(Debug, Clone)]
pub struct MatrixView<'a, T>
where
    T: Scalar,
{
    view: ArrayView2<'a, T>,
}

//////////////////////////////////////

impl<'a, T> From<ArrayView2<'a, T>> for MatrixView<'a, T>
where
    T: Scalar,
{
    fn from(value: ArrayView2<'a, T>) -> Self {
        MatrixView { view: value }
    }
}

impl<'a, T> From<MatrixView<'a, T>> for ArrayView2<'a, T>
where
    T: Scalar,
{
    fn from(value: MatrixView<'a, T>) -> Self {
        value.view
    }
}

//////////////////////////////////////

impl<'a, T> From<DMatrixView<'a, T, Dyn, Dyn>> for MatrixView<'a, T>
where
    T: Scalar,
{
    fn from(value: DMatrixView<'a, T, Dyn, Dyn>) -> Self {
        unsafe {
            let view =
                ArrayView2::from_shape_ptr(value.shape().strides(value.strides()), value.as_ptr());

            MatrixView { view }
        }
    }
}

impl<'a, T> From<MatrixView<'a, T>> for DMatrixView<'a, T, Dyn, Dyn>
where
    T: Scalar,
{
    fn from(value: MatrixView<'a, T>) -> Self {
        let nrows = Dyn(value.view.nrows());
        let ncols = Dyn(value.view.ncols());
        let ptr = value.view.as_ptr();
        let stride_row: usize =
            TryFrom::try_from(value.view.strides()[0]).expect("Negative row stride");
        let stride_col: usize =
            TryFrom::try_from(value.view.strides()[1]).expect("Negative column stride");
        let storage = unsafe {
            nalgebra::ViewStorage::from_raw_parts(
                ptr,
                (nrows, ncols),
                (Dyn(stride_row), Dyn(stride_col)),
            )
        };
        nalgebra::Matrix::from_data(storage)
    }
}
