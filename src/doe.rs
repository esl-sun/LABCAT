use faer::MatRef;

use crate::{bounds::UpperLowerBounds, dtype, memory::ObservationIO};

pub trait DoE<T>: Default
where
    T: dtype,
{
    fn build_DoE<B: UpperLowerBounds<T>>(&mut self, n: usize, bounds: &B);
    fn DoE(&self) -> MatRef<T>;
    fn n(&self) -> usize {
        self.DoE().ncols()
    }
    fn i(&self, id: usize) -> &[T] {
        let ptr = self.DoE().ptr_at(0, id);
        unsafe { core::slice::from_raw_parts(ptr, self.DoE().nrows()) } // Safety: i and n should always be in bounds, DoE stays in memory
    }
    fn get(&self, current_n: usize) -> Option<&[T]> {
        match current_n < self.n() {
            true => Some(self.i(current_n)),
            false => None,
        }
    }
}
