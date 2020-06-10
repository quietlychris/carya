use crate::cuda::*;

use accel::*;
use ndarray::prelude::*;

#[test]
#[serial]
fn vec_squared() -> error::Result<()> {
    let backend = BackEnd::new()?;
    let CudaArray = CudaArray::new(&backend, 3, 4);

    let mut a = CudaArray::from_vec(&backend, 1, 10, vec![2.; 10]);
    println!("a to start:\n{:?}", a.v.as_slice());
    &a.square();
    println!("a after squaring:\n{:?}", a.v.as_slice());

    Ok(())
}
