#[cfg(test)]
#[macro_use]
extern crate serial_test;

pub mod cuda;
mod test_cuda;
mod test_opencl;
use crate::cuda::*;
pub mod opencl;
use crate::opencl::*;

use accel::*;
use ndarray::prelude::*;
use ocl::Error;

pub mod prelude {
    pub use crate::cuda::*;
    pub use crate::opencl::*;
}
