#[cfg(test)]
#[macro_use]
extern crate serial_test;

pub mod opencl;
mod test_opencl;
use crate::opencl::*;

use ndarray::prelude::*;
use ocl::Error;

pub mod prelude {
    #[cfg(feature = "cuda_through_accel")]
    pub use carya_accel::*;    

    pub use crate::opencl::*;
}
