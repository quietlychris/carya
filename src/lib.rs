#[cfg(test)]
#[macro_use]
extern crate serial_test;

mod cuda;
mod test_cuda;
mod test_opencl;
use crate::cuda::*;
mod opencl;
use crate::opencl::*;

use accel::*;
use ndarray::prelude::*;
use ocl::Error;
