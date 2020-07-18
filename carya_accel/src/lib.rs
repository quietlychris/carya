#[cfg(test)]
#[macro_use]
extern crate serial_test;

pub mod cuda;
mod test_cuda;
use crate::cuda::*;
use accel::*;
