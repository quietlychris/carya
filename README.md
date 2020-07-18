### carya

A sketch of a GPU-hosted two-dimensional linear algebra library in Rust.

Tested on Pop!_OS 20.04, Intel i7-6700HQ + NVIDIA GeForce GTX 960M

At the moment, a very specific nightly toolchain is required for the `accel` crate dependency. Directions for that can be found [here](https://gitlab.com/termoshtt/accel), which has a script for installing the specific compiler version. 
