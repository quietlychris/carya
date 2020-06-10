use ndarray::prelude::*;

use std::iter::FromIterator;
use std::time::{Duration, Instant};

// Note: From benchmarking, the highest contribution to the runtime of this function is the conversion from an Array2 struct into a vector. In the context of a dense neural network, it's probably possible to do all of that overhead at the beginning, then keep exchanging the already-built vectors back and forth.
use ocl::enums::DeviceSpecifier::*;
use ocl::error::Error;
use ocl::{Buffer, Device, MemFlags, Platform, ProQue, SpatialDims::*};

#[derive(Debug,Clone)]
pub struct OpenCLArray {
    pub backend: CLBackEnd,
    pub v: Buffer<f32>,
    pub rows: usize,
    pub cols: usize,
}

pub fn create_vec(arr: &Array2<f32>) -> Vec<f32> {
    Array::from_iter(arr.iter().cloned()).to_vec()
}

impl OpenCLArray {
    pub fn new(backend: CLBackEnd, rows: usize, cols: usize) -> Result<Self, Error> {
        let a = vec![0.; rows * cols];
        let buffer = Buffer::builder()
            .queue(backend.proque.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(Two(rows, cols))
            .copy_host_slice(&a)
            .build()?;
        Ok(OpenCLArray {
            backend: backend,
            v: buffer,
            rows: rows,
            cols: cols,
        })
    }

    pub fn from_vec(
        backend: CLBackEnd,
        rows: usize,
        cols: usize,
        v: Vec<f32>,
    ) -> Result<OpenCLArray, Error> {
        assert_eq!(v.len(), rows * cols);
        let buffer = Buffer::builder()
            .queue(backend.proque.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(Two(rows, cols))
            .copy_host_slice(&v)
            .build()?;
        let mut arr = OpenCLArray {
            backend: backend,
            v: buffer,
            rows: rows,
            cols: cols,
        };
        Ok(arr)
    }

    pub fn from_array(backend: CLBackEnd, array: &Array2<f32>) -> Result<OpenCLArray, Error> {
        let v = create_vec(array);
        let (rows, cols) = (array.nrows(), array.ncols());
        assert_eq!(v.len(), rows * cols);
        let buffer = Buffer::builder()
            .queue(backend.proque.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(Two(rows, cols))
            .copy_host_slice(&v)
            .build()?;
        let mut arr = OpenCLArray {
            backend: backend,
            v: buffer,
            rows: rows,
            cols: cols,
        };
        Ok(arr)
    }

    pub fn to_vec(self) -> Result<Vec<f32>, Error> {
        let mut vec_result = vec![0.; self.rows * self.cols];
        self.v.read(&mut vec_result).enq()?;
        Ok(vec_result)
    }

    pub fn to_array(self) -> Result<Array2<f32>, Error> {
        let mut vec_result = vec![0.; self.rows * self.cols];
        self.v.read(&mut vec_result).enq()?;

        // println!("vec_result: {:?}",vec_result);
        let arr = Array::from_shape_vec((self.rows, self.cols), vec_result)
            .expect("Coudn't convert result to properly sized array");
        Ok(arr)
    }

    pub fn square(&mut self) -> Result<(), Error> {
        let mut kern = self
            .backend
            .proque
            .kernel_builder("square")
            .arg(&self.v)
            .build()?;

        kern.set_default_global_work_size(One(self.rows * self.cols)); // This one alone works for MNIST-size sets

        unsafe {
            kern.enq()?;
        }

        Ok(())
    }

    pub fn t(&mut self) -> Result<OpenCLArray, Error> {
        let v = vec![0.; self.rows * self.cols];
        let buffer = Buffer::builder()
            .queue(self.backend.proque.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(Two(self.cols, self.rows))
            .copy_host_slice(&v)
            .build()?;

        let mut kern = self
            .backend
            .proque
            .kernel_builder("transpose")
            .arg(&self.v)
            .arg(&buffer)
            .arg(self.rows)
            .arg(self.cols)
            .build()?;

        kern.set_default_global_work_size(Two(self.rows, self.cols));

        unsafe {
            kern.enq()?;
        }

        let mut result_ocl_array = OpenCLArray::new(self.backend.clone(), self.cols, self.rows)?;
        result_ocl_array.v = buffer;
        Ok(result_ocl_array)
    }

    pub fn dot(&self, b: &OpenCLArray) -> Result<OpenCLArray, Error> {
        let (n, m, k) = (self.rows, self.cols, b.cols);

        let mut result = OpenCLArray::new(self.backend.clone(), n, k)?;
        result.backend.proque.set_dims([n, k]);
        let v = vec![0.; n * k];

        let mut c = Buffer::builder()
            .queue(result.backend.proque.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(Two(n, k))
            .copy_host_slice(&v)
            .build()?;

        let mut kern = self
            .backend
            .proque
            .kernel_builder("dot_product")
            .arg(&self.v)
            .arg(&b.v)
            .arg(&c)
            .arg(&m)
            .arg(&k)
            .build()?;

        kern.set_default_global_work_size(Two(n, k)); // This one alone works for MNIST-size sets

        unsafe {
            kern.enq()?;
        }

        result.v = c;

        Ok(result)
    }

    pub fn hadamard(&self, b: &OpenCLArray) -> Result<OpenCLArray, Error> {
        assert_eq!((self.rows, self.cols), (b.rows, b.cols));
        let (n, m) = (self.rows, self.cols);

        let mut result = OpenCLArray::new(self.backend.clone(), n, m)?;
        result.backend.proque.set_dims([n, m]);
        let v = vec![0.; n * m];

        let mut c = Buffer::builder()
            .queue(result.backend.proque.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(One(n * m))
            .copy_host_slice(&v)
            .build()?;

        let mut kern = self
            .backend
            .proque
            .kernel_builder("hadamard")
            .arg(&self.v)
            .arg(&b.v)
            .arg(&c)
            .build()?;

        kern.set_default_global_work_size(One(n * m)); // This one alone works for MNIST-size sets

        unsafe {
            kern.enq()?;
        }

        result.v = c;

        Ok(result)
    }

    pub fn add(&self, b: &OpenCLArray) -> Result<OpenCLArray, Error> {
        assert_eq!((self.rows, self.cols), (b.rows, b.cols));
        let (n, m) = (self.rows, self.cols);

        let mut result = OpenCLArray::new(self.backend.clone(), n, m)?;
        result.backend.proque.set_dims([n, m]);
        let v = vec![0.; n * m];

        let mut c = Buffer::builder()
            .queue(result.backend.proque.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(One(n * m))
            .copy_host_slice(&v)
            .build()?;

        let mut kern = self
            .backend
            .proque
            .kernel_builder("add")
            .arg(&self.v)
            .arg(&b.v)
            .arg(&c)
            .build()?;

        kern.set_default_global_work_size(One(n * m)); // This one alone works for MNIST-size sets

        unsafe {
            kern.enq()?;
        }

        result.v = c;

        Ok(result)
    }

    pub fn subtract(&self, b: &OpenCLArray) -> Result<OpenCLArray, Error> {
        assert_eq!((self.rows, self.cols), (b.rows, b.cols));
        let (n, m) = (self.rows, self.cols);

        let mut result = OpenCLArray::new(self.backend.clone(), n, m)?;
        result.backend.proque.set_dims([n, m]);
        let v = vec![0.; n * m];

        let mut c = Buffer::builder()
            .queue(result.backend.proque.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(One(n * m))
            .copy_host_slice(&v)
            .build()?;

        let mut kern = self
            .backend
            .proque
            .kernel_builder("subtract")
            .arg(&self.v)
            .arg(&b.v)
            .arg(&c)
            .build()?;

        kern.set_default_global_work_size(One(n * m)); // This one alone works for MNIST-size sets

        unsafe {
            kern.enq()?;
        }

        result.v = c;

        Ok(result)
    }

    pub fn scalar_multiply(&self, coeff: f32) -> Result<OpenCLArray, Error> {
        let (n, m) = (self.rows, self.cols);

        let mut result = OpenCLArray::new(self.backend.clone(), n, m)?;
        result.backend.proque.set_dims([n, m]);
        let v = vec![0.; n * m];

        let mut b = Buffer::builder()
            .queue(result.backend.proque.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(One(n * m))
            .copy_host_slice(&v)
            .build()?;

        let mut kern = self
            .backend
            .proque
            .kernel_builder("multiply_by_scalar")
            .arg(&self.v)
            .arg(&b)
            .arg(&coeff)
            .build()?;

        kern.set_default_global_work_size(One(n * m)); // This one alone works for MNIST-size sets

        unsafe {
            kern.enq()?;
        }

        result.v = b;

        Ok(result)
    }

    pub fn sigmoid(&self) -> Result<OpenCLArray, Error> {
        let (n, m) = (self.rows, self.cols);

        let mut result = OpenCLArray::new(self.backend.clone(), n, m)?;
        result.backend.proque.set_dims([n, m]);
        let v = vec![0.; n * m];

        let mut b = Buffer::builder()
            .queue(result.backend.proque.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(One(n * m))
            .copy_host_slice(&v)
            .build()?;

        let mut kern = self
            .backend
            .proque
            .kernel_builder("sigmoid")
            .arg(&self.v)
            .arg(&b)
            .build()?;

        kern.set_default_global_work_size(One(n * m)); // This one alone works for MNIST-size sets

        unsafe {
            kern.enq()?;
        }

        result.v = b;

        Ok(result)
    }
}

#[derive(Debug, Clone)]
pub struct CLBackEnd {
    pub proque: ProQue,
}

impl CLBackEnd {
    pub fn new(gpu_type: &str) -> ocl::Result<Self> {
        let proque = build_ocl_proque(gpu_type.to_string())?;
        Ok(CLBackEnd { proque: proque })
    }
}

pub fn build_ocl_proque(gpu_type: String) -> ocl::Result<ProQue> {
    let src = include_str!("cl/functions.cl");

    let mut dev = None;
    let platforms = Platform::list();
    for p_idx in 0..platforms.len() {
        let platform = &platforms[p_idx];
        let devices = Device::list_all(platform)?;
        for d_idx in 0..devices.len() {
            let device = devices[d_idx];
            println!("Device: {:?}", device.name());
            if device.name()?.to_string().contains(&gpu_type) {
                dev = Some(device);
            }
            //let deviceinforesult = core::get_device_info(&device, DeviceInfo::MaxComputeUnits);
            //let units = deviceinforesult.to_string().parse().unwrap();
        }
    }

    println!("The desired GPU device is: {:?}", dev);
    //println!("The WORK_SIZE is {}",WORK_SIZE);
    let mut ocl_pq = ProQue::builder().src(src).device(dev.unwrap()).build()?;

    println!("The specified device is: {}", ocl_pq.device().name()?);
    println!(
        "It has a maximum working group size of {}",
        ocl_pq.device().max_wg_size()?
    );
    assert!(ocl_pq.device().is_available().unwrap());
    Ok(ocl_pq)
}
