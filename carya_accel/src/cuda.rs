use accel::error::AccelError;
use accel::*;
use std::sync::Arc;

#[kernel]
unsafe fn add(a: *const f32, b: *const f32, c: *mut f32, n: usize) {
    let i = accel_core::index();
    let j = accel_core::index();

    if (i as usize) < n {
        *c.offset(i) = *a.offset(i) + *b.offset(i);
    }
}

#[kernel]
unsafe fn square(a: *mut f32, n: usize) {
    let i = accel_core::index();

    if (i as usize) < n {
        *a.offset(i) = *a.offset(i) * *a.offset(i);
    }
}

pub struct CudaArray {
    pub ctx: Arc<Context>,
    pub v: DeviceMemory<f32>,
    pub rows: usize,
    pub cols: usize,
}

pub struct BackEnd {
    pub ctx: Arc<Context>,
}

impl CudaArray {
    pub fn new(backend: &BackEnd, rows: usize, cols: usize) -> Self {
        CudaArray {
            ctx: backend.ctx.clone(),
            v: DeviceMemory::<f32>::zeros(backend.ctx.clone(), rows * cols),
            rows: rows,
            cols: cols,
        }
    }

    pub fn from_vec(backend: &BackEnd, rows: usize, cols: usize, v: Vec<f32>) -> Self {
        assert_eq!(v.len(), rows * cols);
        let mut arr = CudaArray {
            ctx: backend.ctx.clone(),
            v: DeviceMemory::<f32>::zeros(backend.ctx.clone(), rows * cols),
            rows: rows,
            cols: cols,
        };
        for i in 0..v.len() {
            arr.v[i] = v[i];
        }
        //println!("{:?}",arr.v.as_slice());
        arr
    }

    pub fn square(&mut self) {
        let n = self.rows * self.cols;
        square(
            self.ctx.clone(),
            1, /* grid */
            n, /* block */
            &(&self.v.as_mut_ptr(), &n),
        )
        .expect("Kernel call failed");
    }
}

impl BackEnd {
    pub fn new() -> Result<Self, AccelError> {
        let device = Device::nth(0)?;
        println!("The device is: {}", device.get_name()?);
        let ctx = device.create_context();
        Ok(BackEnd { ctx: ctx })
    }
}
