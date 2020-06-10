use crate::opencl::*;

use ndarray::prelude::*;
#[cfg(test)]
use ndarray_rand::rand_distr::Uniform;
#[cfg(test)]
use ndarray_rand::RandomExt;

use ocl::Error;

#[test]
#[serial]
fn vec_squared() -> Result<(), Error> {
    let backend = CLBackEnd::new("GeForce")?;
    let (n, m) = (1, 20);
    let mut a = OpenCLArray::from_vec(backend, n, m, vec![0.5; m * n])?;
    &a.square();
    let a_result = a.to_vec()?;
    println!("a_result: {:?}", a_result);
    assert_eq!(a_result, vec![0.25; n * m]);

    Ok(())
}

#[test]
#[serial]
fn array_squared() -> Result<(), Error> {
    let backend = CLBackEnd::new("GeForce")?;
    let (n, m) = (20, 20);
    let mut array = Array2::<f32>::from_elem((n, m), 2.);
    let mut a = OpenCLArray::from_array(backend, &array)?;
    &a.square();
    let array_result = a.to_array()?;
    println!("a_result:\n{:?}", array_result);
    assert_eq!(array_result, array.mapv(|x| x.powf(2.0)));

    Ok(())
}

#[test]
#[serial]
fn array_tranpose() -> Result<(), Error> {
    let backend = CLBackEnd::new("GeForce")?;

    let mut array = array![[1., 2., 3.], [4., 5., 6.]];

    let mut a = OpenCLArray::from_array(backend, &array)?;
    let b = a.t()?;
    let result = b.to_array()?;
    println!("result: {:#?}", result);
    assert_eq!(result, array.t());

    Ok(())
}

#[test]
#[serial]
fn array_dot() -> Result<(), Error> {
    let backend = CLBackEnd::new("GeForce")?;
    // let a = array![[1., 2., 3.], [4., 5., 6.]];
    // let b = array![[1.,1.],[1.,1.],[1.,1.]];
    let a = Array::random((10, 3), Uniform::new(0., 1.));
    let b = Array::random((3, 2), Uniform::new(0., 1.));
    let c = a.dot(&b);

    let a_gpu = OpenCLArray::from_array(backend.clone(), &a)?;
    let b_gpu = OpenCLArray::from_array(backend, &b)?;
    let c_gpu = a_gpu.dot(&b_gpu)?.to_array()?;

    println!("c:\n{:#?}", c);
    println!("c_gpu:\n{:#?}", c_gpu);

    assert_eq!(c_gpu, c);

    Ok(())
}

#[test]
#[serial]
fn array_hadamard() -> Result<(), Error> {
    let backend = CLBackEnd::new("GeForce")?;

    let a = Array::random((10, 3), Uniform::new(0., 1.));
    let b = Array::random((10, 3), Uniform::new(0., 1.));

    let a_gpu = OpenCLArray::from_array(backend.clone(), &a)?;
    let b_gpu = OpenCLArray::from_array(backend, &b)?;

    let c_gpu = a_gpu.hadamard(&b_gpu)?.to_array()?;
    let c = a * b;

    println!("c:\n{:#?}", c);
    println!("c_gpu:\n{:#?}", c_gpu);
    assert_eq!(c, c_gpu);

    Ok(())
}

#[test]
#[serial]
fn array_add() -> Result<(), Error> {
    let backend = CLBackEnd::new("GeForce")?;

    let a = Array::random((10, 3), Uniform::new(0., 1.));
    let b = Array::random((10, 3), Uniform::new(0., 1.));

    let a_gpu = OpenCLArray::from_array(backend.clone(), &a)?;
    let b_gpu = OpenCLArray::from_array(backend, &b)?;

    let c_gpu = a_gpu.add(&b_gpu)?.to_array()?;
    let c = a + b;

    println!("c:\n{:#?}", c);
    println!("c_gpu:\n{:#?}", c_gpu);
    assert_eq!(c, c_gpu);

    Ok(())
}

#[test]
#[serial]
fn array_subtract() -> Result<(), Error> {
    let backend = CLBackEnd::new("GeForce")?;

    let a = Array::random((10, 3), Uniform::new(0., 1.));
    let b = Array::random((10, 3), Uniform::new(0., 1.));

    let a_gpu = OpenCLArray::from_array(backend.clone(), &a)?;
    let b_gpu = OpenCLArray::from_array(backend, &b)?;

    let c_gpu = a_gpu.subtract(&b_gpu)?.to_array()?;
    let c = a - b;

    println!("c:\n{:#?}", c);
    println!("c_gpu:\n{:#?}", c_gpu);
    assert_eq!(c, c_gpu);

    Ok(())
}

fn sigmoid_op(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[test]
#[serial]
fn array_sigmoid() -> Result<(), Error> {
    let backend = CLBackEnd::new("GeForce")?;
    let a: Array2<f32> = Array::random((8, 10), Uniform::new(0.49, 0.51));
    let (n, m): (usize, usize) = (a.nrows(), a.ncols());

    let b = a.mapv(|x| sigmoid_op(x));

    let a_gpu = OpenCLArray::from_array(backend, &a)?;
    let b_gpu = a_gpu.sigmoid()?.to_array()?;

    let epsilon = 1e-5;
    for y in 0..n {
        for x in 0..m {
            println!(
                "{} - {} = {} < {}",
                b_gpu[[y, x]],
                b[[y, x]],
                b_gpu[[y, x]] - b[[y, x]],
                epsilon
            );
            assert!((b_gpu[[y, x]] - b[[y, x]]).abs() < epsilon);
        }
    }
    Ok(())
}