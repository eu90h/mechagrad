use std::iter::zip;
use all_asserts::assert_near;
use mechagrad::tensor::Tensor;
use ndarray::{ArcArray, Array};
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use tch::no_grad;

fn check_close(a: Tensor, b: tch::Tensor) {
  let a_shape = a.data.shape();
  let b_shape:Vec<i64> = b.size();
  for (x,y) in zip(a_shape, b_shape) {
      assert_eq!(*x, y as usize);
  }

  let a: Vec<f64>  = a.try_into().unwrap();
  let b: Vec<f64> = b.flatten(0, -1).try_into().unwrap();
  for (x,y) in zip(a, b) {
      assert_near!(x, y, 1e-8);
  }
}

#[allow(non_snake_case)]
fn main() {
    let mnist = tch::vision::mnist::load_dir("MNIST_ORG").unwrap();
    let A = Array::random((128,784), Uniform::new(-1., 1.));
    let B = Array::random((10,128), Uniform::new(-1., 1.));
    let C = Array::random(128, Uniform::new(-1., 1.));
    let D = Array::random(10, Uniform::new(-1., 1.));

    let mut W1 = Tensor::from(A.clone().into_dyn().into());
    W1.requires_grad = true;
    let mut W2  = Tensor::from(B.clone().into_dyn().into());
    W2.requires_grad = true;
    let mut B1  = Tensor::from(C.clone().into_dyn().into());
    B1.requires_grad = true;
    let mut B2  = Tensor::from(D.clone().into_dyn().into());
    B2.requires_grad = true;

    let (A, _) = A.into_raw_vec_and_offset();
    let (B, _) = B.into_raw_vec_and_offset();
    let (C, _) = C.into_raw_vec_and_offset();
    let (D, _) = D.into_raw_vec_and_offset();

    let mut pyW2 = tch::Tensor::from_slice(&B).reshape(&[10, 128]).to(tch::Device::Cpu).set_requires_grad(true);
    let mut pyW1 = tch::Tensor::from_slice(&A).reshape(&[128, 784]).set_requires_grad(true).to(tch::Device::Cpu);
    let mut pyB1 = tch::Tensor::from_slice(&C).reshape(&[128]).set_requires_grad(true).to(tch::Device::Cpu);
    let mut pyB2 = tch::Tensor::from_slice(&D).reshape(&[10]).set_requires_grad(true).to(tch::Device::Cpu);
    let mut my_hits = 0;
    let mut py_hits = 0;
    let mut trials = 0;
    for i in 0..mnist.train_images.size()[0] {
      trials += 1;
      let xx = mnist.train_images.get(i).internal_cast_double(false);
      let yy = mnist.train_labels.get(i).internal_cast_long(false);
      let v : Vec<f64> = xx.copy().try_into().unwrap();
      let x = Tensor::from(Array::from_vec(v).into_dyn().into());
      let v : f64 = yy.copy().try_into().unwrap();
      let v = v as usize;

      // Mechagrad
      let my_output = (&W1.matmul(&x) + &B1).relu();
      let mut logits = &mut W2.matmul(&my_output) + &mut B2;
      let logits2 = logits.logsoftmax()  ;
      let y_hat = logits2.argmax();
      if y_hat == v {
          my_hits += 1;
      }
      let mut z = vec![0.0, 0.0, 0.0, 0., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
      z[v as usize] = 1.0;
      let rhs = Tensor::from(ArcArray::from_vec(z).into_dyn().into());
      let mut loss = logits2.dot(&rhs)*-1.0;
      W1.zero_grad();
      W2.zero_grad();
      B1.zero_grad();
      B2.zero_grad();
      loss.backward();

      // Torch
      let pyX = xx;
      let pyoutput = (pyW1.matmul(&pyX) + &pyB1).relu();
      let pyoutput2 = pyW2.matmul(&pyoutput) + &pyB2;
      let mut z = vec![0.0, 0.0, 0.0, 0., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
      z[v as usize] = 1.0;
      let rhs = tch::Tensor::from_slice(&z).internal_cast_double(false);
      let pyloss = pyoutput2.cross_entropy_loss::<&tch::Tensor>(&rhs, None, tch::Reduction::Sum, -1, 0.);
      pyW1.zero_grad();
      pyW2.zero_grad();
      pyB1.zero_grad();
      pyB2.zero_grad();
      pyloss.backward();
      no_grad(|| {
        let y_hat: f64 = pyoutput2.argmax(-1, false).try_into().unwrap();
        if y_hat as usize == v {
            py_hits += 1;
        }
      });

      //Comparisons & Updates
      if i % 100 == 0 { println!("[train example {}] mechagrad {}\ttorch {}", i, my_hits as f64/trials as f64, py_hits as f64/trials as f64); }
      check_close(W1.clone(), pyW1.copy());
      check_close(W2.clone(), pyW2.copy());
      check_close(W1.clone().grad().unwrap(), pyW1.grad());
      check_close(W2.clone().grad().unwrap(), pyW2.grad());

      W1.grad_step(1e-3);
      W2.grad_step(1e-3);
      B1.grad_step(1e-3);
      B2.grad_step(1e-3);
      tch::no_grad(|| {
        pyW1 += pyW1.grad() * (-1e-3);
        pyW2 += pyW2.grad() * (-1e-3);
        pyB1 += pyB1.grad() * (-1e-3);
        pyB2 += pyB2.grad() * (-1e-3);
      });
  
      check_close(W1.clone(), pyW1.copy());
      check_close(W2.clone(), pyW2.copy());
    }

    let mut my_test_hits = 0;
    let mut py_test_hits = 0;
    let mut trials = 0;
    for i in 0..mnist.test_images.size()[0] {
      trials += 1;
      let xx = mnist.train_images.get(i).internal_cast_double(false);
      let yy = mnist.train_labels.get(i).internal_cast_long(false);
      let v : Vec<f64> = xx.copy().try_into().unwrap();
      let x = Tensor::from(Array::from_vec(v).into_dyn().into());
      let v : f64 = yy.copy().try_into().unwrap();
      let v = v as usize;

      // Mechagrad
      let my_output = (&W1.matmul(&x) + &B1).relu();
      let mut logits = &mut W2.matmul(&my_output) + &mut B2;
      let logits2 = logits.logsoftmax()  ;
      let y_hat = logits2.argmax();
      if y_hat == v {
        my_test_hits += 1;
      }
      let mut z = vec![0.0, 0.0, 0.0, 0., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
      z[v as usize] = 1.0;
      let rhs = Tensor::from(ArcArray::from_vec(z).into_dyn().into());
      let mut loss = logits2.dot(&rhs)*-1.0;
      W1.zero_grad();
      W2.zero_grad();
      B1.zero_grad();
      B2.zero_grad();
      loss.backward();

      // Torch
      let pyX = xx;
      let pyoutput = (pyW1.matmul(&pyX) + &pyB1).relu();
      let pyoutput2 = pyW2.matmul(&pyoutput) + &pyB2;
      let mut z = vec![0.0, 0.0, 0.0, 0., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
      z[v as usize] = 1.0;
      let rhs = tch::Tensor::from_slice(&z).internal_cast_double(false);
      let pyloss = pyoutput2.cross_entropy_loss::<&tch::Tensor>(&rhs, None, tch::Reduction::Sum, -1, 0.);
      pyW1.zero_grad();
      pyW2.zero_grad();
      pyB1.zero_grad();
      pyB2.zero_grad();
      pyloss.backward();
      no_grad(|| {
        let y_hat: f64 = pyoutput2.argmax(-1, false).try_into().unwrap();
        if y_hat as usize == v {
          py_test_hits += 1;
        }
      });

      //Comparisons & Updates
      if i % 100 == 0 { println!("[test example {}] mechagrad {}\ttorch {}", i, my_test_hits as f64/trials as f64, py_test_hits as f64/trials as f64); }
    }

    println!("mechagrad model test accuracy: {}\ntorch model test accuracy: {}", my_test_hits as f64/trials as f64, py_test_hits as f64/trials as f64);
}