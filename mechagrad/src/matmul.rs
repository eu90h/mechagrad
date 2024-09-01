use std::{cell::RefCell, fmt::Debug, rc::Rc};
use ndarray::prelude::*;
use ndarray::Array;

use crate::{accumulate_grad::AccumulateGrad, backward_function::BackwardFunction, function::Function, node::Node, tensor::Tensor, Cell}; 

//credit to github user @zgbkdlm
//source: https://github.com/rust-ndarray/ndarray/issues/1148
pub fn outer(x: &ArcArray<f64, Ix1>, y: &Array<f64, Ix1>) -> Array<f64, Ix2> {
    let (size_x, size_y) = (x.shape()[0], y.shape()[0]);
    let x_reshaped = x.view().into_shape_with_order((size_x, 1)).unwrap();
    let y_reshaped = y.view().into_shape_with_order((1, size_y)).unwrap();
    x_reshaped.dot(&y_reshaped)
}

#[derive(Debug, Clone)]
pub struct MatMul {
    pub(crate) left: Cell<Tensor>,
    pub(crate) right: Cell<Tensor>,
    pub(crate) right_is_vec: bool,
}

impl Function for MatMul {
    fn forward(&mut self) -> Tensor {

        let a = self.left.borrow_mut();
        let c: ArcArray<f64, Dim<[usize; 2]>> = a.data.clone().into_dimensionality().unwrap();
        let bb = self.right.borrow();
        if bb.data.shape().len() == 1 {
            self.right_is_vec = true;
            let b: ArcArray<f64, Dim<[usize; 1]>> = bb.data.clone().into_dimensionality().unwrap();
            let c = c.dot(&b);
            let c = Tensor {
                data: c.into_dyn().into(),
                requires_grad: a.requires_grad || bb.requires_grad,
                is_leaf: !(a.requires_grad || bb.requires_grad),
                grad: Rc::new(RefCell::new(None)),
                grad_fn: None,
                detached: (a.detached || bb.detached),
            };
            return c;
        } else {
            assert!(bb.data.shape().len() == 2);
            self.right_is_vec = false;
            let b: ArcArray<f64, Dim<[usize; 2]>> = bb.data.clone().into_dimensionality().unwrap();
            let c = c.dot(&b);
            let c = Tensor {
                data: c.into_dyn().into(),
                requires_grad: a.requires_grad || bb.requires_grad,
                is_leaf: !(a.requires_grad || bb.requires_grad),
                grad: Rc::new(RefCell::new(None)),
                grad_fn: None,
                detached: (a.detached || bb.detached),
            };
            return c;
        }
        
        
    }

    fn backward(&mut self, grad_outputs: Vec<Cell<Tensor>>) -> Vec<Tensor> {
        let (grad_a, grad_b) = {
            let g = grad_outputs.get(0).unwrap();
            let g = g.borrow();
            let g: ArcArray<f64, Dim<[usize; 1]>> = g.data.clone().into_dimensionality().unwrap();
            let left: ArcArray<f64, Dim<[usize; 2]>>  = self.left.borrow().data.clone().into_dimensionality().unwrap();
            let right: ArcArray<f64, Dim<[usize; 1]>> = self.right.borrow().data.clone().into_dimensionality().unwrap();
            let dw: ArcArray<f64, Dim<[usize; 2]>> = outer(&g, &right.t().to_owned()).into() ;
            let da: ArcArray<f64, Dim<[usize; 1]>>  = left.t().dot(&g).into();
            (dw, da)
        };
        let binding = self.right.borrow();
        let b_shape = binding.data.shape();
        if grad_b.shape().len() == 1 && grad_b.shape() != b_shape {
            let grad_a = Tensor {
                data: grad_a.into_dimensionality().unwrap().into(), 
                requires_grad: false,
                is_leaf: true,
                grad: Rc::new(RefCell::new(None)),
                grad_fn: None,
                detached: true,
            };
            let gb = grad_b.clone();
            let g = gb.sum();
            let grad_b = Tensor {
                data: ArcArray::from_iter([g]).into_dyn().into(),
                requires_grad: false,
                is_leaf: true,
                grad: Rc::new(RefCell::new(None)),
                grad_fn: None,
                detached: true,
            };
            
            assert_eq!(self.left.borrow().data.shape(), grad_a.data.shape());
            assert_eq!(self.right.borrow().data.shape(), grad_b.data.shape());
            vec![grad_a, grad_b]
        
        } else {
            let grad_a = Tensor {
                data: grad_a.into_dimensionality().unwrap().into(), 
                requires_grad: false,
                is_leaf: true,
                grad: Rc::new(RefCell::new(None)),
                grad_fn: None,
                detached: true,
            };
            
            let grad_b = Tensor {
                data: grad_b.into_dimensionality().unwrap().into(),
                requires_grad: false,
                is_leaf: true,
                grad: Rc::new(RefCell::new(None)),
                grad_fn: None,
                detached: true,
            };
            
            assert_eq!(self.left.borrow().data.shape(), grad_a.data.shape());
            assert_eq!(self.right.borrow().data.shape(), grad_b.data.shape());
            vec![grad_a, grad_b]
        }
    
    }

    fn apply(&mut self) -> Option<Vec<Tensor>> {
        //create a BackwardFunction obj representing the current node
        let mut backward_function = BackwardFunction::new(Rc::new(RefCell::new(self.clone())));
        //run subclass's forward with context manager and operation input args
        let args = [self.left.clone(), self.right.clone()];
        for arg in args.iter()//{
        {
            let oarg = arg.clone();
            let mut arg = arg.borrow_mut();
            if arg.grad_fn.is_none() && !arg.detached {
                if arg.is_leaf && arg.requires_grad {
                    arg.grad_fn = Some(Rc::new(RefCell::new(Node::AccumulateGrad { inner: AccumulateGrad::from(oarg) })))
                } else if arg.is_leaf && !arg.requires_grad {
                    arg.grad_fn = None;
                }
            }
            let c = arg.grad_fn.clone();
            if c.is_some() {
                backward_function.next_functions.push(c.unwrap());
            }
        }

        let mut output_tensor = self.forward();

        if !output_tensor.detached {
            output_tensor.grad_fn = Some(Rc::new(RefCell::new(Node::BackwardFunctionWrapper { inner: backward_function })));
        }

        Some(vec![output_tensor])
    }
}

#[allow(non_snake_case, unused)]
#[cfg(test)]
mod tests {
    use std::iter::zip;

    use all_asserts::assert_near;
    use ndarray_rand::{rand_distr::Uniform, RandomExt};

    use crate::backward::backward;
    use super::*;
    fn check_close(a: Tensor, b: tch::Tensor) {
        let a_shape = a.data.shape();
        let b_shape:Vec<i64> = b.size();
        for (x,y) in zip(a_shape, b_shape) {
            assert_eq!(*x, y as usize);
        }
    
        let a: Vec<f64>  = a.try_into().unwrap();
        let b: Vec<f64> = b.flatten(0, -1).try_into().unwrap();
        for (x,y) in zip(a, b) {
            assert_near!(x, y, 1e-1);
        }
    }

    #[test]
    fn test_simple_matmul1() {
        //Output to test
        ////////////
        let mut W = Tensor::from(arr2(&[[0.7f64, 0.9], [0.5, 0.34]]).into_dyn().into());
        let x = Tensor::from(arr1(&[1.0f64, 2.0]).into_dyn().into());
        W.requires_grad = true;
        let my_output = W.matmul(&x);
        let mut my_output = my_output.dot(&my_output);
        backward(&mut my_output);

        //Ground truth
        //////////////
        let pyX = tch::Tensor::from_slice(&[1.0, 2.0]).to(tch::Device::Cpu);
        let pyW = tch::Tensor::from_slice2(&[[0.7f64, 0.9], [0.5, 0.34]]).set_requires_grad(true).to(tch::Device::Cpu);

        let pyout = pyW.matmul(&pyX);
        let pyout = pyout.dot(&pyout);
       
        pyout.backward();
        let ground_truth = pyW.grad();

        //The comparison
        ////////////////
        check_close(W.grad().unwrap(), ground_truth);
    }

    #[test]
    fn test_simple_matmul2() {
        //Output to test
        ////////////
        let mut W = Tensor::from(arr2(&[[0.7f64, 0.9], [0.5, 0.34]]).into_dyn().into());
        let x = Tensor::from(arr1(&[1.0f64, 2.0]).into_dyn().into());
        W.requires_grad = true;
        let my_output = W.matmul(&x);
        let my_output = W.matmul(&my_output);
        let mut my_output = my_output.dot(&my_output);
        backward(&mut my_output);

        //Ground truth
        //////////////
        let pyX = tch::Tensor::from_slice(&[1.0, 2.0]).to(tch::Device::Cpu);
        let pyW: tch::Tensor = tch::Tensor::from_slice2(&[[0.7f64, 0.9], [0.5, 0.34]]).set_requires_grad(true).to(tch::Device::Cpu);

        let pyout = pyW.matmul(&pyX);
        let pyout = pyW.matmul(&pyout);

        let pyout = pyout.dot(&pyout);
       
        pyout.backward();
        let ground_truth = pyW.grad();

        //The comparison
        ////////////////
        check_close(W.grad().unwrap(), ground_truth);
    }

    #[test]
    fn test_simple_matmul3() {
        //Output to test
        ////////////
        let mut W = Tensor::from(arr2(&[[0.7f64, 0.9], [0.5, 0.34]]).into_dyn().into());
        let mut W2 = Tensor::from(arr2(&[[0.2f64, 0.3], [0.2, 0.1]]).into_dyn().into());

        let x = Tensor::from(arr1(&[1.0f64, 2.0]).into_dyn().into());
        W.requires_grad = true;
        W2.requires_grad = true;

        let my_output = W.matmul(&x);
        let my_output: Tensor = W2.matmul(&my_output);
        let mut my_output = my_output.dot(&my_output);
        my_output.backward();

        //Ground truth
        //////////////
        let pyX = tch::Tensor::from_slice(&[1.0, 2.0]).to(tch::Device::Cpu);
        let pyW = tch::Tensor::from_slice2(&[[0.7f64, 0.9], [0.5, 0.34]]).set_requires_grad(true).to(tch::Device::Cpu);
        let pyW2 = tch::Tensor::from_slice2(&[[0.2f64, 0.3], [0.2, 0.1]]).set_requires_grad(true).to(tch::Device::Cpu);

        let pyout = pyW.matmul(&pyX);
        let pyout = pyW2.matmul(&pyout);

        let pyout = pyout.dot(&pyout);
       
        pyout.backward();
        let ground_truth = pyW.grad();
        let ground_truth2 = pyW2.grad();

        //The comparison
        ////////////////
        check_close(W.grad().unwrap(), ground_truth);
        check_close(W2.grad().unwrap(), ground_truth2);
    }

    #[test]
    fn test_mnist2() {
        //Output to test
        ////////////
        let A = Array::random((128, 728), Uniform::new(-100., 100.));
        let B = Array::random((10, 128), Uniform::new(-100., 100.));
        let mut W = Tensor::from(A.clone().into_dyn().into());
        let mut W2 = Tensor::from(B.clone().into_dyn().into());

        let C = Array::random(728, Uniform::new(-100., 100.));
        let x = Tensor::from(C.clone().into_dyn().into());
        W.requires_grad = true;
        W2.requires_grad = true;

        let my_output = W.matmul(&x);
        let my_output: Tensor = W2.matmul(&my_output);
        let mut my_output = my_output.dot(&my_output);
        my_output.backward();

        //Ground truth
        //////////////
        let (A, l) = A.into_raw_vec_and_offset();
        let (C, l) = C.into_raw_vec_and_offset();
        let pyX = tch::Tensor::from_slice(&C).to(tch::Device::Cpu);
        let pyW = tch::Tensor::from_slice(&A).reshape(&[128, 728]).set_requires_grad(true).to(tch::Device::Cpu);
        let (B, l) = B.into_raw_vec_and_offset();
        let pyW2 = tch::Tensor::from_slice(&B).reshape(&[10, 128]).set_requires_grad(true).to(tch::Device::Cpu);

        let pyout = pyW.matmul(&pyX);
        let pyout = pyW2.matmul(&pyout);

        let pyout = pyout.dot(&pyout);
       
        pyout.backward();
        let ground_truth = pyW.grad();
        let ground_truth2 = pyW2.grad();

        //The comparison
        ////////////////
        check_close(W.grad().unwrap(), ground_truth);
        check_close(W2.grad().unwrap(), ground_truth2);
    }
}
