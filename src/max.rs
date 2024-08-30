use std::{cell::RefCell, fmt::Debug, rc::Rc};
use ndarray::prelude::*;

use crate::{accumulate_grad::AccumulateGrad, backward_function::BackwardFunction, function::Function, node::Node, tensor::Tensor, Cell}; 

#[derive(Debug, Clone)]
pub struct Max {
    pub(crate) left: Cell<Tensor>,
    pub(crate) argmax: usize,
}

impl Function for Max {
    fn forward(&mut self) -> Tensor {
        let a = &self.left;
        let a = a.borrow_mut();
        let c = a.data.clone();
        let m = c.into_iter().reduce(|arg0: f64, arg1: f64| f64::max(arg1, arg0)).unwrap();

        let c = Tensor {
            data: ArcArray::from_vec(vec![m]).into_dyn(),
            requires_grad: a.requires_grad,
            is_leaf: !a.requires_grad,
            grad: Rc::new(RefCell::from(None)),
            grad_fn: None,
            detached: a.detached,
        };

        return c;
    }

    fn backward(&mut self, grad_outputs: Vec<Cell<Tensor>>) -> Vec<Tensor> {
        let grad_a = {
            let g = grad_outputs.get(0).unwrap();
            let g = g.borrow();
            let g = g.data.clone();
            g.to_owned()
        };
        let l = self.left.borrow_mut();
        let mut z = ArcArray::zeros(l.data.shape());
        z[self.argmax] = 1.0;
        let grad_a = Tensor {
            data:(grad_a*z).into(),
            requires_grad: false,
            is_leaf: false,
            grad: Rc::new(RefCell::new(None)),
            grad_fn: None,
            detached: true,
        };
        vec![grad_a]
    }

    fn apply(&mut self) -> Option<Vec<Tensor>> {
        //create a BackwardFunction obj representing the current node
        let mut backward_function = BackwardFunction::new(Rc::new(RefCell::new(self.clone())));
        //run subclass's forward with context manager and operation input args
        {
            let arg = self.left.clone();
            let oarg = arg.clone();
            let mut arg = arg.borrow_mut();
            if arg.grad_fn.is_none() && !arg.detached {
                if arg.is_leaf && arg.requires_grad {
                    arg.grad_fn = Some(Rc::new(RefCell::new(Node::AccumulateGrad { inner: AccumulateGrad::from(oarg) })))
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
            assert_near!(x, y, 1e-8);
        }
    }

    #[test]
    fn test_random1() {
        for _ in 0..1000 {
        //Output to test
        ////////////
        let A = Array::random((10, 728), Uniform::new(-188., 188.));
        let B = Array::random((10), Uniform::new(-188., 188.));
        let mut W = Tensor::from(A.clone().into_dyn().into());
        let mut b = Tensor::from(B.clone().into_dyn().into());

        let C = Array::random(728, Uniform::new(-10., 10.));
        let x = Tensor::from(C.clone().into_dyn().into());
        b.requires_grad = true;
      
        let mut my_output = b.max();
        my_output.backward();
        //////////////
        let (A, l) = A.into_raw_vec_and_offset();
        let (B, l) = B.into_raw_vec_and_offset();
        let (C, l) = C.into_raw_vec_and_offset();
        let pyX = tch::Tensor::from_slice(&C).to(tch::Device::Cpu);
        let pyW = tch::Tensor::from_slice(&A).reshape(&[10, 728]).set_requires_grad(true).to(tch::Device::Cpu);
        let pyB = tch::Tensor::from_slice(&B).reshape(&[10]).set_requires_grad(true).to(tch::Device::Cpu);
        let pyout =  pyB.max();
        pyout.backward();
        let ground_truth2 = pyB.grad();

        //The comparison
        ////////////////
        check_close(my_output, pyout);
        check_close(b.grad().unwrap(), ground_truth2);
        }
    }

    #[test]
    fn test_random2() {
        for _ in 0..1000 {
        //Output to test
        ////////////
        let A = Array::random((10, 728), Uniform::new(-188., 188.));
        let B = Array::random((10), Uniform::new(-188., 188.));
        let mut b = Tensor::from(B.clone().into_dyn().into());
        let C = Array::random(728, Uniform::new(-10., 10.));
        b.requires_grad = true;
        let my_output = b.clone() + b.max();
        let mut my_output2 = my_output.sum();
        my_output2.backward();

        //////////////
        let (A, l) = A.into_raw_vec_and_offset();
        let (B, l) = B.into_raw_vec_and_offset();
        let (C, l) = C.into_raw_vec_and_offset();
        let pyX = tch::Tensor::from_slice(&C).to(tch::Device::Cpu);
        let pyB = tch::Tensor::from_slice(&B).reshape(&[10]).set_requires_grad(true).to(tch::Device::Cpu);
        let pyout = &pyB + &pyB.max();
        let pyout2 = pyout.sum(None);
        pyout2.backward();
        let ground_truth2 = pyB.grad();

        //The comparison
        ////////////////
        check_close(b.grad().unwrap(), ground_truth2);
        }
    }
}