use std::{cell::RefCell, fmt::Debug, rc::Rc};
use crate::{accumulate_grad::AccumulateGrad, backward_function::BackwardFunction, function::Function, node::Node, tensor::Tensor, Cell}; 


#[derive(Debug, Clone)]
pub struct Reshape {
    pub(crate) left: Cell<Tensor>,
    pub(crate) shape: Vec<usize>,
    pub(crate) old_shape: Vec<usize>,
}

impl Function for Reshape {
    fn forward(&mut self) -> Tensor {
        let a = &self.left;
        let a = a.borrow_mut();
        let c = a.data.clone();
        let c = c.to_shape(self.shape.clone()).unwrap().to_owned();
        let c = Tensor {
            data: c.into_dyn().into(),
            requires_grad: a.requires_grad,
            is_leaf: !a.requires_grad,
            grad: Rc::new(RefCell::new(None)),
            grad_fn: None,
            detached: a.detached,
        };
        return c;
    }

    fn backward(&mut self, grad_outputs: Vec<Cell<Tensor>>) -> Vec<Tensor> {
        let grad_a = {
            let g = grad_outputs.get(0).unwrap().borrow();
            g.data.to_shape(self.old_shape.clone()).unwrap().to_owned()
        };

        let grad_a = Tensor {
            data: grad_a.into_dimensionality().unwrap().into(),
            requires_grad: false,
            is_leaf: true,
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
        //let args = [self.left.clone(), self.right.clone()];
        //for arg in args.iter() {
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
    use ndarray::Array;
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
    fn test_reshape1() {
        for _ in 0..1000 {
        //Output to test
        ////////////
        let A = Array::random((7280), Uniform::new(-1., 1.));
        let B = Array::random((10), Uniform::new(-1., 1.));

        let mut W = Tensor::from(A.clone().into_dyn().into());
        let mut b = Tensor::from(B.clone().into_dyn().into());

        let C = Array::random(728, Uniform::new(0., 10.));
        let x = Tensor::from(C.clone().into_dyn().into());
        W.requires_grad = true;
      
        let my_output = W.reshape(&[10, 728]).matmul(&x) + b;
        let mut my_output = my_output.dot(&my_output);
        my_output.backward();

        //Ground truth
        //////////////
        let (A, l) = A.into_raw_vec_and_offset();
        let (B, l) = B.into_raw_vec_and_offset();
        let (C, l) = C.into_raw_vec_and_offset();
        let pyX = tch::Tensor::from_slice(&C).to(tch::Device::Cpu);
        let pyW = tch::Tensor::from_slice(&A).reshape(&[7280]).set_requires_grad(true).to(tch::Device::Cpu);
        let pyB = tch::Tensor::from_slice(&B).reshape(&[10]).set_requires_grad(true).to(tch::Device::Cpu);
        let pyout = pyW.reshape([10, 728]).matmul(&pyX) + pyB;
        let pyout = pyout.dot(&pyout);
    
        pyout.backward();
        let ground_truth = pyW.grad();

        //The comparison
        ////////////////
        check_close(W.grad().unwrap(), ground_truth);
        }
    }
}