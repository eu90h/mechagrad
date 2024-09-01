use std::borrow::Borrow;
use std::f64::consts::E;
use std::{cell::RefCell, fmt::Debug, rc::Rc};
use num_traits::Inv;

use crate::{accumulate_grad::AccumulateGrad, backward_function::BackwardFunction, function::Function, node::Node, tensor::Tensor, Cell}; 

#[derive(Debug, Clone)]
pub struct Log {
    pub(crate) left: Cell<Tensor>,
    //TODO: add a base arg
}

impl Function for Log {
    fn forward(&mut self) -> Tensor {
        let a = &self.left;
        let a = a.borrow_mut();
        let c = a.data.borrow().clone();
       
        let c = Tensor {
            data: c.mapv_into(|v| v.log(E)).into_dyn(),
            requires_grad: a.requires_grad,
            is_leaf: !a.requires_grad,
            grad: Rc::new(RefCell::new(None)),
            grad_fn: None,
            detached: false,
        };
        return c;
    }

    fn backward(&mut self, grad_outputs: Vec<Cell<Tensor>>) -> Vec<Tensor> {
        let grad_a = {
            let a = &self.left.borrow_mut();
            let g = grad_outputs.get(0).unwrap();
            let from = a.data.clone();
            from.mapv_into(|v: f64| v.inv()) * g.borrow_mut().data.clone()
        };

        let grad_a = Tensor {
            data: grad_a.into_dimensionality().unwrap(),
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
        {
            let arg = self.left.clone();
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
    use ndarray::{arr1, Array};
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
    fn test_log() {
        //Output to test
        ////////////
        let mut x = Tensor::from(arr1(&[3.0, 1., 4., 1., 5.]).into_dyn().into());
        x.requires_grad = true;
        let my_output = x.log();
        let mut my_output: Tensor = Tensor::dot(&my_output, &my_output);
        backward(&mut my_output);
        let my_truth = x.grad().unwrap();

        //Ground truth
        //////////////
        let t = tch::Tensor::from_slice(&[3.0, 1., 4., 1., 5.]).set_requires_grad(true);
        let z = t.log();
        let z = z.dot(&z);
        z.backward();
        let ground_truth = t.grad();

        //The comparison
        ////////////////
        check_close(my_truth, ground_truth);
    }

    #[test]
    fn test_log2() {
        for _ in 0..1000 {
        //Output to test
        ////////////
        let A = Array::random((7280), Uniform::new(0.01, 100.));
        let mut W = Tensor::from(A.clone().into_dyn().into());
        W.requires_grad = true;
      
        let mut my_output = W.exp().sum().log();
        my_output.backward();

        //Ground truth
        //////////////
        let (A, l) = A.into_raw_vec_and_offset();
        let pyW = tch::Tensor::from_slice(&A).reshape(&[7280]).set_requires_grad(true).to(tch::Device::Cpu);

        let pyout = pyW.exp().sum(None).log();
        pyout.backward();
        let ground_truth = pyW.grad();

        //The comparison
        ////////////////
        check_close(my_output, pyout);
        check_close(W.grad().unwrap(), ground_truth);
        }
    }
}
