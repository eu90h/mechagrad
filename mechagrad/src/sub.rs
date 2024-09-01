use std::{cell::RefCell, rc::Rc};

use ndarray::ArcArray;

use crate::{accumulate_grad::AccumulateGrad, backward_function::BackwardFunction,  function::Function, node::Node, tensor::Tensor, Cell};

#[derive(Debug, Clone)]
pub(crate) struct Sub {
    pub(crate) left: Cell<Tensor>,
    pub(crate) right: Cell<Tensor>
}
impl Function for Sub {
    fn forward(&mut self) -> Tensor {
        let a = self.left.clone();
        let b = self.right.clone();
        let a = a.borrow_mut();
        let b = b.borrow_mut();
        let c = a.data.clone() - b.data.clone();
        let c = Tensor {
            data: c,
            requires_grad: a.requires_grad || b.requires_grad,
            is_leaf: !(a.requires_grad || b.requires_grad),
            grad: Rc::new(RefCell::new(None)),
            grad_fn: None,
            detached: a.detached || b.detached,
        };
        return c;
    }

    fn backward(&mut self, grad_outputs: Vec<Cell<Tensor>>) -> Vec<Tensor> {
        let (grad_a, grad_b) = {
            let g = grad_outputs.get(0).unwrap().borrow_mut();
            let a = &self.left.borrow_mut();
            let b = &self.right.borrow_mut();
            (ndarray::Array::ones(a.data.shape()) * g.data.clone(), ndarray::Array::ones(b.data.shape()) * g.data.clone() * -1.0)
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
            let grad_b = Tensor {
                data: ArcArray::from_iter([grad_b.clone().sum()]).into_dyn().into(),
                requires_grad: false,
                is_leaf: true,
                grad: Rc::new(RefCell::new(None)),
                grad_fn: None,
                detached: true,
            };
            
            assert_eq!(self.left.borrow().data.shape(), grad_a.data.shape());
            assert_eq!(self.right.borrow().data.shape(), grad_b.data.shape());
            vec![grad_a, grad_b]
        } else if grad_b.shape() != b_shape {
            let grad_a = Tensor {
                data: grad_a.into_dimensionality().unwrap().into(), 
                requires_grad: false,
                is_leaf: true,
                grad: Rc::new(RefCell::new(None)),
                grad_fn: None,
                detached: true,
            };
            let mut gb = grad_b.clone();
            let g = {
                while gb.shape() != b_shape {
                    gb = gb.sum_axis(ndarray::Axis(0));
                }
            
                gb
            };
            let grad_b = Tensor {
                data: g.into_dyn().into(),
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
        for arg in args.iter() {
            let oarg = arg.clone();
            let mut arg = arg.borrow_mut();
            if arg.grad_fn.is_none () && !arg.detached {
                if arg.is_leaf && arg.requires_grad {
                    arg.grad_fn = Some(Rc::new(RefCell::new(Node::AccumulateGrad { inner: AccumulateGrad::from(oarg) })))
                }
            }
            let c = arg.grad_fn.clone();
            if c.is_some() && !arg.detached {
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
    fn test_sub1() {
        for _ in 0..1000 {
        //Output to test
        ////////////
        let A = Array::random((7280), Uniform::new(-1., 1.));
        let B = Array::random((7280), Uniform::new(-1., 1.));

        let mut W = Tensor::from(A.clone().into_dyn().into());
        let mut b = Tensor::from(B.clone().into_dyn().into());

        let C = Array::random(728, Uniform::new(0., 10.));
        let x = Tensor::from(C.clone().into_dyn().into());
        W.requires_grad = true;
        b.requires_grad = true;
      
        let my_output = &W - &b;
        let mut my_output = my_output.dot(&my_output);
        my_output.backward();

        //Ground truth
        //////////////
        let (A, l) = A.into_raw_vec_and_offset();
        let (B, l) = B.into_raw_vec_and_offset();
        let (C, l) = C.into_raw_vec_and_offset();
        let pyX = tch::Tensor::from_slice(&C).to(tch::Device::Cpu);
        let pyW = tch::Tensor::from_slice(&A).reshape(&[7280]).set_requires_grad(true).to(tch::Device::Cpu);
        let pyB = tch::Tensor::from_slice(&B).reshape(&[7280]).set_requires_grad(true).to(tch::Device::Cpu);
        let pyout = &pyW - &pyB;
        let pyout = pyout.dot(&pyout);
    
        pyout.backward();
        let ground_truth = pyW.grad();
        let ground_truth2 = pyB.grad();

        //The comparison
        ////////////////
        check_close(W.grad().unwrap(), ground_truth);
        check_close(b.grad().unwrap(), ground_truth2);
        }
    }
}
