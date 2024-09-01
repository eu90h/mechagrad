use std::{cell::RefCell, fmt::Debug, rc::Rc};

use ndarray::ArcArray;

use crate::{accumulate_grad::AccumulateGrad, backward_function::BackwardFunction, function::Function, node::Node, tensor::Tensor, Cell};


#[derive(Debug, Clone)]
pub struct Mul {
    pub(crate) left: Cell<Tensor>,
    pub(crate) right: Cell<Tensor>
}

impl Function for Mul {
    fn forward(&mut self) -> Tensor {
        let a = self.left.borrow();
        let b = self.right.borrow();
        let c = Tensor {
            data: a.data.clone() * b.data.clone(),
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
            let g = grad_outputs.get(0).unwrap().borrow();
            let a = self.left.borrow();
            let b = self.right.borrow();
            (ndarray::Array::ones(a.data.shape()) * g.data.clone() * b.data.clone(), ndarray::Array::ones(b.data.shape()) * g.data.clone() * a.data.clone())
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
                data: ArcArray::from_iter([grad_b.sum()]).into_dyn().into(),
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
    use ndarray::{arr1, arr2, Array};
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
    fn test_mul1() {
        for _ in 0..1000 {
            let A: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>> = Array::random((7280), Uniform::new(-100.0, 100.));
            let B: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>> = Array::random((7280), Uniform::new(-100.0, 100.));
    
            let mut W = Tensor::from(A.clone().into_dyn().into());
            let mut W2 = Tensor::from(B.clone().into_dyn().into());
            W2.requires_grad = true;

            W.requires_grad = true;
            let my_output = &W * &W2 * &W;
            let mut my_output2 = my_output.log().exp().sqrt().log().sum();
            backward(&mut my_output2);
    
            //Ground truth
            //////////////
            let (A, l) = A.into_raw_vec_and_offset();
            let pyW: tch::Tensor = tch::Tensor::from_slice(&A).reshape(&[7280]).set_requires_grad(true).to(tch::Device::Cpu);
            let (B, l) = B.into_raw_vec_and_offset();
            let pyW2: tch::Tensor = tch::Tensor::from_slice(&B).reshape(&[7280]).set_requires_grad(true).to(tch::Device::Cpu);
            let pyout = &pyW * &pyW2 * &pyW;
            let pyout2 = pyout.log().exp().sqrt().log().sum(None);
           
            pyout2.backward();
            let ground_truth: tch::Tensor = pyW.grad();
            let ground_truth2: tch::Tensor = pyW2.grad();

            //The comparison
            ////////////////
            check_close(W.grad().unwrap(), ground_truth);
            check_close(W2.grad().unwrap(), ground_truth2);
        }
    }
}