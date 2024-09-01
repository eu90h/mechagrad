use std::borrow::Borrow;
use std::{cell::RefCell, fmt::Debug, rc::Rc};

use crate::{accumulate_grad::AccumulateGrad, backward_function::BackwardFunction, function::Function, node::Node, tensor::Tensor, Cell}; 

#[derive(Debug, Clone)]
pub struct Sqrt {
    pub(crate) left: Cell<Tensor>,
}

impl Function for Sqrt {
    fn forward(&mut self) -> Tensor {
        let a = &self.left;
        let a = a.borrow_mut();
        let c = a.data.borrow().clone();
       
        let c = Tensor {
            data: c.mapv_into(|v| v.sqrt()).into_dyn(),
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
            let a = &self.left.borrow_mut();
            let g = grad_outputs.get(0).unwrap();
            a.data.clone().mapv_into(|v: f64| 0.5/v.sqrt()) * g.borrow_mut().data.clone()
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
    use ndarray::arr1;

    use crate::backward::backward;

    use super::*;
    
    fn check_close2(a: Tensor, b: tch::Tensor) {
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
    fn test_sqrt() {
        //Output to test
        ////////////
        let mut x = Tensor::from(arr1(&[3.0, 1., 4., 1., 5.]).into_dyn().into());
        x.requires_grad = true;
        let my_output = x.sqrt();
        let mut my_output: Tensor = Tensor::dot(&my_output, &my_output);
        backward(&mut my_output);
        let mine = x.grad().unwrap();

        //Ground truth
        //////////////
        let t = tch::Tensor::from_slice(&[3.0, 1., 4., 1., 5.]).set_requires_grad(true);
        let z = t.sqrt();
        let z = z.dot(&z);
        z.backward();
        let ground_truth = t.grad();

        //The comparison
        ////////////////
        check_close2(mine, ground_truth);
    }
}
