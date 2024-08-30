use std::{cell::RefCell, fmt::Debug, rc::Rc};
use ndarray::OwnedArcRepr;
use ndarray::ArrayBase;
use ndarray::prelude::*;

use crate::{accumulate_grad::AccumulateGrad, backward_function::BackwardFunction, function::Function, node::Node, tensor::Tensor, Cell}; 

#[derive(Debug, Clone)]
pub struct DotProduct {
    pub(crate) left: Cell<Tensor>,
    pub(crate) right: Cell<Tensor>
}

impl Function for DotProduct {
    fn forward(&mut self) -> Tensor {
        let a = &self.left;
       
        let a = a.borrow_mut();
      
        let c = a.data.clone();
        let c: ArrayBase<OwnedArcRepr<f64>, Dim<[usize; 1]>> = c.into_dimensionality().unwrap(); //: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>

        let b = &self.right;
        let bb = b.borrow_mut();
        let b: ArrayBase<OwnedArcRepr<f64>, Dim<[usize; 1]>> = bb.data.clone().into_dimensionality().unwrap(); //: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>
        
        let c = c.dot(&b);
        let c = Tensor {
            data: ArcArray::from(vec![c]).into_dimensionality().unwrap(),
            requires_grad: a.requires_grad || bb.requires_grad,
            is_leaf: !(a.requires_grad || bb.requires_grad),
            grad: Rc::new(RefCell::new(None)),
            grad_fn: None,
            detached: (a.detached || bb.detached),
        };
        return c;
    }

    fn backward(&mut self, grad_outputs: Vec<Cell<Tensor>>) -> Vec<Tensor> {
        let (grad_a, grad_b) = {
            let a = &self.left;
            let b = &self.right;
            let g = grad_outputs.get(0).unwrap();
            let g = g.borrow_mut();
            let d = g.data.clone(); 
            let a = a.borrow_mut();
            let v = ndarray::Array::ones(a.data.shape());
            let b = b.borrow_mut();
            let w = ndarray::Array::ones(b.data.shape());

            let ww = v;
            let ww = ww * (g.data.clone());
            let ww = ww * b.data.clone();
            let zz = w;
            let zz = zz * d.clone();
            let zz = zz * a.data.clone();

            (ww, zz)
        };

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
        
        vec![grad_a, grad_b]
    }

    fn apply(&mut self) -> Option<Vec<Tensor>> {
        //create a BackwardFunction obj representing the current node
        let mut backward_function = BackwardFunction::new(Rc::new(RefCell::new(self.clone())));
        //run subclass's forward with context manager and operation input args
        let args = vec![self.left.clone(), self.right.clone()];
        for arg in args.iter() 
        {
           // let arg = self.left.clone();
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
    fn test_dot_self() {
        //Output to test
        ////////////
        let mut x = Tensor::from(arr1(&[3.0, 1., 4., 1., 5.]).into_dyn().into());
        x.requires_grad = true;
        let mut my_output = x.dot(&x);
        my_output.backward();
        let my_truth = x.grad().unwrap();

        //Ground truth
        //////////////
        let t = tch::Tensor::from_slice(&[3.0, 1., 4., 1., 5.]).set_requires_grad(true);
        let z = t.dot(&t);
        z.backward();
        let ground_truth = t.grad();

        //The comparison
        ////////////////
        check_close(my_truth, ground_truth);
       
    }
    #[test]
    fn test_dot_freund() {
        //Output to test
        ////////////
        let mut x = Tensor::from(arr1(&[3.0, 1., 4., 1., 5.]).into_dyn().into());
        let mut y = Tensor::from(arr1(&[6.0, 7., 8., 9., 10.]).into_dyn().into());
        y.requires_grad = true;

        x.requires_grad = true;
        let mut my_output = x.dot(&y);
        my_output.backward();
        let my_truth = x.grad().unwrap();
        let my_truth2 = y.grad().unwrap();

        //Ground truth
        //////////////
        let t = tch::Tensor::from_slice(&[3.0, 1., 4., 1., 5.]).set_requires_grad(true);
        let s = tch::Tensor::from_slice(&[6.0, 7., 8., 9., 10.]).set_requires_grad(true);

        let z = t.dot(&s);
        z.backward();
        let ground_truth = t.grad();
        let ground_truth2 = s.grad();

        //The comparison
        ////////////////
        check_close(my_truth, ground_truth);
        check_close(my_truth2, ground_truth2);

       
    }
}
