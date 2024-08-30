use std::{cell::RefCell, fmt::Debug, rc::Rc};

use crate::{accumulate_grad::AccumulateGrad, backward_function::BackwardFunction, function::Function, node::Node, tensor::Tensor, Cell}; 

#[derive(Debug, Clone)]
pub struct Tranpose {
    pub(crate) left: Cell<Tensor>,
}

impl Function for Tranpose {
    fn forward(&mut self) -> Tensor {
        let a = &self.left.borrow_mut();
        Tensor {
            data: a.data.t().to_owned().into_dyn().into(),
            requires_grad: a.requires_grad,
            is_leaf: false || (a.detached),
            grad: Rc::new(RefCell::from(None)),
            grad_fn: None,
            detached: false,
        }
    }

    fn backward(&mut self, grad_outputs: Vec<Cell<Tensor>>) -> Vec<Tensor> {
        let grad_a = {
            let g = grad_outputs.get(0).unwrap().borrow();
            let g = &g.data;
            g.t().to_owned()
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