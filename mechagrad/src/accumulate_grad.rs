use std::{cell::RefCell, rc::{Rc, Weak}};

use crate::{function::Function, node::Node, tensor::Tensor, Cell};

#[derive(Debug, Clone)]
pub struct AccumulateGrad {
    pub(crate) variable: Weak<RefCell<Tensor>>,
    pub(crate) next_functions: Vec<Cell<Node>>,
    pub(crate) args: Vec<Cell<Tensor>>,
}

impl AccumulateGrad {
    pub(crate) fn from(variable: Cell<Tensor>) -> Self {
        AccumulateGrad {
            variable: Rc::downgrade(&variable),
            next_functions: vec![],
            args: vec![]
        }
    }
}

impl Function for AccumulateGrad {
    fn forward(&mut self) -> Tensor {
        todo!()
    }

    fn backward(&mut self, _grad_outputs: Vec<Cell<Tensor>>) -> Vec<Tensor> {
        todo!()
    }

    fn apply(&mut self) -> Option<Vec<Tensor>> {
        match self.variable.upgrade() {
            Some(variable) => {
                let grad_is_none = variable.borrow_mut().grad.as_ref().borrow().is_none();
                if grad_is_none {
                    let variable = variable.borrow_mut();
                    let a = self.args.get(0).unwrap().borrow_mut();
                    let arg = a.clone();
                    variable.grad.replace(Some(arg));
                } else {
                    let variable = variable.borrow_mut();
                    let a = self.args.get(0).unwrap().borrow_mut();
                    let arg = a.clone();
                    variable.grad.replace_with(|x| {
                        let x = x.as_ref().unwrap();
                        assert!(x.detached && !x.requires_grad);
                        let y = arg;
                        assert!(y.detached && !y.requires_grad);
        
                        Some(x + &y)
                    });
                }
            }
            None => todo!(),
        }
        
        None
    }
}
