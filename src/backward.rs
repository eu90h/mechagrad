#![allow(unused)]
use std::{cell::RefCell, collections::HashMap, rc::Rc};

use crate::{function::Function, node::Node, tensor::Tensor, Cell};

fn backward_aux(grad_fn: Option<Cell<Node>>, grad_of_outputs: Cell<Tensor>) {
    match grad_fn {
        Some(grad_fn_node) => {
            let g = grad_fn_node.borrow_mut().clone();
            let (grads, inner_functions) = match g {
                Node::AccumulateGrad { mut inner } => {
                    let v = vec![grad_of_outputs.clone()];
                    inner.args = v;
                    (inner.apply(), inner.next_functions)
                },
                Node::BackwardFunctionWrapper { mut inner } => {
                    let v = vec![grad_of_outputs];
                    inner.args = v;
                    (inner.apply(), inner.next_functions)
                }
            };
            if grads.is_some() {
                let grads = grads.unwrap();
                for (node, grad) in std::iter::zip(inner_functions, grads) {
                    backward_aux( Some(node), Rc::new(RefCell::new(grad)))
                }
            }
        }
        None => {}
    }
}

pub fn backward(output: &mut Tensor) {
    let x = Tensor::ones_like(output);
    backward_aux( output.grad_fn.clone(), Rc::new(RefCell::new(x)));
}