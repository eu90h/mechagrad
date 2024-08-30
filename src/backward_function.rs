use crate::{function::Function, node::Node, tensor::Tensor, Cell};

#[derive(Clone)]
pub struct BackwardFunction {
    forward_cls: Cell<dyn Function>,
    pub(crate) next_functions: Vec<Cell<Node>>,
    pub(crate) args: Vec<Cell<Tensor>>,
}

impl BackwardFunction {
    pub(crate) fn new(forward_cls: Cell<dyn Function>) -> Self {
        BackwardFunction {
            forward_cls: forward_cls,
            next_functions: vec![],
            args: vec![],
        }
    }
}

impl std::fmt::Debug for BackwardFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackwardFunction").finish()
    }
}

impl Function for BackwardFunction {
    fn forward(&mut self) -> Tensor {
        todo!()
    }

    fn backward(&mut self,_grad_outputs: Vec<Cell<Tensor>>) -> Vec<Tensor> {
        todo!()
    }

    fn apply(&mut self) -> Option<Vec<Tensor>> {
        let mut c = self.forward_cls.borrow_mut();
        return Some(c.backward(self.args.clone()));
    }
}
