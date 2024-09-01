use crate::{tensor::Tensor, Cell};

pub(crate) trait Function {
    ///Links nodes to the computational graph.
    fn forward(&mut self) -> Tensor;
    fn backward(&mut self, grad_outputs: Vec<Cell<Tensor>>) -> Vec<Tensor>;
    ///Runs forward of subclass and links node to the computational graph.
    fn apply(&mut self) -> Option<Vec<Tensor>>;
}