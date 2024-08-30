use crate::{accumulate_grad::AccumulateGrad, backward_function::BackwardFunction};

#[derive(Debug, Clone)]
pub enum Node {
    BackwardFunctionWrapper { inner: BackwardFunction },
    AccumulateGrad { inner: AccumulateGrad },
}