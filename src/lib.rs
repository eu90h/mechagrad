mod accumulate_grad;
mod add;
mod backward_function;
mod function;
mod mul;
mod node;
mod matmul;
mod sqrt;
mod transpose;
mod dotprod;
mod reshape;
pub mod tensor;
pub mod backward;
mod relu;
mod log;
mod exp;
mod sum;
mod max;
mod div;
mod pow;
mod sub;

use std::{cell::RefCell, rc::Rc};
///TODO: get rid of this
type Cell<T> = Rc<RefCell<T>>;