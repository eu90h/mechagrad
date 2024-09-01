use std::{borrow::BorrowMut, cell::RefCell, rc::Rc};
use ndarray::{ArcArray, IxDyn};
use num_traits::Inv;
use crate::div::Div;
use crate::exp::Exp;
use crate::log::Log;
use crate::max::Max;
use crate::pow::Pow;
use crate::relu::ReLU;
use crate::sub::Sub;
use crate::sum::Sum;
use crate::{add::Add, backward::backward, dotprod::DotProduct, function::Function, matmul::MatMul, mul::Mul, node::Node, reshape::Reshape, sqrt::Sqrt, transpose::Tranpose, Cell};

#[derive(Clone)]
pub struct Tensor {
    pub data: ArcArray<f64, IxDyn>,
    pub requires_grad: bool,
    pub is_leaf: bool,
    pub(crate) grad: Cell<Option<Tensor>>,
    pub(crate) grad_fn: Option<Cell<Node>>,
    pub(crate) detached: bool,
}

impl Tensor {
    pub fn grad(&self) -> Option<Tensor> {
        let g = self.grad.as_ref();
        match g.clone().into_inner() {
            Some(t) => Some(t),
            None => None
        }
    }

    pub fn zero_grad(&mut self) {
        self.grad_fn = None;
        self.grad.replace(None);
    }

    pub fn detach(&mut self) {
        self.detached = true;
    }
    
    pub fn reattach(&mut self) {
        self.detached = false;
    }

    pub fn from(data: ArcArray<f64, IxDyn>) -> Self {
        Tensor {
            data,
            grad: Rc::new(RefCell::new(None)),
            requires_grad: false,
            is_leaf: true,
            grad_fn: None,
            detached: false,
        }
    }

    pub fn zeros(shape: &[usize]) -> Tensor {
        Tensor {
            data: ndarray::ArcArray::zeros(shape).into_dimensionality().unwrap(),
            requires_grad: false,
            is_leaf: true,
            grad: Rc::new(RefCell::new(None)),
            grad_fn: None,
            detached: false,
        }
    }

    pub fn ones_like(other: &Tensor) -> Tensor {
        Tensor {
            data: ndarray::ArcArray::ones(other.data.shape()).into_dimensionality().unwrap(),
            requires_grad: false,
            is_leaf: true,
            grad: Rc::new(RefCell::new(None)),
            grad_fn: None,
            detached: false,
        }
    }

    /*pub fn add(lhs: Cell<Tensor>, rhs: Cell<Tensor>) -> Tensor {
        let mut add = Add {
            left: lhs,
            right: rhs,
        };
        let result = add.apply();
        result.unwrap().get(0).unwrap().clone()
    }

    pub fn mul(lhs: Cell<Tensor>, rhs: Cell<Tensor>) -> Tensor {
        let mut mul = Mul {
            left: lhs,
            right: rhs,
        };
        let result = mul.apply();
        result.unwrap().get(0).unwrap().clone()
    }*/

    pub fn matmul(&self, rhs: &Tensor) -> Tensor {
        let mut mul = MatMul {
            left: Rc::new(RefCell::new(self.clone())),
            right: Rc::new(RefCell::new(rhs.clone())),
            right_is_vec: false,
        };
        let result = mul.apply();
        result.unwrap().get(0).unwrap().clone()
    }

    pub fn relu(&self) -> Tensor {
        let mut mul = ReLU {
            left: Rc::new(RefCell::new(self.clone())),
        };
        let result = mul.apply();
        result.unwrap().get(0).unwrap().clone()
    }

 

    pub fn dot(&self, rhs: &Tensor) -> Tensor {
        let mut mul = DotProduct {
            left: Rc::new(RefCell::new(self.clone())),
            right: Rc::new(RefCell::new(rhs.clone())),
        };
        let result = mul.apply();
        result.unwrap().get(0).unwrap().clone()
    }

    pub fn transpose(&self) -> Tensor {
        let mut mul = Tranpose {
            left: Rc::new(RefCell::new(self.clone())),
        };
        let result = mul.apply();
        result.unwrap().get(0).unwrap().clone()
    }

    pub fn sqrt(&self) -> Tensor {
        let mut mul = Sqrt {
            left: Rc::new(RefCell::new(self.clone())),
        };
        let result = mul.apply();
        result.unwrap().get(0).unwrap().clone()
    }

    pub fn log(&self) -> Tensor {
        let mut mul = Log {
            left: Rc::new(RefCell::new(self.clone())),
        };
        let result = mul.apply();
        result.unwrap().get(0).unwrap().clone()
    }

    pub fn exp(&self) -> Tensor {
        let mut mul = Exp {
            left: Rc::new(RefCell::new(self.clone())),
        };
        let result = mul.apply();
        result.unwrap().get(0).unwrap().clone()
    }

    pub fn powi(&self, n: i32) -> Tensor {
        let mut mul = Pow {
            left: Rc::new(RefCell::new(self.clone())),
            n
        };
        let result = mul.apply();
        result.unwrap().get(0).unwrap().clone()
    }

    pub fn square(&self) -> Tensor {
        self.powi(2)
    }

    pub fn reshape(&self, shape: &[usize]) -> Tensor {
        let mut mul = Reshape {
            left: Rc::new(RefCell::new(self.clone())),
            shape: shape.to_owned(),
            old_shape: self.data.shape().to_owned(),
        };
        let result = mul.apply();
        result.unwrap().get(0).unwrap().clone()
    }

    pub fn grad_step(&mut self, k: f64) {
        let grad = self.grad.borrow().is_some();
        if grad {
            self.detach();
            let g = self.grad.as_ref().borrow().borrow_mut().clone().unwrap().data;
            self.data = self.data.clone() - (g*k);
            self.reattach();
        }
    }

    pub fn backward(&mut self) {
        backward(self)
    }

    //TODO: should this be a node?
    pub fn max(&self) -> Tensor {
        let d = self.data.flatten();
        let mut result = 0;
        let mut max = d.get(0).unwrap();
        for i in 0..d.len() {
            if d.get(i).unwrap() > max {
                max = d.get(i).unwrap();
                result = i;
            }
        }
        let mut mul = Max {
            left: Rc::new(RefCell::new(self.clone())),
            argmax: result
        };
        let result = mul.apply();
        result.unwrap().get(0).unwrap().clone()
    }
    
    //TODO: should this be a node?
    pub fn argmax(&self) -> usize {
        let d = self.data.flatten();
        let mut result = 0;
        let mut max = d.get(0).unwrap();
        for i in 0..d.len() {
            if d.get(i).unwrap() > max {
                max = d.get(i).unwrap();
                result = i;
            }
        }
        return result;
    }

    pub fn sum(&self) -> Tensor {
        let mut mul = Sum {
            left: Rc::new(RefCell::new(self.clone())),
        };
        let result = mul.apply();
        result.unwrap().get(0).unwrap().clone()
    }

    pub fn logsoftmax(&mut self) -> Tensor {
        let x = self.clone();
        let b = x.max();
        let delta = x - b;
        let l = delta.exp().sum().log();
        delta -  l
    }

    pub fn mean(&self) -> Tensor {
        let d = self.data.flatten();
        let n = d.len();
        self.sum() / (n as f64)
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.data.as_ptr() == other.data.as_ptr()
    }
}

impl Eq for Tensor {}

impl std::hash::Hash for Tensor {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.as_ptr().hash(state);
        self.requires_grad.hash(state);
        self.is_leaf.hash(state);
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
        .field("data", &self.data)
        .field("requires_grad", &self.requires_grad)
        .field("is_leaf", &self.is_leaf)
        .field("grad", &self.grad)
        .field("detached", &self.detached)
        .finish()
    }
}
impl std::ops::Sub<&Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: &Tensor) -> Self::Output {
        let mut add = Sub {
            left: Rc::new(RefCell::new(self.clone())),
            right: Rc::new(RefCell::new(rhs.clone())),
        };
        let result = add.apply();
        result.unwrap().get(0).unwrap().clone()
    }
}
/*impl std::ops::Sub<f64> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: f64) -> Self::Output {
        self + Tensor::from(arr1(&[(rhs.clone() * -1.0)]).into_dyn().into())
    }
}
impl std::ops::Sub<f64> for &mut Tensor {
    type Output = Tensor;

    fn sub(self, rhs: f64) -> Self::Output {
        self + &mut Tensor::from(arr1(&[(rhs.clone() * -1.0)]).into_dyn().into())
    }
}*/
impl std::ops::Add<&Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: &Self) -> Self::Output {
        let mut add = Add {
            left: Rc::new(RefCell::new(self)),
            right: Rc::new(RefCell::new(rhs.clone())),
        };
        let result = add.apply();
        result.unwrap().get(0).unwrap().clone()
    }
}
impl std::ops::Add for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Self::Output {
        let mut add = Add {
            left: Rc::new(RefCell::new(self)),
            right: Rc::new(RefCell::new(rhs)),
        };
        let result = add.apply();
        result.unwrap().get(0).unwrap().clone()
    }
}

impl std::ops::Add<&mut Tensor> for &mut Tensor {
    type Output = Tensor;

    fn add(self, rhs: &mut Tensor) -> Self::Output {
        let mut add = Add {
            left: Rc::new(RefCell::new(self.clone())),
            right: Rc::new(RefCell::new(rhs.clone())),
        };
        let result = add.apply();
        result.unwrap().get(0).unwrap().clone()
    }
}

impl std::ops::Add<&Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: &Tensor) -> Self::Output {
        let mut add: Add = Add {
            left: Rc::new(RefCell::new(self.clone())),
            right: Rc::new(RefCell::new(rhs.clone())),
        };
        let result = add.apply();
        result.unwrap().get(0).unwrap().clone()
    }
}

impl std::ops::Mul for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut mul = Mul {
            left: Rc::new(RefCell::new(self.clone())),
            right: Rc::new(RefCell::new(rhs.clone())),
        };
        let result = mul.apply();
        result.unwrap().get(0).unwrap().clone()
    }
}

impl std::ops::Mul<&mut Tensor> for &mut Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &mut Tensor) -> Self::Output {
        let mut mul = Mul {
            left: Rc::new(RefCell::new(self.clone())),
            right: Rc::new(RefCell::new(rhs.clone())),
        };
        let result = mul.apply();
        result.unwrap().get(0).unwrap().clone()
    }
}
impl std::ops::Mul<&Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Self::Output {
        let mut mul = Mul {
            left: Rc::new(RefCell::new(self.clone())),
            right: Rc::new(RefCell::new(rhs.clone())),
        };
        let result = mul.apply();
        result.unwrap().get(0).unwrap().clone()
    }
}
impl std::ops::Mul<&Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Self::Output {
        let mut mul = Mul {
            left: Rc::new(RefCell::new(self.clone())),
            right: Rc::new(RefCell::new(rhs.clone())),
        };
        let result = mul.apply();
        result.unwrap().get(0).unwrap().clone()
    }
}
impl std::ops::Mul<Rc<RefCell<Tensor>>> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Rc<RefCell<Tensor>>) -> Self::Output {
        let lhs = Rc::new(RefCell::new(self));
        let mut mul = Mul {
            left: lhs.clone(),
            right: rhs.clone(),
        };
        let result = mul.apply();
        result.unwrap().get(0).unwrap().clone()
    }
}

impl std::ops::Mul<f64> for Tensor {
    type Output = Tensor;

    fn mul(mut self, rhs: f64) -> Self::Output {
        let rhs = {
            let binding = self.data.borrow_mut();
            let binding = binding.to_owned();
            let sh = binding.shape();
            ArcArray::ones(sh) * rhs
        };

        let mut rhs = Tensor::from(rhs.into_dimensionality().unwrap());
        rhs.is_leaf = true;
        rhs.requires_grad = false;
        rhs.detached = self.detached;
        let mut mul = Mul {
            left: Rc::new(RefCell::new(self)),
            right: Rc::new(RefCell::new(rhs)),
        };
        let result = mul.apply();
        result.unwrap().get(0).unwrap().clone()
    }
}

impl std::ops::Div<f64> for Tensor {
    type Output = Tensor;

    fn div(mut self, rhs: f64) -> Self::Output {
        let rhs = {
            let binding = self.data.borrow_mut();
            let binding = binding.to_owned();
            let sh = binding.shape();
            ArcArray::ones(sh) * rhs.inv()
        };

        let mut rhs = Tensor::from(rhs.into_dimensionality().unwrap());
        rhs.is_leaf = true;
        rhs.requires_grad = false;
        rhs.detached = self.detached;
        let mut mul = Mul {
            left: Rc::new(RefCell::new(self)),
            right: Rc::new(RefCell::new(rhs)),
        };
        let result = mul.apply();
        result.unwrap().get(0).unwrap().clone()
    }
}

impl std::ops::Div<Tensor> for Tensor {
    type Output = Tensor;

    fn div(self, rhs: Tensor) -> Self::Output {
        let mut mul = Div {
            left: Rc::new(RefCell::new(self)),
            right: Rc::new(RefCell::new(rhs)),
        };
        let result = mul.apply();
        result.unwrap().get(0).unwrap().clone()
    }
}

impl std::ops::Div<&Tensor> for &Tensor {
    type Output = Tensor;

    fn div(self, rhs: &Tensor) -> Self::Output {
        let mut mul = Div {
            left: Rc::new(RefCell::new(self.clone())),
            right: Rc::new(RefCell::new(rhs.clone())),
        };
        let result = mul.apply();
        result.unwrap().get(0).unwrap().clone()
    }
}

impl std::ops::Div<&Tensor> for Tensor {
    type Output = Tensor;

    fn div(self, rhs: &Tensor) -> Self::Output {
        let mut mul = Div {
            left: Rc::new(RefCell::new(self.clone())),
            right: Rc::new(RefCell::new(rhs.clone())),
        };
        let result = mul.apply();
        result.unwrap().get(0).unwrap().clone()
    }
}

impl std::ops::Sub for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut add = Sub {
            left: Rc::new(RefCell::new(self.clone())),
            right: Rc::new(RefCell::new(rhs.clone())),
        };
        let result = add.apply();
        result.unwrap().get(0).unwrap().clone()
    }
}

impl std::ops::Sub for &mut Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut add = Sub {
            left: Rc::new(RefCell::new(self.clone())),
            right: Rc::new(RefCell::new(rhs.clone())),
        };
        let result = add.apply();
        result.unwrap().get(0).unwrap().clone()
    }
}

impl std::ops::Sub for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut add = Sub {
            left: Rc::new(RefCell::new(self.clone())),
            right: Rc::new(RefCell::new(rhs.clone())),
        };
        let result = add.apply();
        result.unwrap().get(0).unwrap().clone()
    }
}

impl TryInto<Vec<f64>> for Tensor {
    //TODO: return a real error
    type Error = i64;

    fn try_into(self) -> Result<Vec<f64>, Self::Error> {
        let (mytorch, _) = self.data.to_owned().into_raw_vec_and_offset();
        return Ok(mytorch);
    }
}

#[allow(non_snake_case, unused)]
#[cfg(test)]
mod tests {
    use std::iter::zip;

    use all_asserts::assert_near;
    use ndarray::Array;
    use ndarray_rand::{rand_distr::Uniform, RandomExt};
    use tch::no_grad;
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
            assert_near!(x, y, 1e-3);
        }
    }

    #[test]
    fn test_logsumexp() {
        for _ in 0..1000 {
        //Output to test
        ////////////
        let A = Array::random((3,5), Uniform::new(-188., 188.));
        let mut W = Tensor::from(A.clone().into_dyn().into());
        let mut z = vec![0.0, 0.0, 0.0];
        let v: usize = 1;
        z[v as usize] = 1.0;
        let rhs = Tensor::from(ArcArray::from_vec(z.clone()).into_dyn().into());


        let C = Array::random(5, Uniform::new(-10., 10.));
        let x = Tensor::from(C.clone().into_dyn().into());
        W.requires_grad = true;     
        let mut my_output = W.matmul(&x);
        let my_output2 = my_output.logsoftmax();
        let mut my_output3 = my_output2.dot(&rhs);
        my_output3.backward();

        //Ground truth
        //////////////
        let (A, l) = A.into_raw_vec_and_offset();
        let (C, l) = C.into_raw_vec_and_offset();
        let pyX = tch::Tensor::from_slice(&C).to(tch::Device::Cpu);
        let pyW = tch::Tensor::from_slice(&A).reshape(&[3,5]).set_requires_grad(true).to(tch::Device::Cpu);

        let y = pyW.matmul(&pyX);
        let pyout2 = y.log_softmax(-1, None);
        no_grad(|| {
            let gt: Vec<f64> = pyout2.flatten(0, -1).try_into().unwrap();
         });

        let rhs = tch::Tensor::from_slice(&z);
        let pyout3 = pyout2.dot(&rhs);
        pyout3.backward();
       
        let ground_truth = pyW.grad();

        //The comparison
        ////////////////
        check_close(W.grad().unwrap(), ground_truth);
        }
    }

    #[test]
    fn test_mean() {
        //Output to test
        ////////////
        let A = Array::random(7280, Uniform::new(0.01, 100.));

        let mut x = Tensor::from(A.clone().into_dyn().into());
        x.requires_grad = true;
        let mut my_output = x.mean();
        my_output.backward();
        let my_truth = x.grad().unwrap();

        //Ground truth
        //////////////
        let (A, l) = A.into_raw_vec_and_offset();
        let t = tch::Tensor::from_slice(&A).reshape(&[7280]).set_requires_grad(true).to(tch::Device::Cpu);
        let z = t.mean(None);
        z.backward();
        let ground_truth = t.grad();

        //The comparison
        ////////////////
        check_close(my_output, z);
        check_close(my_truth, ground_truth);
    }

}