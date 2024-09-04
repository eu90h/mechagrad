use std::{cell::RefCell, fmt::Debug, rc::Rc};
use ndarray::prelude::*;
use ndarray::Array;

use crate::{accumulate_grad::AccumulateGrad, backward_function::BackwardFunction, function::Function, node::Node, tensor::Tensor, Cell}; 

/// Pad the edges of an array with zeros.
///
/// `pad_width` specifies the length of the padding at the beginning
/// and end of each axis.
///
/// **Panics** if `arr.ndim() != pad_width.len()`.
/// credit: jturner314, fzyzcjy
/// https://github.com/rust-ndarray/ndarray/issues/823
pub fn pad<A,D>(s: ArcArray<A, D>, pad_width: Vec<[usize; 2]>, const_value: A) -> Array<A, D>
where
    A: Clone,
   // S: ndarray::Data<Elem = A>,
    D: ndarray::Dimension
{
    assert_eq!(
        s.ndim(),
        pad_width.len(),
        "Array ndim must match length of `pad_width`."
    );

    // Compute shape of final padded array.
    let mut padded_shape = s.raw_dim();
    for (ax, (&ax_len, &[pad_lo, pad_hi])) in s.shape().iter().zip(&pad_width).enumerate() {
        padded_shape[ax] = ax_len + pad_lo + pad_hi;
    }

    let mut padded = Array::from_elem(padded_shape, const_value);
    let padded_dim = padded.raw_dim();
    {
        // Select portion of padded array that needs to be copied from the
        // original array.
        let mut orig_portion = padded.view_mut();
        for (ax, &[pad_lo, pad_hi]) in pad_width.iter().enumerate() {
            orig_portion.slice_axis_inplace(
                Axis(ax),
                ndarray::Slice::from(pad_lo as isize..padded_dim[ax] as isize - (pad_hi as isize)),
            );
        }
        // Copy the data from the original array.
        orig_portion.assign(&s);
    }
    padded
}

#[derive(Debug, Clone)]
pub struct Conv2D {
    pub(crate) left: Cell<Tensor>,
    pub(crate) kernel: Cell<Tensor>,
}

impl Function for Conv2D {
    fn forward(&mut self) -> Tensor {
        let a = self.left.borrow();
        let a_data = a.data.clone();
        let kernel = self.kernel.borrow();
        let kernel_data = &kernel.data;
        let mut c= Array::zeros(a.data.shape()).into_dimensionality().unwrap();
        let a_data_padded = pad(a_data.clone(), vec![[1, 1], [1, 1]], 0.0);
        let sh = a_data_padded.shape();
        for i in 0..sh[0]-2 {
            for j in 0..sh[1]-2 {
                let mut conv: f64 = 0.0;
                for k in 0..3 {
                    for l in 0..3 {
                        unsafe {
                            conv += *a_data_padded.uget([i + k, j + l]) * kernel_data.uget([k, l]);
                        }
                    }
                }
                unsafe { *c.uget_mut((i, j)) = conv } 
            }
        }
        Tensor { data: c.into_dyn().into(), 
            requires_grad: a.requires_grad || kernel.requires_grad,
            is_leaf: !a.requires_grad && !kernel.requires_grad,
            grad: Rc::new(RefCell::new(None)),
            grad_fn: None,
            detached: a.detached || kernel.detached,
        }
    }

    fn backward(&mut self, grad_outputs: Vec<Cell<Tensor>>) -> Vec<Tensor> {
        let grad_output = &grad_outputs.get(0).unwrap().borrow().data;
        let a = self.left.borrow();
        let a_data = a.data.clone();
        let kernel = self.kernel.borrow();
        let kernel_data = &kernel.data;
        let mut grad_left_data: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>> = Array::zeros(a.data.shape()).into_dimensionality().unwrap();
        let mut grad_kernel_data: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>> = Array::zeros(kernel.data.shape()).into_dimensionality().unwrap();
        let a_data_padded: ArrayBase<ndarray::OwnedRepr<f64>, Dim<ndarray::IxDynImpl>> = pad(a_data.clone(), vec![[1, 1], [1, 1]], 0.0);
        let sh = a_data_padded.shape();
        for i in 0..sh[0]-2 {
            for j in 0..sh[1]-2 {
                for k in 0..3 {
                    for l in 0..3 {
                        unsafe {
                            if i+k >0 && i + k < sh[0]-1 && j+l>0 && j+l<sh[1]-1 {
                                *grad_kernel_data.uget_mut([k, l]) += *a_data_padded.uget([i + k, j + l]) * grad_output.uget([i + k-1 , j + l - 1]);
                            }
                            if i+k >0 && i + k < sh[0]-1 && j+l>0 && j+l<sh[1]-1 {
                                *grad_left_data.uget_mut([i + k-1, j + l-1]) += kernel_data.uget([k, l])  * grad_output.uget([i + k -1 , j + l-1]);
                             }
                        }
                    }
                }
            }
        }
        if self.left.borrow().requires_grad && self.kernel.borrow().requires_grad {
            let grad_left = Tensor { data: grad_left_data.into_dyn().into(), 
                requires_grad: false,
                is_leaf: true,
                grad: Rc::new(RefCell::new(None)),
                grad_fn: None,
                detached: true,
            };
            let grad_kernel = Tensor { data: grad_kernel_data.into_dyn().into(), 
                requires_grad: false,
                is_leaf: true,
                grad: Rc::new(RefCell::new(None)),
                grad_fn: None,
                detached: true,
            };
            vec![grad_left, grad_kernel]
        } else if self.left.borrow().requires_grad && !self.kernel.borrow().requires_grad {
            let grad_left = Tensor { data: grad_left_data.into_dyn().into(), 
                requires_grad: false,
                is_leaf: true,
                grad: Rc::new(RefCell::new(None)),
                grad_fn: None,
                detached: true,
            };
            vec![grad_left]
        } else if !self.left.borrow().requires_grad && self.kernel.borrow().requires_grad {
            let grad_kernel = Tensor { data: grad_kernel_data.into_dyn().into(), 
                requires_grad: false,
                is_leaf: true,
                grad: Rc::new(RefCell::new(None)),
                grad_fn: None,
                detached: true,
            };
            vec![grad_kernel]
        } else {
            vec![]
        }
    }

    fn apply(&mut self) -> Option<Vec<Tensor>> {
        //create a BackwardFunction obj representing the current node
        let mut backward_function = BackwardFunction::new(Rc::new(RefCell::new(self.clone())));
        //run subclass's forward with context manager and operation input args
        let args = [self.left.clone(), self.kernel.clone()];
        for arg in args.iter()//{
        {
            let oarg = arg.clone();
            let mut arg = arg.borrow_mut();
            if arg.grad_fn.is_none() && !arg.detached {
                if arg.is_leaf && arg.requires_grad {
                    arg.grad_fn = Some(Rc::new(RefCell::new(Node::AccumulateGrad { inner: AccumulateGrad::from(oarg) })))
                } else if arg.is_leaf && !arg.requires_grad {
                    arg.grad_fn = None;
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
    use ndarray_rand::{rand_distr::Uniform, RandomExt};
    use tch::nn::ModuleT;

    use crate::backward::backward;
    use super::*;

    //https://github.com/LaurentMazare/tch-rs/blob/main/examples/mnist/mnist_conv.rs
    #[derive(Debug)]
    struct Net {
        conv1: tch::nn::Conv2D,
    }

    impl Net {
        fn new(vs: &tch::nn::Path) -> Net {
            let mut conf: tch::nn::ConvConfigND<i64> = Default::default();
            conf.padding_mode = tch::nn::PaddingMode::Zeros;
            conf.padding = 1;
            let conv1 = tch::nn::conv2d(vs, 1, 1, 3, conf);
            Net { conv1 }
        }
    }

    impl tch::nn::ModuleT for Net {
        fn forward_t(&self, xs: &tch::Tensor, train: bool) -> tch::Tensor {
            xs.apply(&self.conv1)
        }
    }

    fn check_close(a: Tensor, b: tch::Tensor) {
        let a_shape = a.data.shape();
        let b_shape:Vec<i64> = b.size();
        for (x,y) in zip(a_shape, b_shape) {
            assert_eq!(*x, y as usize);
        }
    
        let a: Vec<f64>  = a.try_into().unwrap();
        let b: Vec<f64> = b.flatten(0, -1).try_into().unwrap();
        for (x,y) in zip(a, b) {
            assert_near!(x, y, 1e-10);
        }
    }

    #[test]
    fn test_conv2d_wikipedia() {
        //Output to test
        ////////////
        let binding = Array::from_iter(vec![5.0, 0.0, 8.0, 7.0, 8.0, 1.0,1.0, 9.0, 5.0, 0.0, 7.0, 7.0, 6.0, 0.0, 2.0, 4.0, 6.0, 6.0, 9.0, 7.0, 6.0, 6.0, 8.0, 4.0, 8.0, 3.0, 8.0, 5.0, 1.0, 3.0, 7.0, 2.0, 7.0, 0.0, 1.0, 0.0]);
        let x = Tensor::from(arr1(&[1.0f64, 1.0, 1.0, 1.0, 1.0, 1.0]).into_dyn().into());

        let m: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>> = binding.clone().to_shape((6, 6)).unwrap().to_owned();
        let mut m = Tensor::from(m.into_dyn().into());
        let mut kernel_array = Array::from_iter([0., 1., 0., 1., -4., 1., 0., 1., 0.]).to_shape((3, 3)).unwrap().to_owned();
        let mut kernel = Tensor::from(kernel_array.clone().into_dyn().into());
        m.requires_grad = true;
        kernel.requires_grad = true;
        let my_output = m.conv2d(&kernel);
        let mut loss = my_output.sum();
        loss.backward();

        //Ground truth
        //////////////
        let vs = tch::nn::VarStore::new(tch::Device::cuda_if_available());
        let mut net = Net::new(&vs.root());

        let py_m = tch::Tensor::from_slice(&binding.to_vec()).to(tch::Device::Cpu).reshape([1,1,6,6]).set_requires_grad(true);
        let py_kernel = tch::Tensor::from_slice(&kernel_array.into_raw_vec()).to(tch::Device::Cpu).reshape([1,1,3,3]).transpose(-1,-1).set_requires_grad(true);
        net.conv1.ws = py_kernel;
        net.conv1.bs = None;

        let py_output = net.forward_t(&py_m, false);
        let py_loss = py_output.sum(None);
        py_loss.backward();
        //The comparison
        ////////////////
        check_close(my_output, py_output.reshape([6,6]));
        check_close(kernel.grad().unwrap(), net.conv1.ws.grad().reshape([3,3]));
        check_close(m.grad().unwrap(), py_m.grad().reshape([6,6]));
    }
}
