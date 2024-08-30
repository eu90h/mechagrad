# mechagrad
Mechagrad is a Rust implementation of reverse-mode automatic differentiation heavily inspired by MyTorch/PyTorch


# Outline
This readme details the design and implementation of a simple automatic differentiation (AD for short, or, sometimes, autograd) system meant as an educational exercise.
I'm new to machine learning (and Rust) and thought it would be fun to build everything from scratch (in tandem with reading Kevin Murphy's Probabilistic
Machine Learning). 

Models are typically fit via minimization of a scalar-valued loss function -- hence the need to compute gradients.
This makes the automatic differentiation system a natural starting point, given its central role in learning.

There's lots of resources out there on automatic differentiation, particularly the oft-cited book "Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation" by Griewank, but I opted to study actually-existing implementations.

------

First let's take a look at Andrej Karpathy's micrograd. This small library shows the essence of an AD implementation.
The Value object is central here. It wraps a floating-point value, providing storage for a gradient, along with wrapped arithmetical operations
that operate on other values, working to dynamically build a graph of the expression being computed.

When we're building the graph of the computed expression, we actually compute the value of the expression at the same time. This process is known
as the forward pass. We can take the resulting graph and perform a backward pass in order to compute derivatives. This is embodied in the Value's backward method.
First, the graph is topologically sorted into a list of nodes, the gradient of that final node is set to 1, and then each node in the reversed topologically-sorted
node list has its backward_ function called, which is actually responsible for computing the derivatives.

It's worth noting that the topological sort is essentially a reverse-order depth-first-search on the computational graph (viewed as a directed acyclic graph).
As it happens, this guarantees that at least one path through the graph exists that visits every node in linear time.

The Value object stores a few pieces of data relating to the computational graph. We have _backward, which stores a function responsible for the actual computation of
derivatives. There is also _prev which stores that nodes children as a set. In the computation, for instance, of "x + y", a node corresponding to the result of the addition operation
is constructed with Values x and y as its children. The final field is _op which is mainly for human convenience, merely denoting (as a string) 
the operation that produced that Value.

Let's look at how one of the derivatives are computed. The _backward function for add looks like
def _backward():
    self.grad += out.grad
    other.grad += out.grad
This closure captures self and other, the two children of that add node, and gets stored in the result's _backward field.

It's fairly easy now to imagine how this all fits together. We call backward, which sets off a chain reaction of calls to different node's _backward functions.
These set the .grad field of their children, which then get their _backward called, which continues pushing the grads around, etc., until we've traversed the whole graph.

Karpathy's micrograd is delightfully simple and shows the entire essence of a AD engine. 
------
Now let's take a look at a look at the MyTorch project for the fall 2020 semester of CMU's 11-785 Introduction to Deep Learning.
This, as the name hints, intends to replicate a reasonable subset of the widely-used PyTorch library.

As with micrograd, MyTorch revolves around a central wrapper object called a Tensor, which in this case wraps over a numpy ndarray instead of a scalar float.
A micrograd _backward is a MyTorch Tensor's grad_fn field. Tensors need to be explicitly marked as requiring gradients by setting the requires_grad field to True.
A Tensor may also be a leaf node, which means that none of its parents require gradients. This status is tored in the is_leaf field.
The combination of requires_grad and is_leaf determines what that Tensor is doing in the computational graph:

* AccumulateGrad nodes (is_leaf=True, requires_grad=True). This node does what is says -- it adds a given value to its associated tensor's grad field.
* BackwardFunction nodes. (is_leaf=False, requires_grad=True). These nodes calculate gradients and send them out to associated nodes, much like the _backward function
for add in micrograd seen above.
* Constant nodes (is_leaf=True, requires_grad=False). Just a tensor that doesn't need a gradient and wasn't created by a series of operations. Think input data.

We can see that MyTorch is a bit more explicit about the computational graph. Now we'll look at MyTorch's Add function.
Operations are represented as objects that implement a particular interface derived from the Function class that looks something like:
* forward(ctx, args...)
* backward(ctx, grad_output)
* apply(ctx, args...)

Forward methods implement the forward pass. They're responsible for computing the output Tensor as well as storing any data required for the backward pass in
the context (ctx) object.
The apply method is typically the same for all operations, being inherited from the Function class.
The job of apply is to do the forward pass and actually perform the construction of the computational graph.
The backward method is directly analogous to micrograd's _backward functions.

Notice how micrograd avoids the need for a context object by using a closure in the forward pass to capture the information needed later.

Finally, we'll consider the backward function at the heart of the backward pass (the Tensor's backward method just sets the grad to 1 and calls this).
The MyTorch backward function is left as an exercise in that class, but it's simple enough: we perform a depth-first search on the graph, calling backward on each node
we come across. There is no explicit reversal/sorting of the graph in the case of MyTorch, rather, it seems that graph is in the proper format by construction.

-------

At this point, I felt ready to dive in and try my hand at writing an AD engine. I chose to use the Rust language for this project and decided to mimic the architecture
laid out by MyTorch.

My AD engine, Mechagrad, hews to MyTorch pretty closely. It wraps an N-dimensional array, helpfully provided by the ndarray crate.

Exactly like PyTorch, I override the various arithmetical operations on Tensors so as to build up a computational graph.
Consider the addition of two tensors: x + y.

```
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
```

This shows a general pattern for the library. Internally, basic operations on tensors are represented by structs that implement the Function trait:

```
pub(crate) trait Function {
    ///Computes value and links nodes to the computational graph.
    fn forward(&mut self) -> Tensor;
    ///Computes derivatives.
    fn backward(&mut self, grad_outputs: Vec<Cell<Tensor>>) -> Vec<Tensor>;
    ///Runs forward of subclass and links node to the computational graph.
    fn apply(&mut self) -> Option<Vec<Tensor>>;
}
```

The forward function is typically simple. For Add this looks like

```
fn forward(&mut self) -> Tensor {
    let a = self.left.clone();
    let b = self.right.clone();
    let a = a.borrow_mut();
    let b = b.borrow_mut();
    
    Tensor {
        data: a.data.clone() + b.data.clone(),
        requires_grad: a.requires_grad || b.requires_grad,
        is_leaf: !(a.requires_grad || b.requires_grad),
        grad: Rc::new(RefCell::new(None)),
        grad_fn: None,
        detached: a.detached || b.detached,
    }
}
```

It merely breaks the tensors apart and adds their guts while ensuring that the resulting tensor has its flags correctly set.
The logic of the flags is simple enough:
A tensor requires a gradient if any of its inputs require gradients.
A tensor isn't a leaf tensor if one or more of its parents require gradients.
A tensor is detached if one or more of its parents are detached.

The backward function can be a bit messy, but in this case is relatively simple:

```
fn backward(&mut self, grad_outputs: Vec<Cell<Tensor>>) -> Vec<Tensor> {
    let (mut grad_a, mut grad_b) = {
        let gradient = grad_outputs.get(0).unwrap().borrow_mut();
        let gradient = g.data.clone();
        let a = &self.left;
        let b = &self.right;
        let a = a.borrow_mut();
        let b = b.borrow_mut();

        (ndarray::Array::ones(a.data.shape()) * gradient.clone(), ndarray::Array::ones(b.data.shape()) * gradient.clone())
    };
    //rest of function excluded
}
```

We see here the computation of the derivatives with respect to both arguments of the sum x + y. The excluded part mainly is responsible for handling some reshaping,
bundling the result into a new Tensor, and setting flags appropriately for the gradient tensor.

```
let grad_a = Tensor {
    data: grad_a.into_dimensionality().unwrap().into(), 
    requires_grad: false,
    is_leaf: true,
    grad: Rc::new(RefCell::new(None)),
    grad_fn: None,
    detached: true,
};
```
This shows a fairly typical example of creating the gradient tensor object. Gradients never require gradients, are always leaves, and are always detached.

The apply function is quite similar across operations. Here is the apply function for Add:

```
fn apply(&mut self) -> Option<Vec<Tensor>> {
    let mut backward_function = BackwardFunction::new(Rc::new(RefCell::new(self.clone())));
    let args = [self.left.clone(), self.right.clone()];
    for arg in args.iter() {
        let oarg = arg.clone();
        let mut arg = arg.borrow_mut();
        if arg.grad_fn.is_none() && !arg.detached {
            if arg.is_leaf && arg.requires_grad {
                arg.grad_fn = Some(Rc::new(RefCell::new(Node::AccumulateGrad { inner: AccumulateGrad::from(oarg) })))
            }
        }
        let c = arg.grad_fn.clone();
        if c.is_some() && !arg.detached {
            backward_function.next_functions.push(c.unwrap());
        }
    }
    let mut output_tensor = self.forward();
    if !output_tensor.detached {
        output_tensor.grad_fn = Some(Rc::new(RefCell::new(Node::BackwardFunctionWrapper { inner: backward_function })));
    }
    Some(vec![output_tensor])
}
```

We see here that apply performs the critical job of creating computational graph Node objects and storing them in that operation's argument's grad_fn field.
Inputs accumulate gradients while operations compute them and pass them back.


When we've built up some scalar expression out of Tensors, we can retrieve the gradient by calling `.backward()` on the scalar-valued output tensor.
BackwardFunction objects are simple -- they just call the backward method of their given arg:

```
fn apply(&mut self) -> Option<Vec<Tensor>> {
    let mut c = self.forward_cls.borrow_mut();
    return Some(c.backward(self.args.clone()));
}
```

this results in the actual computation of the gradient.

AccumulateGrad objects are also simple -- they just update the grad field of the tensor held in the variable field.

```
fn apply(&mut self) -> Option<Vec<Tensor>> {
    let grad_is_none = self.variable.borrow_mut().grad.as_ref().borrow().is_none();
    if grad_is_none {
        let variable = self.variable.borrow_mut();
        let a = self.args.get(0).unwrap().borrow_mut();
        let arg = a.clone();
        variable.grad.replace(Some(arg));
    } else {
        let variable = self.variable.borrow_mut();
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
    None
}
```

So much for the outline of the Add operation. Similar objects are implemented for other basic operations including multiplication, 
matrix multiplication, dot product, division, exp, log, etc.

Accuracy of the result is of paramount importance. It's hard to train models if the gradients are subtly wrong, so we need a good testing methodology.
I settled on the idea of using the well-vetted torch library, or more particularly its Rust bindings `tch`, as a source of ground truth by which to compare results.

The idea is that we build the same expression in both my Tensor implementation and in torch, call the backward function, and then compare the gradients computed.

As a capstone test, I created a feedforward neural network with ReLU activation and trained it to label digits in the MNIST set. I tasked torch with the same. At every
step of the training of the network, I compare the gradients computed by my implementation and torch, ensuring that they agree to some determined precision.
