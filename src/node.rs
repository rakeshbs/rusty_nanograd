use rand::Rng;
use std::cell::RefCell;
use std::rc::Rc;

pub type NodeRef = Rc<Node>;

pub struct Node {
    pub output: RefCell<f32>,
    pub grad: RefCell<f32>,
    pub requires_grad: bool,
    backward_fn: Option<Box<dyn Fn(f32)>>,
}

impl Node {
    pub fn new(output: f32, requires_grad: bool, backward_fn: impl Fn(f32) + 'static) -> NodeRef {
        let node = Node {
            output: RefCell::new(output),
            grad: RefCell::new(0.),
            requires_grad,
            backward_fn: Some(Box::new(backward_fn)),
        };
        Rc::new(node)
    }

    pub fn output(&self) -> f32 {
        *self.output.borrow()
    }

    pub fn grad(&self) -> f32 {
        *self.grad.borrow()
    }

    pub fn backward(&self) {
        self.backward_fn.as_ref().unwrap()(1.);
    }

    fn _backward(&self, prev_grad: f32) {
        self.backward_fn.as_ref().unwrap()(prev_grad);
    }

    pub fn step(&self, learning_rate: f32, batch_size: usize) {
        if self.requires_grad {
            let mut output_ref = self.output.borrow_mut();
            *output_ref -= learning_rate * self.grad() / batch_size as f32;
        }
    }
    pub fn zero_grad(&self) {
        let mut grad = self.grad.borrow_mut();
        *grad = 0.;
    }
}

pub fn val(v: f32, requires_grad: bool) -> NodeRef {
    let func = |_: f32| move |_: f32| {};
    Node::new(v, requires_grad, func(v))
}

pub fn array_to_val(v: Vec<f32>, requires_grad: bool) -> Vec<NodeRef> {
    let mut nodes = Vec::new();
    for i in v {
        nodes.push(val(i, requires_grad));
    }
    nodes
}

pub fn add(a: &NodeRef, b: &NodeRef) -> NodeRef {
    let func = |a: &NodeRef, b: &NodeRef| {
        let (a, b) = (a.clone(), b.clone());
        move |prev_grad: f32| {
            let mut ga = a.grad.borrow_mut();
            let mut gb = b.grad.borrow_mut();
            *ga += prev_grad;
            *gb += prev_grad;
            a._backward(*ga);
            b._backward(*gb);
        }
    };

    Node::new(a.output() + b.output(), true, func(&a, &b))
}

pub fn mul(a: &NodeRef, b: &NodeRef) -> NodeRef {
    let func = |a: &NodeRef, b: &NodeRef| {
        let (a, b) = (a.clone(), b.clone());
        move |prev_grad: f32| {
            let mut ga = a.grad.borrow_mut();
            let mut gb = b.grad.borrow_mut();
            let v_a = a.output();
            let v_b = b.output();
            *ga += prev_grad * v_b;
            *gb += prev_grad * v_a;
            a._backward(*ga);
            b._backward(*gb);
        }
    };
    Node::new(a.output() * b.output(), true, func(&a, &b))
}

pub fn sub(a: &NodeRef, b: &NodeRef) -> NodeRef {
    let func = |a: &NodeRef, b: &NodeRef| {
        let (a, b) = (a.clone(), b.clone());
        move |prev_grad: f32| {
            let mut ga = a.grad.borrow_mut();
            let mut gb = b.grad.borrow_mut();
            *ga += prev_grad;
            *gb -= prev_grad;
            a._backward(*ga);
            b._backward(*gb);
        }
    };
    Node::new(a.output() - b.output(), true, func(&a, &b))
}

pub fn pow(a: &NodeRef, b: f32) -> NodeRef {
    let func = |a: &NodeRef, b: &NodeRef| {
        let (a, b) = (a.clone(), b.clone());
        move |prev_grad: f32| {
            let mut ga = a.grad.borrow_mut();
            let v_a = a.output();
            let v_b = b.output();
            *ga += prev_grad * v_b * v_a.powf(v_b - 1.);
            a._backward(*ga);
        }
    };
    Node::new(a.output().powf(b), true, func(&a, &val(b, false)))
}

pub fn exp(a: &NodeRef) -> NodeRef {
    let func = |a: &NodeRef| {
        let a = a.clone();
        move |prev_grad: f32| {
            let mut ga = a.grad.borrow_mut();
            let v_a = a.output();
            *ga += prev_grad * v_a.exp();
            a._backward(*ga);
        }
    };
    Node::new(a.output().exp(), true, func(&a))
}

pub fn tanh(a: &NodeRef) -> NodeRef {
    let func = |a: &NodeRef| {
        let a = a.clone();
        move |prev_grad: f32| {
            let mut ga = a.grad.borrow_mut();
            let v_a = a.output();
            *ga += prev_grad * (1. - v_a.tanh().powi(2));
            a._backward(*ga);
        }
    };
    Node::new(a.output().tanh(), true, func(&a))
}

pub fn relu(a: &NodeRef) -> NodeRef {
    let func = |a: &NodeRef| {
        let a = a.clone();
        move |prev_grad: f32| {
            let mut ga = a.grad.borrow_mut();
            let v_a = a.output();
            *ga += prev_grad * if v_a > 0. { 1. } else { 0. };
            a._backward(*ga);
        }
    };
    Node::new(a.output().max(0.), true, func(&a))
}

pub fn leaky_relu(a: &NodeRef) -> NodeRef {
    let func = |a: &NodeRef| {
        let a = a.clone();
        move |prev_grad: f32| {
            let mut ga = a.grad.borrow_mut();
            let v_a = a.output();
            *ga += prev_grad * if v_a > 0. { 1. } else { 0.01 };
            a._backward(*ga);
        }
    };
    Node::new(a.output().max(0.), true, func(&a))
}
