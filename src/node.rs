use rand::Rng;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

pub type NodeRef = Rc<Node>;

pub enum NodeOp {
    Value(Rc<RefCell<f32>>),
    Add(NodeRef, NodeRef),
    Mul(NodeRef, NodeRef),
    Sub(NodeRef, NodeRef),
}

pub struct Node {
    pub id: usize,
    pub op: Option<NodeOp>,
}

impl Node {
    pub fn new(_op: NodeOp) -> NodeRef {
        let mut rng = rand::thread_rng();
        let _id: usize = rng.gen();
        let node = Node {
            id: _id,
            op: _op.into(),
        };
        Rc::new(node)
    }
}

pub fn val(v: f32) -> NodeRef {
    let node = Node::new(NodeOp::Value(Rc::new(RefCell::new(v))));
    node
}

pub fn add(a: &NodeRef, b: &NodeRef) -> NodeRef {
    let node = Node::new(NodeOp::Add(a.clone(), b.clone()));
    node
}

pub fn mul(a: &NodeRef, b: &NodeRef) -> NodeRef {
    let node = Node::new(NodeOp::Mul(a.clone(), b.clone()));
    node
}

pub fn sub(a: &NodeRef, b: &NodeRef) -> NodeRef {
    let node = Node::new(NodeOp::Sub(a.clone(), b.clone()));
    node
}

pub fn forward(node: &NodeRef, cache: &mut HashMap<usize, f32>) -> f32 {
    if let Some(v) = cache.get(&node.id) {
        return *v;
    }
    let output = match node.op {
        Some(NodeOp::Value(ref v)) => *v.borrow(),
        Some(NodeOp::Add(ref a, ref b)) => {
            let a = forward(&a.clone(), cache);
            let b = forward(&b.clone(), cache);
            a + b
        }
        Some(NodeOp::Mul(ref a, ref b)) => {
            let a = forward(&a.clone(), cache);
            let b = forward(&b.clone(), cache);
            a * b
        }
        Some(NodeOp::Sub(ref a, ref b)) => {
            let a = forward(&a.clone(), cache);
            let b = forward(&b.clone(), cache);
            a - b
        }
        None => 0.,
    };
    cache.insert(node.id, output);
    output
}

pub fn backward(
    node: &NodeRef,
    cache: &mut HashMap<usize, f32>,
    gradients: &mut HashMap<usize, f32>,
    prev_grad: f32,
) {
    gradients.insert(node.id, prev_grad);
    match node.op {
        Some(NodeOp::Value(_)) => {
            //gradients.insert(node.id, 1.);
        }
        Some(NodeOp::Add(ref a, ref b)) => {
            backward(&a.clone(), cache, gradients, prev_grad);
            backward(&b.clone(), cache, gradients, prev_grad);
        }
        Some(NodeOp::Mul(ref a, ref b)) => {
            let v_a = cache.get(&a.id).unwrap();
            let v_b = cache.get(&b.id).unwrap();
            let grad_a = prev_grad * v_b;
            let grad_b = prev_grad * v_a;
            backward(&a.clone(), cache, gradients, grad_a);
            backward(&b.clone(), cache, gradients, grad_b);
        }
        Some(NodeOp::Sub(ref a, ref b)) => {
            backward(&a.clone(), cache, gradients, prev_grad);
            backward(&b.clone(), cache, gradients, -prev_grad);
        }
        None => {}
    };
}
