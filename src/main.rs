mod nn;
mod node;
mod optimizer;
mod tests;
use crate::nn::*;
use crate::node::*;
use crate::optimizer::*;
use crate::tests::*;

fn main() -> () {
    test_computational_graph();
    test_neural_network();
}

struct Network {
    l1: LinearLayer,
    l2: LinearLayer,
}

impl Network {
    pub fn new() -> Network {
        let l1 = LinearLayer::new(4, 10);
        let l2 = LinearLayer::new(10, 3);
        Network { l1, l2 }
    }

    pub fn forward(&self, inputs: &Vec<NodeRef>) -> Vec<NodeRef> {
        let mut x = self.l1.forward(&inputs);
        x = x.iter().map(|x| relu(&x)).collect();
        x = self.l2.forward(&x);
        x
    }

    pub fn parameters(&self) -> Vec<NodeRef> {
        let mut params = Vec::new();
        params.append(&mut self.l1.parameters());
        params.append(&mut self.l2.parameters());
        return params;
    }
}
