use crate::node::*;

pub struct Optimizer {
    parameters: Vec<NodeRef>,
    learning_rate: f32,
}

impl Optimizer {
    pub fn new(parameters: Vec<NodeRef>, learning_rate: f32) -> Optimizer {
        Optimizer {
            parameters,
            learning_rate,
        }
    }
    pub fn step(&self) {
        for param in &self.parameters {
            param.step(self.learning_rate);
        }
    }
    pub fn zero_grad(&self) {
        for param in &self.parameters {
            param.zero_grad();
        }
    }
}
