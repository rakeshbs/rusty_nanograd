use crate::node::*;

pub struct Optimizer {
    parameters: Vec<NodeRef>,
    learning_rate: f32,
    batch_size: usize,
}

impl Optimizer {
    pub fn new(parameters: Vec<NodeRef>, learning_rate: f32, batch_size: usize) -> Optimizer {
        Optimizer {
            parameters,
            learning_rate,
            batch_size,
        }
    }
    pub fn step(&self) {
        for param in &self.parameters {
            param.step(self.learning_rate / self.batch_size as f32);
        }
    }
    pub fn zero_grad(&self) {
        for param in &self.parameters {
            param.zero_grad();
        }
    }
}
