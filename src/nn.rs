use crate::node::*;
use rand::Rng;

pub struct Neuron {
    weights: Vec<NodeRef>,
    bias: NodeRef,
}

impl Neuron {
    pub fn new(input_size: usize) -> Neuron {
        let mut rng = rand::thread_rng();
        let mut weights = Vec::new();
        for _ in 0..input_size {
            let w: f32 = rng.gen_range(0..1000) as f32 / 100000.;
            weights.push(val(w, true));
        }
        let b = rng.gen_range(0..1000) as f32 / 100000.;
        let bias = val(b, true);
        Neuron { weights, bias }
    }

    pub fn forward(&self, inputs: &Vec<NodeRef>) -> NodeRef {
        let mut sum = val(0., true);
        for i in 0..self.weights.len() {
            let mul = mul(&self.weights[i], &inputs[i]);
            sum = add(&sum, &mul);
        }
        add(&sum, &self.bias)
    }
}

pub trait Layer {
    fn parameters(&self) -> Vec<NodeRef>;
    fn forward(&self, inputs: &Vec<NodeRef>) -> Vec<NodeRef>;
}

pub struct LinearLayer {
    neurons: Vec<Neuron>,
}

impl LinearLayer {
    pub fn new(input_size: usize, output_size: usize) -> LinearLayer {
        let mut neurons = Vec::new();
        for _ in 0..output_size {
            neurons.push(Neuron::new(input_size));
        }
        LinearLayer { neurons }
    }
}

impl Layer for LinearLayer {
    fn parameters(&self) -> Vec<NodeRef> {
        let mut params = Vec::new();
        for neuron in &self.neurons {
            for weight in &neuron.weights {
                params.push(weight.clone());
            }
            params.push(neuron.bias.clone());
        }
        params
    }

    fn forward(&self, inputs: &Vec<NodeRef>) -> Vec<NodeRef> {
        let mut outputs = Vec::new();
        for neuron in &self.neurons {
            outputs.push(neuron.forward(inputs));
        }
        outputs
    }
}
