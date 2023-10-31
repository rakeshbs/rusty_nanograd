mod nn;
mod node;
mod optimizer;
mod sine_generator;
use crate::nn::*;
use crate::node::*;
use crate::optimizer::*;

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

fn test_neural_network() {
    let nn = Network::new();
    let x = array_to_val(vec![1., 2., 3., 4.], false);
    let _y = array_to_val(vec![-10., 20.54553, 30.], false);
    let optimizer = Optimizer::new(nn.parameters(), 0.1, 1);

    for _ in 0..100 {
        let y = nn.forward(&x);
        let mut loss = val(0., false);
        for j in 0..y.len() {
            let diff = sub(&y[j], &_y[j]);
            loss = add(&loss, &pow(&diff, 2.));
        }
        loss.backward();
        optimizer.step();
        optimizer.zero_grad();
    }

    let y = nn
        .forward(&x)
        .iter()
        .map(|x| x.output())
        .collect::<Vec<f32>>();

    println!("y = {:?}", &y);
}

fn test_computational_graph() {
    let a = val(1., true);
    let b = val(2., true);

    let learning_rate = 0.01;

    for i in 0..10 {
        let c = pow(&a, 2.);
        let d = pow(&b, 2.);
        let e = add(&c, &d);
        let f = tanh(&e);
        let o = f.output();

        e.backward();
        //println!("gradients = {:?}", gradients);

        a.step(learning_rate);
        b.step(learning_rate);
        a.zero_grad();
        b.zero_grad();
    }
    println!("a = {}", a.output());
    println!("b = {}", b.output());
}
