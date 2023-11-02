use crate::nn::*;
use crate::node::*;
use crate::optimizer::*;

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

pub fn test_neural_network() {
    let x = array_to_val(vec![1., 2., 3., 4.], false);
    let _y = array_to_val(vec![4., 1.3, 2.], false);
    let num_epochs = 200;
    let batch_size = 64;
    let learning_rate = 0.1;

    let nn = Network::new();
    let optimizer = Optimizer::new(nn.parameters(), learning_rate);

    for i in 0..num_epochs {
        println!("Epoch {}", i);
        let mut batch_loss = val(0., true);
        for _ in 0..batch_size {
            let y = nn.forward(&x);
            let mut loss = val(0., true);
            for j in 0..y.len() {
                let diff = sub(&y[j], &_y[j]);
                loss = add(&loss, &pow(&diff, 2.));
            }
            batch_loss = add(&batch_loss, &loss);
        }
        batch_loss = div(&batch_loss, &val(batch_size as f32, false));
        batch_loss.backward();
        optimizer.step();
        optimizer.zero_grad();
        println!("Batch loss = {}", batch_loss.output());
    }

    let y = nn
        .forward(&x)
        .iter()
        .map(|x| x.output())
        .collect::<Vec<f32>>();

    println!("y = {:?}", &y);
}

pub fn test_computational_graph() {
    let a = val(3., true);
    let b = val(2., true);

    let target = val(0., false);

    let learning_rate = 0.01;

    for i in 0..1000 {
        let c = pow(&a, 2.);
        let d = pow(&b, 2.);
        let e = add(&c, &d);
        let f = leaky_relu(&e);
        let diff = sub(&f, &target);
        let loss = pow(&diff, 2.);

        loss.backward();
        a.step(learning_rate);
        b.step(learning_rate);
        a.zero_grad();
        b.zero_grad();
    }
    println!("a = {}", a.output());
    println!("b = {}", b.output());
}
