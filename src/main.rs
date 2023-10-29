mod node;
use crate::node::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

fn main() {
    let a = val(1.);
    let b = val(2.);
    let c = mul(&a, &a);
    let d = mul(&b, &b);
    let e = add(&c, &d);

    let learning_rate = 0.01;

    for i in 0..1000 {
        println!("i = {}", i);

        let cache = &mut HashMap::new();
        let o = forward(&e, cache);
        println!("output = {}", o);
        println!("cache = {:?}", cache);

        let gradients = &mut HashMap::new();
        backward(&e, cache, gradients, 1.);
        println!("gradients = {:?}", gradients);

        if let Some(NodeOp::Value(ref value)) = a.op {
            let mut value_ref = value.borrow_mut();
            *value_ref -= learning_rate * gradients.get(&a.id).unwrap();
            println!("a = {}", *value_ref);
        }

        if let Some(NodeOp::Value(ref value)) = b.op {
            let mut value_ref = value.borrow_mut();
            *value_ref -= learning_rate * gradients.get(&b.id).unwrap();
            println!("b = {}", *value_ref);
        }
    }
}
