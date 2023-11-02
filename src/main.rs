mod nn;
mod node;
mod optimizer;
mod tests;
use crate::tests::*;

fn main() -> () {
    test_computational_graph();
    test_neural_network();
}
