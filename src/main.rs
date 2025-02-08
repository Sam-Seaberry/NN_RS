use nalgebra::{DMatrix};
use rand::prelude::*;
use std::env;

fn main() {
    env::set_var("RUST_BACKTRACE", "1");

    let mut rng = rand::thread_rng();
    let layers = 3;
    let neurons = vec![2, 3, 3, 1];
    let W1 = xavier_init(neurons[1], neurons[0]);
    let W2 = xavier_init(neurons[2], neurons[1]);
    let W3 = xavier_init(neurons[3], neurons[2]);
    let mut weights: Vec<DMatrix<f32>> = vec![W1, W2, W3];
    
    let b1 = xavier_init(neurons[1], 1);
    let b2 = xavier_init(neurons[2], 1);
    let b3 = xavier_init(neurons[3], 1);
    let mut bias: Vec<DMatrix<f32>> = vec![b1, b2, b3];

    let x = DMatrix::from_row_slice(4, 2, &[
        0.0, 0.0, 
        1.0, 0.0, 
        1.0, 1.0, 
        0.0, 1.0
    ]);

    let y = DMatrix::from_row_slice(1, 4, &[
        0.0, 1.0, 0.0, 1.0
    ]);
    
    let m = 10;
    let A0 = x.transpose();  // Input matrix transposed

    let epochs = 500000;
    let alpha = 0.1;

    let mut costs = Vec::new();

    for epoch in 0..epochs {
        let (a1, a2, a3) = feedfrwd(A0.clone(), weights.clone(), bias.clone());

        let error = cost(a3.clone(), y.clone());
        costs.push(error.clone());

        let (dc_dw3, dc_db3, dc_da2) = back1(a3.clone(), y.clone(), m, a2.clone(), weights.clone(), neurons.clone());
        let (dc_dw2, dc_db2, dc_da1) = back2(dc_da2, a1.clone(), a2.clone(), weights.clone(), neurons.clone());
        let (dc_dw1, dc_db1) = back3(dc_da1, a1.clone(), A0.clone(), weights.clone(), neurons.clone());

        weights[2] -= dc_dw3 * alpha;
        weights[1] -= dc_dw2 * alpha;
        weights[0] -= dc_dw1 * alpha;

        bias[2] -= dc_db3 * alpha;
        bias[1] -= dc_db2 * alpha;
        bias[0] -= dc_db1 * alpha;


        if epoch % 20 == 0 {
            println!("epoch: {}: cost = {:?}", epoch, costs[epoch]);
        }
    }
    println!("Training complete!");
    // Evaluate the trained network on the XOR inputs
    let (a1, a2, a3) = feedfrwd(A0.clone(), weights.clone(), bias.clone());

    // Output the final prediction for each XOR test case
    for i in 0..4 {
        let output = a3[(0, i)];
        println!("XOR input: {}, output: {:.3}", &x.row(i), output);
    }
    
}

fn xavier_init(rows: usize, cols: usize) -> DMatrix<f32> {
    let mut rng = rand::thread_rng();
    let std_dev = (2.0 / (rows + cols) as f32).sqrt();
    DMatrix::from_fn(rows, cols, |_, _| rng.gen_range(-std_dev..std_dev))
}

fn cost(y_hat: DMatrix<f32>, y: DMatrix<f32>) -> f32 {
    assert_eq!(y_hat.nrows(), y.nrows());

    let epsilon = 1e-15; // Avoid log(0)
    let y_hat = y_hat.map(|x| x.max(epsilon).min(1.0 - epsilon)); // Clipping values
    let losses = y.zip_map(&y_hat, |yi, y_hat_i| {
        -(yi * y_hat_i.ln() + (1.0 - yi) * (1.0 - y_hat_i).ln())
    });

    // Manually compute the mean
    let sum_of_losses: f32 = losses.iter().sum();
    let mean_loss = sum_of_losses / y_hat.ncols() as f32;

    mean_loss
}

fn sigmoid(arr: &mut DMatrix<f32>) {
    arr.iter_mut().for_each(|x| *x = 1.0 / (1.0 + (-*x).exp()));
}

fn feedfrwd(input: DMatrix<f32>, weights: Vec<DMatrix<f32>>, bias: Vec<DMatrix<f32>>)
    -> (DMatrix<f32>, DMatrix<f32>, DMatrix<f32>) {

    let mut z1 = weights[0].clone() * input.clone() + repeat_bias(&bias[0], input.clone().ncols());
    sigmoid(&mut z1);

    let mut z2 = weights[1].clone() * z1.clone() + repeat_bias(&bias[1], z1.ncols());
    sigmoid(&mut z2);

    let mut z3 = weights[2].clone() * z2.clone() + repeat_bias(&bias[2], z2.ncols());
    sigmoid(&mut z3);
    //println!("Dimensions: {}, {}, {}", z1.nrows(), z2.nrows(),z3.nrows());

    (z1, z2, z3)
}

// Custom function to repeat bias across columns
fn repeat_bias(bias: &DMatrix<f32>, num_cols: usize) -> DMatrix<f32> {
    DMatrix::from_fn(bias.nrows(), num_cols, |i, _| bias[(i, 0)])
}

fn back1(y_hat: DMatrix<f32>, y: DMatrix<f32>, m: usize, z2: DMatrix<f32>, weights: Vec<DMatrix<f32>>, neurons: Vec<usize>)
    -> (DMatrix<f32>, DMatrix<f32>, DMatrix<f32>) {

    assert_eq!(z2.nrows(), neurons[2]);
    assert_eq!(z2.ncols(), y_hat.ncols());

    let dc_dz3 = y_hat - y;
    let dc_dz3 = dc_dz3.map(|x| x / m as f32);
    
    assert_eq!(dc_dz3.nrows(), neurons[3]);
    assert_eq!(dc_dz3.ncols(), 4);

    let dc_dw3 = dc_dz3.clone() * z2.transpose();
    assert_eq!(dc_dw3.nrows(), neurons[3]);
    assert_eq!(dc_dw3.ncols(), neurons[2]);

    let dc_db3 = DMatrix::from_row_slice(dc_dw3.nrows(), 1, &(0..dc_dz3.clone()
        .nrows())
        .map(|i| dc_dz3.row(i)
            .sum())
        .collect::<Vec<f32>>());
    
    assert_eq!(dc_db3.nrows(), neurons[3]);
    assert_eq!(dc_db3.ncols(), 1);

    let dc_da2 = weights[2].transpose() * dc_dz3.clone();
    assert_eq!(dc_da2.nrows(), neurons[2]);
    assert_eq!(dc_da2.ncols(), 4);

    (dc_dw3, dc_db3, dc_da2)
}

fn back2(dc_da2: DMatrix<f32>, z1: DMatrix<f32>, z2: DMatrix<f32>, weights: Vec<DMatrix<f32>>, neurons: Vec<usize>)
    -> (DMatrix<f32>, DMatrix<f32>, DMatrix<f32>) {

    assert_eq!(z1.nrows(), neurons[1]);
    assert_eq!(z1.ncols(), 4);

    //println!("Dimensions: {}, {}", z2.clone().map(|x| 1.0 - x).ncols(),z2.clone().ncols());

    let da2_dz2 = z2.clone().component_mul(&(z2.clone().map(|x| x*(1.0 - x))));
    let dc_dz2 = dc_da2.component_mul(&da2_dz2);
    assert_eq!(dc_dz2.nrows(), neurons[2]);
    assert_eq!(dc_dz2.ncols(), 4);

    let dc_dw2 = dc_dz2.clone() * z1.transpose();
    assert_eq!(dc_dw2.nrows(), neurons[2]);
    assert_eq!(dc_dw2.ncols(), neurons[1]);

    let dc_db2 = DMatrix::from_row_slice(dc_dw2.nrows(), 1, &(0..dc_dw2.clone()
        .nrows())
        .map(|i| dc_dw2.row(i)
            .sum())
        .collect::<Vec<f32>>());
    
    assert_eq!(dc_db2.nrows(), neurons[2]);
    assert_eq!(dc_db2.ncols(), 1);

    let dc_da1 = weights[1].transpose() * dc_dz2;
    assert_eq!(dc_da1.nrows(), neurons[2]);
    assert_eq!(dc_da1.ncols(), 4);

    (dc_dw2, dc_db2, dc_da1)
}

fn back3(dc_da1: DMatrix<f32>, z1: DMatrix<f32>, z0: DMatrix<f32>, weights: Vec<DMatrix<f32>>, neurons: Vec<usize>)
    -> (DMatrix<f32>, DMatrix<f32>) {

    assert_eq!(z0.nrows(), neurons[0]);
    assert_eq!(z0.ncols(), 4);
    
    let da1_dz1 = z1.clone().component_mul(&(z1.map(|x| x * (1.0 - x))));
    let dc_dz1 = dc_da1.component_mul(&da1_dz1);
    assert_eq!(dc_dz1.nrows(), neurons[1]);
    assert_eq!(dc_dz1.ncols(), 4);

    let dc_dw1 = dc_dz1 * z0.transpose();
    assert_eq!(dc_dw1.nrows(), neurons[1]);
    assert_eq!(dc_dw1.ncols(), neurons[0]);

    let dc_db1 = DMatrix::from_row_slice(dc_dw1.nrows(), 1, &(0..dc_dw1.clone()
        .nrows())
        .map(|i| dc_dw1.row(i)
            .sum())
        .collect::<Vec<f32>>());

    assert_eq!(dc_db1.nrows(), neurons[1]);
    assert_eq!(dc_db1.ncols(), 1);

    (dc_dw1, dc_db1)
}
