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
    
    let m = 4;
    let A0 = x.transpose();  // Input matrix transposed

    let mut epochs = 0;
    let alpha = 0.01;

    let mut costs = Vec::new();

    let mut activations: Vec<DMatrix<f32>> = Vec::new();

    loop {
        
        activations = feedfrwd(A0.clone(), y.clone(), weights.clone(), bias.clone(), layers.clone());

        let error = cost(activations[layers-1].clone(), y.clone());
        costs.push(error.clone());

        let (dc_dw, dc_db, dc_da) = backpropagation(activations.clone(), &mut bias, &mut weights, activations[layers-1].clone(), A0.clone(), y.clone(), neurons.clone(), layers.clone(), alpha.clone());

        if epochs % 20 == 0 {
            println!("epoch: {}: cost = {:?}", epochs, costs[epochs]);
        }

        if costs[epochs] < 0.1{
            break;
        }
        epochs += 1;
    };
    
    println!("Training complete!");
    
    for i in 0..4 {
        let output = activations[layers-1][(0, i)];
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

fn relu(arr: &mut DMatrix<f32>) {
    arr.iter_mut().for_each(|x| *x = x.max(0.0));
}

fn feedfrwd(input: DMatrix<f32>, truth:DMatrix<f32>, weights: Vec<DMatrix<f32>>, bias: Vec<DMatrix<f32>>, layers: usize)
    -> (Vec<DMatrix<f32>>) {

    let mut input = input;
    let mut activations = Vec::new();
    assert_eq!(weights.len(), layers);
    assert_eq!(bias.len(), layers);

    for i in 0..layers {
        assert_eq!(weights[i].ncols(), input.nrows());
        assert_eq!(bias[i].nrows(), weights[i].nrows());

        let mut z = weights[i].clone() * input.clone() + repeat_bias(&bias[i], input.ncols());
        sigmoid(&mut z);

        activations.push(z.clone());

        input = z;

    }
    //activations.push(truth.clone());
    //Note: activations[0] is the input layer
    activations
}

// Custom function to repeat bias across columns
fn repeat_bias(bias: &DMatrix<f32>, num_cols: usize) -> DMatrix<f32> {
    DMatrix::from_fn(bias.nrows(), num_cols, |i, _| bias[(i, 0)])
}

fn backpropagation(activations: Vec<DMatrix<f32>>, 
    bias: &mut Vec<DMatrix<f32>>, 
    weights: &mut Vec<DMatrix<f32>>, 
    passthough: DMatrix<f32>,
    input: DMatrix<f32>,
    truth: DMatrix<f32>,
    neurons: Vec<usize>, 
    layer: usize,
    alpha: f32)
    -> (DMatrix<f32>, DMatrix<f32>, DMatrix<f32>) {

    if layer == weights.len() {
        // Output layer
        let current_layer = layer - 1;
        let mut d_activation = activations[current_layer].clone() - truth.clone();
        assert_eq!(d_activation.nrows(), neurons[current_layer + 1]);
        assert_eq!(d_activation.ncols(), truth.ncols());

        d_activation = d_activation.map(|x| x * 2.0);
        d_activation = d_activation.map(|x| x / truth.ncols() as f32);
        assert_eq!(d_activation.nrows(), neurons[current_layer + 1]);
        assert_eq!(d_activation.ncols(), truth.ncols());

        // Weights derivative
        let d_weights = d_activation.clone() * activations[current_layer - 1].transpose();
        assert_eq!(d_weights.nrows(), neurons[current_layer + 1]);
        assert_eq!(d_weights.ncols(), neurons[current_layer]);

        // Bias derivative
        let d_bias = DMatrix::from_row_slice(d_weights.nrows(), 1, &(0..d_weights.nrows())
            .map(|i| d_activation.row(i).sum())
            .collect::<Vec<f32>>());
        assert_eq!(d_bias.nrows(), neurons[current_layer + 1]);
        assert_eq!(d_bias.ncols(), 1);

        // Layer derivative
        let d_layer = weights[current_layer].transpose() * d_activation;
        assert_eq!(d_layer.nrows(), neurons[current_layer]);
        assert_eq!(d_layer.ncols(), truth.ncols());

        weights[current_layer] -= alpha * d_weights.clone();
        bias[current_layer] -= alpha * d_bias.clone();

        let (d_weights_prev, d_bias_prev, d_layer_prev) = backpropagation(activations, bias, weights, d_layer.clone(), input.clone(), truth.clone(), neurons, layer - 1, alpha);


        return (d_weights_prev, d_bias_prev, d_layer_prev);

    } else if layer-1 == 0{

        let current_layer = layer - 1;
        assert_eq!(input.nrows(), neurons[0]);

        let d_activations = activations[current_layer].clone().component_mul(&(activations[current_layer].clone().map(|x| x * (1.0 - x))));
        assert_eq!(d_activations.nrows(), neurons[current_layer + 1]);
        // dc/dz = dc/da * da/dz
        let d_activation = d_activations.component_mul(&passthough);
        assert_eq!(d_activation.nrows(), neurons[current_layer + 1]);
        assert_eq!(d_activation.ncols(), passthough.ncols());

        let d_weights = d_activation.clone() * input.transpose();
        assert_eq!(d_weights.nrows(), neurons[current_layer + 1]);
        assert_eq!(d_weights.ncols(), neurons[current_layer]);

        let d_bias = DMatrix::from_row_slice(d_weights.nrows(), 1, &(0..d_weights.nrows())
            .map(|i| d_activation.row(i).sum())
            .collect::<Vec<f32>>());

        assert_eq!(d_bias.nrows(), neurons[current_layer + 1]);
        assert_eq!(d_bias.ncols(), 1);

        let d_layer = weights[current_layer].transpose() * d_activation;
        assert_eq!(d_layer.nrows(), neurons[current_layer]);

        weights[current_layer] -= alpha * d_weights.clone();
        bias[current_layer] -= alpha * d_bias.clone();

        return (d_weights, d_bias, d_layer);
        
    } else {
        // Hidden layers
        let current_layer = layer -1;
        let d_sigmoid = activations[current_layer].clone().component_mul(&(activations[current_layer].clone().map(|x| x * (1.0 - x))));
        assert_eq!(d_sigmoid.nrows(), neurons[current_layer + 1]);
        // dc/dz = dc/da * da/dz
        let d_activation = d_sigmoid.component_mul(&passthough);
        assert_eq!(d_activation.nrows(), neurons[current_layer + 1]);
        assert_eq!(d_activation.ncols(), passthough.ncols());

        // Weights derivative
        let d_weights = d_activation.clone() * activations[current_layer - 1].transpose();
        assert_eq!(d_weights.nrows(), neurons[current_layer + 1]);
        assert_eq!(d_weights.ncols(), neurons[current_layer]);

        // Bias derivative
        let d_bias = DMatrix::from_row_slice(d_weights.nrows(), 1, &(0..d_weights.nrows())
            .map(|i| d_activation.row(i).sum())
            .collect::<Vec<f32>>());
        assert_eq!(d_bias.nrows(), neurons[current_layer + 1]);
        assert_eq!(d_bias.ncols(), 1);

        // Layer derivative
        let d_layer = weights[current_layer].transpose() * d_activation;
        assert_eq!(d_layer.nrows(), neurons[current_layer]);
        assert_eq!(d_layer.ncols(), passthough.ncols());

        let (d_weights_prev, d_bias_prev, d_layer_prev) = backpropagation(activations, bias, weights, d_layer.clone(), input.clone(), truth.clone(), neurons, layer - 1, alpha);

        weights[current_layer] -= alpha * d_weights.clone();
        bias[current_layer] -= alpha * d_bias.clone();
       

        return (d_weights, d_bias, d_layer.clone());
    }
}

fn back1(y_hat: DMatrix<f32>, y: DMatrix<f32>, m: usize, z2: DMatrix<f32>, weights: Vec<DMatrix<f32>>, neurons: Vec<usize>)
    -> (DMatrix<f32>, DMatrix<f32>, DMatrix<f32>) {

    assert_eq!(z2.nrows(), neurons[2]);
    assert_eq!(z2.ncols(), y_hat.ncols());

    //calculate the derivative of the cost function
    let mut dc_dz3 = y_hat - y;
    dc_dz3 = dc_dz3.map(|x| x * 2.0);
    //calculate the derivative of the activation function
    let dc_dz3 = dc_dz3.map(|x| x / m as f32);
    
    assert_eq!(dc_dz3.nrows(), neurons[3]);
    assert_eq!(dc_dz3.ncols(), 4);

    //calculate the derivative of the weights
    let dc_dw3 = dc_dz3.clone() * z2.transpose();
    assert_eq!(dc_dw3.nrows(), neurons[3]);
    assert_eq!(dc_dw3.ncols(), neurons[2]);

    //calculate the derivative of the bias
    let dc_db3 = DMatrix::from_row_slice(dc_dw3.nrows(), 1, &(0..dc_dz3.clone()
        .nrows())
        .map(|i| dc_dz3.row(i)
            .sum())
        .collect::<Vec<f32>>());
    
    assert_eq!(dc_db3.nrows(), neurons[3]);
    assert_eq!(dc_db3.ncols(), 1);

    //calculate the derivative of the activation function
    let dc_da2 = weights[2].transpose() * dc_dz3.clone();
    assert_eq!(dc_da2.nrows(), neurons[2]);
    assert_eq!(dc_da2.ncols(), 4);

    (dc_dw3, dc_db3, dc_da2)
}

fn back2(dc_da2: DMatrix<f32>, z1: DMatrix<f32>, z2: DMatrix<f32>, weights: Vec<DMatrix<f32>>, neurons: Vec<usize>)
    -> (DMatrix<f32>, DMatrix<f32>, DMatrix<f32>) {

    assert_eq!(z1.nrows(), neurons[1]);
    assert_eq!(z1.ncols(), 4);

    
    //calculate the derivative of the activation function
    let da2_dz2 = z2.clone().component_mul(&(z2.clone().map(|x| x*(1.0 - x))));
    //calculate the derivative of the cost function
    let dc_dz2 = dc_da2.component_mul(&da2_dz2);
    assert_eq!(dc_dz2.nrows(), neurons[2]);
    assert_eq!(dc_dz2.ncols(), 4);

    //calculate the derivative of the weights
    let dc_dw2 = dc_dz2.clone() * z1.transpose();
    assert_eq!(dc_dw2.nrows(), neurons[2]);
    assert_eq!(dc_dw2.ncols(), neurons[1]);

    //calculate the derivative of the bias
    let dc_db2 = DMatrix::from_row_slice(dc_dw2.nrows(), 1, &(0..dc_dw2.clone()
        .nrows())
        .map(|i| dc_dw2.row(i)
            .sum())
        .collect::<Vec<f32>>());
    
    assert_eq!(dc_db2.nrows(), neurons[2]);
    assert_eq!(dc_db2.ncols(), 1);

    //calculate the derivative of the activation function
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
