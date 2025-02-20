use nalgebra::{DMatrix};
use rand::prelude::*;
use std::env;

mod convolve;
mod nn;

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    
    let inputs = vec![
        DMatrix::from_row_slice(4, 4, &[
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0
        ]),
        //DMatrix::from_row_slice(4, 4, &[
        //    1.0, 2.0, 3.0, 4.0,
        //    5.0, 6.0, 7.0, 8.0,
        //    9.0, 10.0, 11.0, 12.0,
        //   13.0, 14.0, 15.0, 16.0
        //])
    ];
    let filters = vec![
        vec![
            DMatrix::from_row_slice(2, 2, &[
                1.0, 0.0,
                0.0, 1.0
            ]),
            DMatrix::from_row_slice(2, 2, &[
                1.0, 0.0,
                0.0, 1.0
            ])
        ],
        vec![
            DMatrix::from_row_slice(2, 2, &[
                1.0, 0.0,
                0.0, 1.0
            ]),
            DMatrix::from_row_slice(2, 2, &[
                1.0, 0.0,
                0.0, 1.0
            ])
        ]
    ];
    let bias = vec![
        DMatrix::from_row_slice(1, 1, &[0.0]),
        DMatrix::from_row_slice(1, 1, &[0.0])
    ];
    assert_eq!(filters.len(), bias.len());

    let stride = 1;
    let outputs = convolve::convolvefrwd(inputs.clone(), filters.clone(), bias.clone(), stride);
    for i in 0..outputs.len() {
        println!("Output {}: {:?}", i, outputs[i]);
    }
    
    let backwards  = convolve::convolvebackwrd(inputs.clone(), filters.clone(), bias.clone(), stride, outputs);
    for i in 0..backwards.len() {
        println!("Backward {}: {:?}", i, backwards[i]);
    }
    
    
    
    
    let mut nn = nn::NeuralNetwork::new(vec![2, 6, 8, 1], 0.5);
    let x = DMatrix::from_row_slice(4, 2, &[
          0.0, 0.0, 
          1.0, 0.0, 
          1.0, 1.0, 
          0.0, 1.0
     ]);
    let y = DMatrix::from_row_slice(1, 4, &[
        0.0, 1.0, 0.0, 1.0
    ]);
    let epochs = 5000;
    nn.train(x.transpose().clone(), y.clone(), epochs);
    for i in 0..4 {
        let output = nn.predict(x.transpose().clone());
        println!("XOR input: {}, output: {:.3}", &x.row(i), output);
    }



    /*let layers = 5;
    let neurons = vec![2, 600, 8000, 400, 60, 1];

    let mut weights: Vec<DMatrix<f32>> = Vec::new();
    let mut bias: Vec<DMatrix<f32>> = Vec::new();
    for i in 0..layers{
        weights.push(xavier_init(neurons[i+1], neurons[i]));
        bias.push(xavier_init(neurons[i+1], 1));
    }

    let x = DMatrix::from_row_slice(4, 2, &[
        0.0, 0.0, 
        1.0, 0.0, 
        1.0, 1.0, 
        0.0, 1.0
    ]);

    let y = DMatrix::from_row_slice(1, 4, &[
        0.0, 1.0, 0.0, 1.0
    ]);
    
    let A0 = x.transpose();  // Input matrix transposed

    let mut epochs = 0;
    let alpha = 1.2;

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
    }*/

   
    
}

