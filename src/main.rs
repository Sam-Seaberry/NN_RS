use nalgebra::{DMatrix};
use rand::prelude::*;
use core::arch;
use std::env;
use std::fs::File;
mod convolve;
mod nn;
mod preprocess;

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let mut images_list:Vec<DMatrix<f32>> = Vec::new();
    let mut lables_list:Vec<u8> = Vec::new();
    match preprocess::load_data("./res/t10k"){
        Ok(images) => {
            for i in 0..images.len(){
                images_list.push(preprocess::create_ndmatrix_from_mnist_image(&images[i], vec![28, 28]));
                lables_list.push(images[i].classification);
            }
        },
        Err(e) => {
            println!("Error: {:?}", e);
        }
    }
    
    let architecture = vec![2, 30, 10, 30, 1];
    let mut filters: Vec<DMatrix<f32>> = Vec::new();
    for i in 0..architecture.len() -1{
        for j in 0..architecture[i]{
            
        }
    }

    let inputs = vec![
        DMatrix::from_row_slice(4, 4, &[
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0
        ]),
        DMatrix::from_row_slice(4, 4, &[
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
           13.0, 14.0, 15.0, 16.0
        ])
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

    let mut layers: Vec<convolve::LayerConfig> = Vec::new();
    
    let conv_layer1 = convolve::Conv2D::new(filters[0][0].nrows(), 
    vec![3,3], 
    convolve::Activation::Relu, 
    vec![28,28],
    1,
    convolve::Padding::Valid,
    filters,
    bias );

    layers.push(convolve::LayerConfig::new(convolve::LayerType::Conv2D, 
        None,
        None, 
        None, 
        Some(conv_layer1),
    None));
    

    /* 
    
    assert_eq!(filters.len(), bias.len());

    let stride = 1;
    let outputs = convolve::convolvefrwd(inputs.clone(), filters.clone(), bias.clone(), stride);
    for i in 0..outputs.len() {
        println!("Output {}: {}", i, outputs[i]);
    }
    
    let backwards  = convolve::convolvebackwrd(inputs.clone(), filters.clone(), bias.clone(), stride, outputs);
    for i in 0..backwards.len() {
        println!("Backward {}: {}", i, backwards[i]);
    }*/

    let mut nn = nn::NeuralNetwork::new(vec![2, 6, 4, 6, 1], 1.5);
    let x = DMatrix::from_row_slice(4, 2, &[
          0.0, 0.0, 
          1.0, 0.0, 
          1.0, 1.0, 
          0.0, 1.0
     ]);
    let y = DMatrix::from_row_slice(1, 4, &[
        0.0, 1.0, 0.0, 1.0
    ]);
    let epochs = 10000;
    nn.train(x.transpose().clone(), y.clone(), epochs);
    for i in 0..4 {
        let output = nn.predict(x.transpose().clone());
        println!("XOR input: {}, output: {:.3}", &x.row(i), output);
    }
    
}

