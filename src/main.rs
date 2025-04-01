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
    let mut lables_list:Vec<Vec<f32>> = Vec::new();
    match preprocess::load_data("./res/t10k"){
        Ok(images) => {
            for i in 0..images.len(){
                images_list.push(preprocess::create_ndmatrix_from_mnist_image(&images[i], vec![1,784]));
                lables_list.push(images[i].classification.clone());
            }
        },
        Err(e) => {
            println!("Error: {:?}", e);
        }
    }
    for i in &lables_list{
        println!("{:?}", i);
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


    let mut nn = nn::NeuralNetwork::new(vec![784, 128, 64, 10],
        nn::Activation::Sigmoid, 
        nn::Cost::CrossEntropy, 
        nn::Optimizer::Adam, 
        0.0000025);

    let x = DMatrix::from_row_slice(4, 2, &[
          0.0, 0.0, 
          1.0, 0.0, 
          1.0, 1.0, 
          0.0, 1.0
     ]);
    let y = DMatrix::from_row_slice(1, 4, &[
        0.0, 1.0, 0.0, 1.0
    ]);
    let truth  = DMatrix::from_row_slice(1, 4, &[
        0.0, 1.0, 1.0, 0.0
    ]);
    
    let truth = convert_to_matrices(lables_list.clone(), 10, 1);
    
    let epochs = 2000;
    nn.train(images_list[0..100].to_vec(), truth[0..100].to_vec(), epochs);

    
    let output = nn.predict(images_list[100..200].to_vec());
    let mut truthcnt = 0;

    for i in 0..output.len() {
        let (max_value, max_index) = find_max_and_index(&output[i]);
        let (_max_value_truth, max_index_truth) = find_max_and_index(&truth[i]);
        if max_index == max_index_truth{
            truthcnt += 1;
            println!("Correct");
        }else{
            println!("Incorrect");
        }
        

        println!("Output: {}, Value: {:.3}, Index: {:?}", i, max_value, max_index);
        println!("Truth: {:?}", truth[i]);
    }
    println!("Accuracy: {}", truthcnt as f32 / output.len() as f32);
    
    
    //println!("Truth: {:?}", truth[0..5].to_vec());
    
    //for i in 0..4 {
      //  let output = nn.predict(x.transpose().clone());
        //println!("XOR input: {}, output: {:.3}", &x.row(i), output);
    //}
    
}

fn convert_to_matrices(data: Vec<Vec<f32>>, rows: usize, cols: usize) -> Vec<DMatrix<f32>> {
    data.into_iter()
        .map(|flat_vec| {
            assert_eq!(flat_vec.len(), rows * cols, "The length of the flat vector does not match the specified dimensions.");
            DMatrix::from_vec(rows, cols, flat_vec)})
        .collect()
}


fn find_max_and_index(matrix: &DMatrix<f32>) -> (f32, (usize, usize)) {
    let mut max_value = f32::MIN;
    let mut max_index = (0, 0);

    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            let value = matrix[(i, j)];
            if value > max_value {
                max_value = value;
                max_index = (i, j);
            }
        }
    }
    
    (max_value, max_index)
}


