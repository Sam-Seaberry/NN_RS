use nalgebra::{DMatrix};
use rand::prelude::*;
use core::arch;
use std::{env, vec};
use std::fs::File;
use tokio::task;
use std::sync::Arc;
mod convolve;
mod nn;
mod preprocess;
use matrix_display::*;


#[tokio::main]
async fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let mut images_list:Vec<DMatrix<f32>> = Vec::new();
    let mut images_raw:Vec<DMatrix<f32>> = Vec::new();
    let mut lables_list:Vec<Vec<f32>> = Vec::new();
    //match preprocess::load_data("${workspaceFolder}/nn/res/t10k"){
    match preprocess::load_data("./res/t10k"){   
        Ok(images) => {
            for i in 0..images.len(){
                images_list.push(preprocess::create_ndmatrix_from_mnist_image(&images[i], vec![1,784]));
                lables_list.push(images[i].classification.clone());
                images_raw.push(preprocess::create_ndmatrix_from_mnist_image(&images[i], vec![28,28]));
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


    let mut nn = nn::NeuralNetwork::new(vec![784, 128, 10],
        nn::Activation::Softmax, 
        nn::Cost::CrossEntropy, 
        nn::Optimizer::Adam, 
        0.01);

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
    random_shuffle_with_corralation(&mut images_list, &mut lables_list);
    let truth = convert_to_matrices(lables_list.clone(), 10, 1);
    
    let epochs = 400;

    let mut one = 0;
    let mut eight = 0;
    let mut nine = 0;
    for i in &truth[0..200]{
        let (_truth_num, index) = find_max_and_index(&i);
        
        match index.0 as i32{
            0=> one+=1,
            8=> eight+=1,
            9=> nine+=1,
            _=> ()
        } 
        
    }
    println!("Percentages: One: {} \n Eight: {} \n Nine: {}", (one as f32/200 as f32), (eight as f32 /200 as f32), (nine as f32 /200 as f32));
    
    nn.train(images_list[0..9900].to_vec(), 
        truth[0..9900].to_vec(),
        images_list[(images_list.len() - 100)..images_list.len()].to_vec(), 
        truth[(images_list.len() - 100)..truth.len()].to_vec(), 
        epochs, 
        127);
    //nn.train_concurrent(images_list[0..800].to_vec(), truth[0..800].to_vec(), epochs, 10, 100);

    
    /*let output = nn.predict(images_list[2900..images_list.len()].to_vec());
    let mut truthcnt = 0;

    for i in 0..output.len() {
        let (max_value, max_index) = find_max_and_index(&output[i]);
        let (_max_value_truth, max_index_truth) = find_max_and_index(&truth[i]);
        println!("---------------------------------------------------");
        if max_index.0 as i32 == max_index_truth.0 as i32{
            truthcnt += 1;
            println!("Correct");
        }else{
            println!("Incorrect");
        }
        
        println!("Max index output: {:?}, max index truth {:?}", max_index.0, max_index_truth.0);
        println!("Output: {}, Value: {:.3}, Index: {:?}", i, max_value, max_index);
        println!("Truth: {:?}", truth[i]);
        println!("--------------------------------------------------- \n\n");
    }
    println!("Accuracy: {}", truthcnt as f32 / output.len() as f32);
    
    
    //println!("Truth: {:?}", truth[0..5].to_vec());
    
    //for i in 0..4 {
      //  let output = nn.predict(x.transpose().clone());
        //println!("XOR input: {}, output: {:.3}", &x.row(i), output);
    //}*/
    
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


fn random_shuffle_with_corralation(
    images: &mut Vec<DMatrix<f32>>, 
    labels: &mut Vec<Vec<f32>>
) {
    let mut combined: Vec<_> = images.drain(..).zip(labels.drain(..)).collect();
    combined.shuffle(&mut thread_rng());

    for (img, lbl) in combined {
        images.push(img);
        labels.push(lbl);
    }
}

fn display_mnist_matrix(matrix: &DMatrix<f32>) {
    // Unicode grayscale blocks, from light to dark
    let shades = [' ', '░', '▒', '▓', '█'];

    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            let val = matrix[(row, col)];
            let idx = (val * (shades.len() as f32 - 1.0)).round() as usize;
            let ch = shades[idx.min(shades.len() - 1)];
            print!("{}", ch);
        }
        println!();
    }
}


