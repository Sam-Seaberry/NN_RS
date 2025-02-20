use nalgebra::{DMatrix};
use std::env;

pub fn convolvefrwd(inputs: Vec<DMatrix<f32>>, filters: Vec<Vec<DMatrix<f32>>>, bias: Vec<DMatrix<f32>>, stride: usize) -> Vec<DMatrix<f32>> {
   let mut outputs = Vec::new();
    for i in 0..inputs.len() {
         let input = &inputs[i];
         let mut output = DMatrix::zeros(input.nrows() - filters[0][0].nrows() + 1, input.ncols() - filters[0][0].ncols() + 1);
         for j in 0..filters.len() {
              let filter = &filters[j];
              let b = &bias[j];
              for k in 0..output.nrows() {
                for l in 0..output.ncols() {
                     let mut sum = 0.0;
                     for m in 0..filter.len() {
                          for n in 0..filter[0].nrows() {
                            for o in 0..filter[0].ncols() {
                                 sum += input[(k + m * stride, l + n * stride)] * filter[m][(n, o)];
                            }
                          }
                     }
                     output[(k, l)] += sum + b[(0, 0)];
                }
              }
            outputs.push(output.clone());
         }
         
    }
    outputs
}

pub fn convolvebackwrd(inputs: Vec<DMatrix<f32>>, filters: Vec<Vec<DMatrix<f32>>>, bias: Vec<DMatrix<f32>>, stride: usize, outputs: Vec<DMatrix<f32>>) -> Vec<DMatrix<f32>> {
    let mut dinputs = Vec::new();
    for i in 0..inputs.len() {
         let input = &inputs[i];
         let mut dinput = DMatrix::zeros(input.nrows(), input.ncols());
         for j in 0..filters.len() {
              let filter = &filters[j];
              let b = &bias[j];
              let output = &outputs[j];
              for k in 0..output.nrows() {
                for l in 0..output.ncols() {
                     let mut sum = 0.0;
                     for m in 0..filter.len() {
                          for n in 0..filter[0].nrows() {
                            for o in 0..filter[0].ncols() {
                                 sum += output[(k, l)] * filter[m][(n, o)];
                                 dinput[(k + m * stride, l + n * stride)] += sum;
                            }
                          }
                     }
                }
              }
            dinputs.push(dinput.clone());
        }
         
    }
    dinputs
}