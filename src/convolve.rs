use nalgebra::{DMatrix};
use std::env;
use rand::prelude::*;

pub enum Activation {
    Sigmoid,
    Relu,
    Tanh,
    Softmax,
}

pub enum Loss {
    BinaryCrossEntropy,
    MeanSquaredError,
    
}
pub enum Optimizer {
    SGD,
    Adam,
    RMSprop,
    
}
pub struct Dense {
    units: usize,
    activation: Activation,
    input_shape: Vec<usize>,
    weights: DMatrix<f32>,
    bias: DMatrix<f32>,
}

pub struct Conv2D {
    filters_shape: usize,
    kernel_size: Vec<usize>,
    activation: Activation,
    input_shape: Vec<usize>,
    stride: usize,
    padding: Padding,
    filters: Vec<Vec<DMatrix<f32>>>,
    bias: Vec<DMatrix<f32>>,
}

pub struct Flatten {
    input_shape: Vec<usize>,
    
}

pub struct MaxPooling2D {
    pool_size: Vec<usize>,
    stride: usize,
}

pub struct Dropout {
    rate: f32,
    mask: Option<DMatrix<f32>>,
}


pub enum LayerType {
    Dense,
    Conv2D,
    Flatten,
    MaxPooling2D,
    Dropout,
}

pub struct LayerConfig{
    layer_type: LayerType,
    dropout: Option<Dropout>,
    max_pooling: Option<MaxPooling2D>,
    flatten: Option<Flatten>,
    conv2d: Option<Conv2D>,
    dense: Option<Dense>,
}

pub enum Padding {
    Same,
    Valid,
}

pub enum Pooling {
    Max,
    Average,
}

struct ConvolutionNN<'a> {
    layers: &'a mut Vec<LayerConfig>,
    optimizer: Optimizer,
    loss: Loss,
    input: Vec<DMatrix<f32>>,
}


impl<'a> ConvolutionNN<'a> {
    pub fn new(layers:&mut Vec<LayerConfig>, optimizer: Optimizer, loss: Loss, input: Vec<DMatrix<f32>>) -> ConvolutionNN {
        ConvolutionNN {
            layers,
            optimizer,
            loss,
            input,
        }
    }
    fn new_layer(&mut self, layer: Option<LayerConfig>)->Result<(), String>{
        
        match &layer.as_ref().unwrap().layer_type{
            LayerType::Dense=>{
                Ok(self.layers.push(LayerConfig{layer_type: LayerType::Dense, dense: layer.unwrap().dense, dropout: None, max_pooling: None, flatten: None, conv2d: None}))
            },
            LayerType::Conv2D=>{
                Ok(self.layers.push(LayerConfig{layer_type: LayerType::Conv2D, dense: None, dropout: None, max_pooling: None, flatten: None, conv2d: layer.unwrap().conv2d}))
            },
            LayerType::Dropout=>{
                Ok(self.layers.push(LayerConfig{layer_type: LayerType::Dropout, dense: None, dropout: layer.unwrap().dropout, max_pooling: None, flatten: None, conv2d: None}))
            },
            LayerType::Flatten=>{
                Ok(self.layers.push(LayerConfig{layer_type: LayerType::Flatten, dense: None, dropout: None, max_pooling: None, flatten: layer.unwrap().flatten, conv2d: None}))
            },
            LayerType::MaxPooling2D=>{
                Ok(self.layers.push(LayerConfig{layer_type: LayerType::MaxPooling2D, dense: None, dropout: None, max_pooling: layer.unwrap().max_pooling, flatten: None, conv2d: None}))
            },
            _=>Err("some message".to_string()),
        }
        
    }
    
    pub fn train(&self, x: DMatrix<f32>, y: DMatrix<f32>, epochs: usize) {
        
       
    }

    fn forward(&mut self, mut x:Vec<DMatrix<f32>>, y: DMatrix<f32>){
        for mut layer in self.layers.iter_mut(){
            match layer.layer_type{
                LayerType::Dense=>{
                    let dense = layer.dense.as_ref().unwrap();
                    x.push(dense.densefrwd(x[0].clone()));
                },
                LayerType::Conv2D=>{
                    let conv2d = layer.conv2d.as_ref().unwrap();
                    x = conv2d.convolvefrwd(x.clone());
                },
                LayerType::Flatten=>{
                    let flatten = layer.flatten.as_ref().unwrap();
                    x = vec![flatten.reshape(x[0].clone())];
                },
                LayerType::MaxPooling2D=>{
                    let max_pooling = layer.max_pooling.as_ref().unwrap();
                    x = max_pooling.max_pooling(x.clone());
                },
                LayerType::Dropout=>{
                    let dropout = layer.dropout.as_mut().unwrap();
                    x[0] = dropout.dropout(x[0].clone());

                },
                _=>{},
            }
        }
    }

    fn backwards(&self, mut x:Vec<DMatrix<f32>>, y: DMatrix<f32>){
        for layer in self.layers.iter().rev(){
            match layer.layer_type{
                LayerType::Dense=>{
                    let dense = layer.dense.as_ref().unwrap();
                    x[0] = dense.densebackwrd(x[0].clone(),  x[0].clone(), y.clone())
                },
                LayerType::Conv2D=>{
                    let conv2d = layer.conv2d.as_ref().unwrap();
                    x = conv2d.convolvebackwrd(x.clone(), x.clone());
                },
                LayerType::Flatten=>{
                    let flatten = layer.flatten.as_ref().unwrap();
                    x = vec![flatten.reshape_prime(x[0].clone(), y.clone())];
                },
                LayerType::MaxPooling2D=>{
                    let max_pooling = layer.max_pooling.as_ref().unwrap();
                    x = max_pooling.max_pooling_prime(x.clone(), x.clone());
                },
                LayerType::Dropout=>{
                    let dropout = layer.dropout.as_ref().unwrap();
                    x[0] = dropout.dropout_prime(x[0].clone());
                },
                _=>{},
            }
        }
    }

    fn xavier_init(rows: usize, cols: usize) -> DMatrix<f32> {
        let mut rng = rand::thread_rng();
        let std_dev = (2.0 / (rows + cols) as f32).sqrt();
        DMatrix::from_fn(rows, cols, |_, _| rng.gen_range(-std_dev..std_dev))
    }

    fn binary_cross_entrapy(y_pred: Vec<DMatrix<f32>>, y_true: Vec<DMatrix<f32>>) -> f32 {
        let mut loss = 0.0;
        for i in 0..y_pred.len() {
            let pred = &y_pred[i];
            let bin = &y_true[i];
            for j in 0..pred.nrows() {
                for k in 0..pred.ncols() {
                    loss += -1.0 * (bin[(j, k)] * pred[(j, k)].ln() + (1.0 - bin[(j, k)]) * (1.0 - pred[(j, k)]).ln());
                }
            }
        }
        loss
    }

    fn binary_cross_entrapy_prime(y_pred: Vec<DMatrix<f32>>, y_true: Vec<DMatrix<f32>>) -> Vec<DMatrix<f32>> {
        let mut dloss = Vec::new();
        for i in 0..y_pred.len() {
            let pred = &y_pred[i];
            let bin = &y_true[i];
            let mut dloss_input = DMatrix::zeros(pred.nrows(), pred.ncols());
            for j in 0..pred.nrows() {
                for k in 0..pred.ncols() {
                    dloss_input[(j, k)] = -1.0 * (bin[(j, k)] / pred[(j, k)] - (1.0 - bin[(j, k)]) / (1.0 - pred[(j, k)]));
                }
            }
            dloss.push(dloss_input.clone());
        }
        dloss
    }

    

}




impl Dropout{
    pub fn new(rate: f32)->Dropout{
        Dropout{
            rate,
            mask: None,
        }
    }
    fn dropout(&mut self, x: DMatrix<f32>) -> DMatrix<f32> {
        let mut mask = DMatrix::zeros(x.nrows(), x.ncols());
        let mut rng = rand::thread_rng();
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                mask[(i, j)] = if rng.gen::<f32>() < self.rate { 0.0 } else { 1.0 };
            }
        }
        self.mask = Some(mask.clone());
        x.component_mul(&mask)
    }
    fn dropout_prime(&self, x: DMatrix<f32>) -> DMatrix<f32> {
        x.component_mul(&self.mask.as_ref().unwrap())
    }
}


impl Dense{
    pub fn new(units: usize, activation: Activation, input_shape: Vec<usize>, input: DMatrix<f32>, weights: DMatrix<f32>, bias: DMatrix<f32>)->Dense{
        Dense{
            units,
            activation,
            input_shape,
            weights,
            bias,
        }
    }
    fn densefrwd(&self, x: DMatrix<f32>) -> DMatrix<f32> {
        let mut output = DMatrix::zeros(self.weights.nrows(), x.ncols());
        for i in 0..self.weights.nrows() {
            for j in 0..x.ncols() {
                let mut sum = 0.0;
                for k in 0..self.weights.ncols() {
                    sum += self.weights[(i, k)] * x[(k, j)];
                }
                output[(i, j)] = sum + self.bias[(i, 0)];
            }
        }
        output
    }
    
    fn densebackwrd(&self, x: DMatrix<f32>, outputs: DMatrix<f32>, y: DMatrix<f32>) -> DMatrix<f32> {
        let mut dinputs = DMatrix::zeros(x.nrows(), x.ncols());
        for i in 0..self.weights.nrows() {
            for j in 0..x.ncols() {
                let mut sum = 0.0;
                for k in 0..self.weights.ncols() {
                    sum += self.weights[(i, k)] * x[(k, j)];
                }
                let doutput = outputs[(i, j)] - y[(i, j)];
                for k in 0..self.weights.ncols() {
                    dinputs[(k, j)] = sum + doutput;
                }
            }
        }
        dinputs
    }
}

impl MaxPooling2D{
    pub fn new(pool_size: Vec<usize>, stride: usize)->MaxPooling2D{
        MaxPooling2D{
            pool_size,
            stride,
        }
    }
    fn max_pooling(&self, inputs: Vec<DMatrix<f32>>,) -> Vec<DMatrix<f32>> {
        let mut pooled = Vec::new();
        for i in 0..inputs.len() {
            let input = &inputs[i];
            let mut output = DMatrix::zeros((input.nrows() - self.pool_size[0]) / self.stride + 1, (input.ncols() - self.pool_size[1]) / self.stride + 1);
            for j in 0..output.nrows() {
                for k in 0..output.ncols() {
                    let mut max = f32::MIN;
                    for l in 0..self.pool_size[0] {
                        for m in 0..self.pool_size[1] {
                            max = max.max(input[(j + l * self.stride, k + m * self.stride)]);
                        }
                    }
                    output[(j, k)] = max;
                }
            }
            pooled.push(output.clone());
        }
        pooled
    }
    fn max_pooling_prime(&self, inputs: Vec<DMatrix<f32>>, outputs: Vec<DMatrix<f32>>) -> Vec<DMatrix<f32>> {
        let mut dinputs = Vec::new();
        for i in 0..inputs.len() {
            let input = &inputs[i];
            let output = &outputs[i];
            let mut dinput = DMatrix::zeros(input.nrows(), input.ncols());
            for j in 0..output.nrows() {
                for k in 0..output.ncols() {
                    let max = output[(j, k)];
                    for l in 0..self.pool_size[0] {
                        for m in 0..self.pool_size[1] {
                            if input[(j + l * self.stride, k + m * self.stride)] == max {
                                dinput[(j + l * self.stride, k + m * self.stride)] = 1.0;
                            }
                        }
                    }
                }
            }
            dinputs.push(dinput.clone());
        }
        dinputs
    }

}

impl Conv2D{
    pub fn new(filters_shape: usize, kernel_size: Vec<usize>, activation: Activation, input_shape: Vec<usize>, stride: usize, padding: Padding, filters: Vec<Vec<DMatrix<f32>>>, bias: Vec<DMatrix<f32>>)->Conv2D{
        Conv2D{
            filters_shape,
            kernel_size,
            activation,
            input_shape,
            stride,
            padding,
            filters,
            bias,
        }
    }
    fn convolvefrwd(&self, inputs: Vec<DMatrix<f32>>) -> Vec<DMatrix<f32>> {
        let mut outputs = Vec::new();
         for i in 0..inputs.len() {
              let input = &inputs[i];
              let mut output = DMatrix::zeros(input.nrows() - self.filters[0][0].nrows() + 1, input.ncols() - self.filters[0][0].ncols() + 1);
              for j in 0..self.filters.len()-1 {
                   let filter = &self.filters[j];
                   let b: &nalgebra::Matrix<f32, nalgebra::Dyn, nalgebra::Dyn, nalgebra::VecStorage<f32, nalgebra::Dyn, nalgebra::Dyn>> = &self.bias[j];
                   for k in 0..output.nrows() {
                     for l in 0..output.ncols() {
                          let mut sum = 0.0;
                          for m in 0..filter.len() {
                               for n in 0..filter[0].nrows() {
                                 for o in 0..filter[0].ncols() {
                                      sum += input[(k + m * self.stride, l + n * self.stride)] * filter[m][(n, o)];
                                 }
                               }
                          }
                         output[(k, l)] += sum + b[(0, 0)];
                     }
                   }
                 
              }
             outputs.push(output.clone());
         }
         outputs
     }
    
     fn convolvebackwrd(&self, inputs: Vec<DMatrix<f32>>, outputs: Vec<DMatrix<f32>>) -> Vec<DMatrix<f32>> {
    
        let mut dinputs = Vec::new();
        for i in 0..inputs.len() {
             let input = &inputs[i];
             let mut dinput = DMatrix::zeros(input.nrows(), input.ncols());
             for j in 0..self.filters.len()-1 {
                  let filter = &self.filters[j];
                  let b = &self.bias[j];
                  let output = &outputs[j];
                  for k in 0..output.nrows() {
                    for l in 0..output.ncols() {
                         let mut sum = 0.0;
                         for m in 0..filter.len() {
                              for n in 0..filter[0].nrows() {
                                for o in 0..filter[0].ncols() {
                                     sum += output[(k, l)] * filter[m][(n, o)];
                                     dinput[(k + m * self.stride, l + n * self.stride)] += sum;
                                }
                              }
                         }
                    }
                  }
                
            }
            dinputs.push(dinput.clone());
        }
        dinputs
    }
}


impl Flatten{
    pub fn new(input_shape: Vec<usize>)->Flatten{
        Flatten{
            input_shape,
        }
    }
    fn reshape(&self, inputs:DMatrix<f32>) -> DMatrix<f32> {
        let mut reshaped = DMatrix::zeros(self.input_shape[0] * self.input_shape[1], inputs.ncols());
        for i in 0..inputs.ncols() {
            for j in 0..inputs.nrows() {
                reshaped[(j + i * inputs.nrows(), 0)] = inputs[(j, i)];
            }
        } 
        reshaped
    }
    
    fn reshape_prime(&self, inputs: DMatrix<f32>, outputs: DMatrix<f32>) -> DMatrix<f32> {
        let mut dinputs = DMatrix::zeros(self.input_shape[0], self.input_shape[1]);
        for i in 0..inputs.ncols() {
            for j in 0..inputs.nrows() {
                dinputs[(j, i)] = outputs[(j + i * inputs.nrows(), 0)];
            }
        }
        dinputs
    }
}

impl LayerConfig{
    pub fn new(layer_type: LayerType, dropout: Option<Dropout>, max_pooling: Option<MaxPooling2D>, flatten: Option<Flatten>, conv2d: Option<Conv2D>, dense: Option<Dense>)->LayerConfig{
        LayerConfig{
            layer_type,
            dropout,
            max_pooling,
            flatten,
            conv2d,
            dense,
        }
    }
}