use nalgebra::{DMatrix};
use rand::prelude::*;
use std::env;

pub struct NeuralNetwork {
    weights: Vec<DMatrix<f32>>,
    bias: Vec<DMatrix<f32>>,
    neurons: Vec<usize>,
    layers: usize,
    alpha: f32,
}

impl NeuralNetwork {
    pub fn new(neurons: Vec<usize>, alpha: f32) -> Self {
        let layers = neurons.len() - 1;
        let mut weights: Vec<DMatrix<f32>> = Vec::new();
        let mut bias: Vec<DMatrix<f32>> = Vec::new();
        for i in 0..layers {
            weights.push(NeuralNetwork::xavier_init(neurons[i + 1], neurons[i]));
            bias.push(NeuralNetwork::xavier_init(neurons[i + 1], 1));
        }

        NeuralNetwork {
            weights,
            bias,
            neurons,
            layers,
            alpha,
        }
    }
    pub fn train(&mut self, input: DMatrix<f32>, truth: DMatrix<f32>, epochs: usize) {
        let mut activations = self.feedfrwd(input.clone());
        for cnt in 0..epochs {
            activations = self.feedfrwd(input.clone());
            let (d_weights, d_bias, d_layer) = self.backpropagation(activations.clone(), 
                                                                                                        DMatrix::zeros(1, 1), 
                                                                                                        input.clone(), 
                                                                                                        truth.clone(), 
                                                                                                        &mut Vec::new(), 
                                                                                                        &mut Vec::new(), 
                                                                                                        self.layers);
            
            let error = self.cost(activations[self.layers-1].clone(), truth.clone());
            if error < 1e-3 {
                println!("Epcoh: {}", cnt);
                break;
            }
            println!("Error: {}", error);
        }
    }
    pub fn xavier_init(rows: usize, cols: usize) -> DMatrix<f32> {
        let mut rng = rand::thread_rng();
        let std_dev = (2.0 / (rows + cols) as f32).sqrt();
        DMatrix::from_fn(rows, cols, |_, _| rng.gen_range(-std_dev..std_dev))
    }
    
    pub fn predict(&self, input: DMatrix<f32>) -> DMatrix<f32> {
        let activations = self.feedfrwd(input.clone());
        activations[self.layers-1].clone()
    }   
    
    pub fn cost(&self, y_hat: DMatrix<f32>, y: DMatrix<f32>) -> f32 {
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
    
    pub fn sigmoid(arr: &mut DMatrix<f32>) {
        arr.iter_mut().for_each(|x| *x = 1.0 / (1.0 + (-*x).exp()));
    }
    
    pub fn relu(arr: &mut DMatrix<f32>) {
        arr.iter_mut().for_each(|x| *x = x.max(0.0));
    }
    pub fn softmax(arr: &mut DMatrix<f32>){
        let sum: f32 = arr.iter().map(|x| x.exp()).sum();
        arr.iter_mut().for_each(|x| *x = x.exp() / sum);
    }
    
    pub fn feedfrwd(&self, input: DMatrix<f32>) -> Vec<DMatrix<f32>> {
        let mut input = input;
        let mut activations = Vec::new();
        assert_eq!(self.weights.len(), self.layers);
        assert_eq!(self.bias.len(), self.layers);

        for i in 0..self.layers {
            assert_eq!(self.weights[i].ncols(), input.nrows());
            assert_eq!(self.bias[i].nrows(), self.weights[i].nrows());

            let mut z = self.weights[i].clone() * input.clone() + NeuralNetwork::repeat_bias(&self.bias[i], input.ncols());
             if i == self.layers - 1 {
                NeuralNetwork::sigmoid(&mut z); // Use sigmoid for the output layer
            } else {
                //NeuralNetwork::relu(&mut z); // Use ReLU for hidden layers
                NeuralNetwork::sigmoid(&mut z); // Use sigmoid for the output layer

            }


            activations.push(z.clone());

            input = z;
        }
        
        activations
        
    }
    
    // Custom function to repeat bias across columns
    pub fn repeat_bias(bias: &DMatrix<f32>, num_cols: usize) -> DMatrix<f32> {
        DMatrix::from_fn(bias.nrows(), num_cols, |i, _| bias[(i, 0)])
    }
    
    pub fn backpropagation(&mut self, 
        activations: Vec<DMatrix<f32>>, 
        passthough: DMatrix<f32>,
        input: DMatrix<f32>,
        truth: DMatrix<f32>,
        der_weights: &mut Vec<DMatrix<f32>>,
         der_bias: &mut Vec<DMatrix<f32>>,
        layer: usize)
        -> (DMatrix<f32>, DMatrix<f32>, DMatrix<f32>) {

        if layer == self.layers {
            // Output layer
            let current_layer = layer - 1;
            let mut d_activation = activations[current_layer].clone() - truth.clone();
            assert_eq!(d_activation.nrows(), self.neurons[current_layer + 1]);
            assert_eq!(d_activation.ncols(), truth.ncols());

            d_activation = d_activation.map(|x| x * 2.0);
            d_activation = d_activation.map(|x| x / truth.ncols() as f32);
            assert_eq!(d_activation.nrows(), self.neurons[current_layer + 1]);
            assert_eq!(d_activation.ncols(), truth.ncols());

            // Weights derivative
            let d_weights = d_activation.clone() * activations[current_layer - 1].transpose();
            assert_eq!(d_weights.nrows(), self.neurons[current_layer + 1]);
            assert_eq!(d_weights.ncols(), self.neurons[current_layer]);

            // Bias derivative
            let d_bias = DMatrix::from_row_slice(d_weights.nrows(), 1, &(0..d_weights.nrows())
                .map(|i| d_activation.row(i).sum())
                .collect::<Vec<f32>>());
            assert_eq!(d_bias.nrows(), self.neurons[current_layer + 1]);
            assert_eq!(d_bias.ncols(), 1);

            // Layer derivative
            let d_layer = self.weights[current_layer].transpose() * d_activation;
            assert_eq!(d_layer.nrows(), self.neurons[current_layer]);
            assert_eq!(d_layer.ncols(), truth.ncols());

            self.weights[current_layer] -= self.alpha * d_weights.clone();
            self.bias[current_layer] -= self.alpha * d_bias.clone();

            der_weights.push(d_weights.clone());
            der_bias.push(d_bias.clone());

            //self.weights[current_layer] -= self.alpha * d_weights.clone();
            //self.bias[current_layer] -= self.alpha * d_bias.clone();

            let (d_weights_prev, d_bias_prev, d_layer_prev) = self.backpropagation(activations, d_layer.clone(), input.clone(), truth.clone(),der_weights, der_bias, layer - 1);
            

            return (d_weights, d_bias, d_layer);

        } else if layer - 1 == 0 {
            // Input layer
            let current_layer = layer-1;
            assert_eq!(input.nrows(), self.neurons[0]);

            let d_activations = activations[current_layer].clone().component_mul(&(activations[current_layer].clone().map(|x| x * (1.0 - x))));
            assert_eq!(d_activations.nrows(), self.neurons[current_layer + 1]);
            // dc/dz = dc/da * da/dz
            let d_activation = d_activations.component_mul(&passthough);
            assert_eq!(d_activation.nrows(), self.neurons[current_layer + 1]);
            assert_eq!(d_activation.ncols(), passthough.ncols());

            let d_weights = d_activation.clone() * input.transpose();
            assert_eq!(d_weights.nrows(), self.neurons[current_layer + 1]);
            assert_eq!(d_weights.ncols(), self.neurons[current_layer]);

            let d_bias = DMatrix::from_row_slice(d_weights.nrows(), 1, &(0..d_weights.nrows())
                .map(|i| d_activation.row(i).sum())
                .collect::<Vec<f32>>());

            assert_eq!(d_bias.nrows(), self.neurons[current_layer + 1]);
            assert_eq!(d_bias.ncols(), 1);

            let d_layer = self.weights[current_layer].transpose() * d_activation;
            assert_eq!(d_layer.nrows(), self.neurons[current_layer]);

            //println!("Weights: {:?}", der_weights[1].ncols());
            //println!("Bias: {:?}", self.weights[1].clone().ncols());

            der_weights.push(d_weights.clone());
            der_bias.push(d_bias.clone());

            der_weights.reverse();
            der_bias.reverse();
            
            for i in 0..der_weights.len()-1 {
                assert_eq!(der_weights[i].nrows(), self.weights[i].nrows());
                assert_eq!(der_weights[i].ncols(), self.weights[i].ncols());
                assert_eq!(der_bias[i].nrows(), self.bias[i].nrows());
                assert_eq!(der_bias[i].ncols(), self.bias[i].ncols());

                self.weights[i] -= self.alpha * der_weights[i].clone();
                self.bias[i] -= self.alpha * der_bias[i].clone();
            }

            //self.weights[current_layer] -= self.alpha * d_weights.clone();
            //self.bias[current_layer] -= self.alpha * d_bias.clone();

            return (d_weights, d_bias, d_layer);
            
        } else {
            // Hidden layers
            let current_layer = layer - 1;
            let d_sigmoid = activations[current_layer].clone().component_mul(&(activations[current_layer].clone().map(|x| x * (1.0 - x))));
            assert_eq!(d_sigmoid.nrows(), self.neurons[current_layer + 1]);
            // dc/dz = dc/da * da/dz
            let d_activation = d_sigmoid.component_mul(&passthough);
            assert_eq!(d_activation.nrows(), self.neurons[current_layer + 1]);
            assert_eq!(d_activation.ncols(), passthough.ncols());

            // Weights derivative
            let d_weights = d_activation.clone() * activations[current_layer - 1].transpose();
            assert_eq!(d_weights.nrows(), self.neurons[current_layer + 1]);
            assert_eq!(d_weights.ncols(), self.neurons[current_layer]);

            // Bias derivative
            let d_bias = DMatrix::from_row_slice(d_weights.nrows(), 1, &(0..d_weights.nrows())
                .map(|i| d_activation.row(i).sum())
                .collect::<Vec<f32>>());
            assert_eq!(d_bias.nrows(), self.neurons[current_layer + 1]);
            assert_eq!(d_bias.ncols(), 1);

            // Layer derivative
            let d_layer = self.weights[current_layer].transpose() * d_activation;
            assert_eq!(d_layer.nrows(), self.neurons[current_layer]);
            assert_eq!(d_layer.ncols(), passthough.ncols());

            der_weights.push(d_weights.clone());
            der_bias.push(d_bias.clone());

            let (d_weights_prev, d_bias_prev, d_layer_prev) = self.backpropagation(activations, d_layer.clone(), input.clone(), truth.clone(), der_weights, der_bias, layer - 1);

            //self.weights[current_layer] -= self.alpha * d_weights.clone();
            //self.bias[current_layer] -= self.alpha * d_bias.clone();
            

            return (d_weights, d_bias, d_layer);
        }
    }
    
}


