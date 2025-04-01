use nalgebra::{DMatrix};
use rand::prelude::*;
use std::env;

pub struct NeuralNetwork {
    weights: Vec<DMatrix<f32>>,
    bias: Vec<DMatrix<f32>>,
    accum_grad_weights: Vec<DMatrix<f32>>,
    accum_grad_bias: Vec<DMatrix<f32>>,
    momentum_weights: Vec<DMatrix<f32>>,
    momentum_bias: Vec<DMatrix<f32>>,
    velocity_weights: Vec<DMatrix<f32>>,
    velocity_bias: Vec<DMatrix<f32>>,
    neurons: Vec<usize>,
    activation: Option<Activation>,
    cost: Option<Cost>,
    optimizer: Option<Optimizer>,
    layers: usize,
    alpha: f32,
    itteration: usize,
}

pub enum Activation {
    Sigmoid,
    ReLU,
    Softmax,
}
pub enum Cost {
    MSE,
    CrossEntropy,
}

pub enum Optimizer {
    SGD,
    Adam,
    RMSprop,
}

impl NeuralNetwork {
    pub fn new(neurons: Vec<usize>, activation: Activation, cost: Cost, optimizer: Optimizer, alpha: f32) -> Self {
        let layers = neurons.len() - 1;
        let mut weights: Vec<DMatrix<f32>> = Vec::new();
        let mut bias: Vec<DMatrix<f32>> = Vec::new();

        let mut accumulated_weight_gradients: Vec<DMatrix<f32>> = Vec::new();
        let mut accumulated_bias_gradients: Vec<DMatrix<f32>> = Vec::new();

        let mut momentum_weights: Vec<DMatrix<f32>> = Vec::new();
        let mut momentum_bias: Vec<DMatrix<f32>> = Vec::new();

        let mut velocity_weights: Vec<DMatrix<f32>> = Vec::new();
        let mut velocity_bias: Vec<DMatrix<f32>> = Vec::new();

        for i in 0..layers {
            weights.push(NeuralNetwork::xavier_init(neurons[i + 1], neurons[i], Some(&activation)));
            bias.push(NeuralNetwork::xavier_init(neurons[i + 1], 1, Some(&activation)));

            accumulated_weight_gradients.push(DMatrix::zeros(neurons[i + 1], neurons[i]));
            accumulated_bias_gradients.push(DMatrix::zeros(neurons[i + 1], 1));

            momentum_weights.push(DMatrix::zeros(neurons[i + 1], neurons[i]));
            momentum_bias.push(DMatrix::zeros(neurons[i + 1], 1));

            velocity_weights.push(DMatrix::zeros(neurons[i + 1], neurons[i]));
            velocity_bias.push(DMatrix::zeros(neurons[i + 1], 1));
        }

        NeuralNetwork {
            weights,
            bias,
            accum_grad_weights: accumulated_weight_gradients,
            accum_grad_bias: accumulated_bias_gradients,
            momentum_weights,
            momentum_bias,
            velocity_weights,
            velocity_bias,
            neurons,
            activation: Some(activation),
            cost: Some(cost),
            optimizer: Some(optimizer),
            layers,
            alpha,
            itteration: 1,
        }
    }
    pub fn train(&mut self, input: Vec<DMatrix<f32>>, truth: Vec<DMatrix<f32>>, epochs: usize) {
        println!("Training on {} Images", input.len());
        let mut activations: Vec<DMatrix<f32>> = Vec::with_capacity(self.layers);
        for layer_index in 0..self.layers {
            self.accum_grad_weights[layer_index].fill(0.0);
            self.accum_grad_bias[layer_index].fill(0.0);
        }

        for cnt in 0..epochs {
            for j in 0..input.len() {
                let i = input[j].transpose();
                let y = truth[j].clone();
                self.feedfrwd(i.clone(), &mut activations);
                let (batch_d_weights, batch_d_bias) = self.backprop(activations.clone(), i, y);

                for layer_index in 0..self.layers {
                    self.accum_grad_weights[layer_index] += &batch_d_weights[layer_index]; // Accumulate weight gradients
                    self.accum_grad_bias[layer_index] += &batch_d_bias[layer_index];     // Accumulate bias gradients
                }
                
            
            }

            //gradient normalization
            //TODO: Change to work for batch size instead of input size
            for layer_index in 0..self.layers{
                self.accum_grad_weights[layer_index] /= input.len() as f32;
                self.accum_grad_bias[layer_index] /= input.len() as f32;
            }

            

            match self.optimizer {
                Some(Optimizer::SGD) => {
                    self.sgd();

                }
                Some(Optimizer::Adam) => {
                    self.adam( 0.9, 0.999, 0.000008);
                }
                Some(Optimizer::RMSprop) => {
                    self.rmsprop( 0.9);
                }
                _ => {
                    for i in 0..self.layers {

                        self.weights[i] -= self.alpha * &self.accum_grad_weights[i];
                        self.bias[i] -= self.alpha * &self.accum_grad_bias[i];
                    }
                }
            }


            let mut error = 0.0;
            for j in 0..input.len() {
                error += self.cost(&activations[self.layers-1], &truth[j]);
            }
            error = error/input.len() as f32;
            
            if error < 1e-3 {
                println!("Epcoh: {}", cnt);
                break;
            }
            if cnt % 100 == 0 {
                println!("Epoch: {}, Error: {}", cnt, error);
            }
        }
    }
    pub fn xavier_init(rows: usize, cols: usize, activation: Option<&Activation>) -> DMatrix<f32> {
        let mut rng = rand::thread_rng();
        let std_dev = match activation{
            Some(Activation::Sigmoid) | Some(Activation::Softmax) => {
                (2.0 / (rows + cols) as f32).sqrt()
            }
            Some(Activation::ReLU) => {
                (6.0 / (rows) as f32).sqrt()
                
            }
            _ => {
                (2.0 / (rows + cols) as f32).sqrt()
            }
        };
        DMatrix::from_fn(rows, cols, |_, _| rng.gen_range(-std_dev..std_dev))
    }
    
    pub fn predict(&self, input: Vec<DMatrix<f32>>) -> Vec<DMatrix<f32>> {
        let mut output: Vec<DMatrix<f32>> = Vec::new();
        for j in 0..input.len() {
            let i = input[j].transpose();
            let mut activations: Vec<DMatrix<f32>> = Vec::new();
            self.feedfrwd(i.clone(), &mut activations);
            output.push(activations[self.layers - 1].clone());
        }

        output
    }   
    
    pub fn cost(&self, y_hat: &DMatrix<f32>, y: &DMatrix<f32>) -> f32 {
        assert_eq!(y_hat.nrows(), y.nrows());
    
        let epsilon = 1e-10; // Avoid log(0)
        let y_hat = y_hat.map(|x| x.max(epsilon).min(1.0 - epsilon)); // Clipping values
        let losses = y.zip_map(&y_hat, |yi, y_hat_i| {
            -(yi * y_hat_i.ln() + (1.0 - yi) * (1.0 - y_hat_i).ln())
        });
    
        // Manually compute the mean
        let sum_of_losses: f32 = losses.sum();
        let num_elements = y.nrows() * y.ncols();
        sum_of_losses / num_elements as f32
        
    }
    
    fn sigmoid(arr: &mut DMatrix<f32>) {
        arr.iter_mut().for_each(|x| *x = 1.0 / (1.0 + (-*x).exp()));
    }

    fn sigmoid_prime(arr: &mut DMatrix<f32>) {
        arr.iter_mut().for_each(|x| *x = *x * (1.0 - *x));
    }
    
    fn relu(arr: &mut DMatrix<f32>) {
        arr.iter_mut().for_each(|x| *x = x.max(0.0));
    }

    fn relu_prime(arr: &mut DMatrix<f32>) {
        arr.iter_mut().for_each(|x| *x = if *x > 0.0 { 1.0 } else { 0.0 });
    }

    //log-sum-exp trick
    fn softmax(arr: &mut DMatrix<f32>) {
        let max_val = arr.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)); 
        let exp_values: Vec<f32> = arr.iter().map(|x| (x - max_val).exp()).collect();
        let sum_exp = exp_values.iter().sum::<f32>();
        arr.iter_mut().zip(exp_values.iter()).for_each(|(x, &exp_x)| *x = exp_x / sum_exp);
    }

    fn softmax_prime(arr: &mut DMatrix<f32>) {
        arr.iter_mut().for_each(|x| *x = *x * (1.0 - *x));
    }
    
    
    pub fn feedfrwd(&self, input: DMatrix<f32>, activations: &mut Vec<DMatrix<f32>>) {
        let mut input = input;
        assert_eq!(self.weights.len(), self.layers);
        assert_eq!(self.bias.len(), self.layers);

        // Store input activation (Important for backpropagation)
        activations.clear();  // Ensure we start fresh for each forward pass

        for i in 0..self.layers {
            assert_eq!(self.weights[i].ncols(), input.nrows());
            assert_eq!(self.bias[i].nrows(), self.weights[i].nrows());

            let mut z = &self.weights[i] * &input + NeuralNetwork::repeat_bias(&self.bias[i], input.ncols());

            // Apply activation function
            if i == self.layers - 1 {
                // Output layer activation
                match self.activation {
                    Some(Activation::Softmax) => NeuralNetwork::softmax(&mut z),
                    _ => NeuralNetwork::softmax(&mut z),
                }
            } else {
                // Hidden layer activation
                NeuralNetwork::relu(&mut z);
            }

            // Store activation
            activations.push(z.clone());

            // Update input for next layer
            input = z;
        }
        
    }
    
    // Custom function to repeat bias across columns
    pub fn repeat_bias(bias: &DMatrix<f32>, num_cols: usize) -> DMatrix<f32> {
        DMatrix::from_fn(bias.nrows(), num_cols, |i, _| bias[(i, 0)])
    }
    

    fn backprop(&mut self, activations: Vec<DMatrix<f32>>, 
        input: DMatrix<f32>,
        truth: DMatrix<f32>)-> (Vec<DMatrix<f32>>, Vec<DMatrix<f32>>) {
            let mut der_weights = Vec::new();
            let mut der_bias = Vec::new();
            let mut passthrough: DMatrix<f32> = DMatrix::zeros(self.neurons[self.layers - 1], truth.ncols());

            for cnt in (0..self.layers).rev(){
                if cnt == self.layers -1 {
                    // Output layer
                    //NeuralNetwork::softmax_prime(&mut activations[cnt].clone());
                    let mut d_activation = &activations[cnt] - &truth;
                    assert_eq!(d_activation.nrows(), self.neurons[cnt + 1]);
                    assert_eq!(d_activation.ncols(), truth.ncols());
        
                    // Weights derivative
                    let d_weights = if cnt > 0 {
                        &d_activation * &activations[cnt - 1].transpose()
                    } else {
                        &d_activation * &input.transpose()
                    };

                    assert_eq!(d_weights.nrows(), self.neurons[cnt + 1]);
                    assert_eq!(d_weights.ncols(), self.neurons[cnt]);
        
                    // Bias derivative
                    let d_bias = DMatrix::from_row_slice(d_weights.nrows(), 1, &(0..d_weights.nrows())
                        .map(|i| d_activation.row(i).sum())
                        .collect::<Vec<f32>>());
                    assert_eq!(d_bias.nrows(), self.neurons[cnt + 1]);
                    assert_eq!(d_bias.ncols(), 1);
        
                    // Layer derivative
                    passthrough  = &self.weights[cnt].transpose() * &d_activation;
                    assert_eq!(passthrough.nrows(), self.neurons[cnt]);
                    assert_eq!(passthrough.ncols(), truth.ncols());
        
                    der_weights.push(d_weights);
                    der_bias.push(d_bias);


                    continue;


                }
                else if cnt == 0{
                    // Input layer
                    assert_eq!(input.nrows(), self.neurons[0]);
                    //sigmoid prime
                    //let d_activations = &activations[cnt].component_mul(&(&activations[cnt].map(|x| x * (1.0 - x))));
                    //relu prime
                    NeuralNetwork::relu_prime(&mut activations[cnt].clone());
                    let d_activations = activations[cnt].clone();
                    assert_eq!(d_activations.nrows(), self.neurons[cnt + 1]);
                    
                    // dc/dz = dc/da * da/dz
                    let d_activation = &d_activations.component_mul(&passthrough);
                    assert_eq!(d_activation.nrows(), self.neurons[cnt + 1]);
                    assert_eq!(d_activation.ncols(), passthrough.ncols());

                    let d_weights = d_activation * &input.transpose();
                    assert_eq!(d_weights.nrows(), self.neurons[cnt + 1]);
                    assert_eq!(d_weights.ncols(), self.neurons[cnt]);

                    let d_bias = DMatrix::from_row_slice(d_weights.nrows(), 1, &(0..d_weights.nrows())
                        .map(|i| d_activation.row(i).sum())
                        .collect::<Vec<f32>>());

                    assert_eq!(d_bias.nrows(), self.neurons[cnt + 1]);
                    assert_eq!(d_bias.ncols(), 1);

                    //println!("Weights: {:?}", der_weights[1].ncols());
                    //println!("Bias: {:?}", self.weights[1].clone().ncols());

                    der_weights.push(d_weights);
                    der_bias.push(d_bias);

                    continue;

                }
                //sigmoid prime
                //let d_activations = &activations[cnt].component_mul(&(activations[cnt].clone().map(|x| x * (1.0 - x))));
                //relu prime
                NeuralNetwork::relu_prime(&mut activations[cnt].clone());
                let d_activations = activations[cnt].clone();
                    
                assert_eq!(d_activations.nrows(), self.neurons[cnt + 1]);
                // dc/dz = dc/da * da/dz
                let d_activation = d_activations.component_mul(&passthrough);
                assert_eq!(d_activation.nrows(), self.neurons[cnt + 1]);
                assert_eq!(d_activation.ncols(), passthrough.ncols());

                // Weights derivative
                let d_weights = &d_activation * &activations[cnt - 1].transpose();
                assert_eq!(d_weights.nrows(), self.neurons[cnt + 1]);
                assert_eq!(d_weights.ncols(), self.neurons[cnt]);

                // Bias derivative
                let d_bias = DMatrix::from_row_slice(d_weights.nrows(), 1, &(0..d_weights.nrows())
                    .map(|i| d_activation.row(i).sum())
                    .collect::<Vec<f32>>());
                assert_eq!(d_bias.nrows(), self.neurons[cnt + 1]);
                assert_eq!(d_bias.ncols(), 1);

                // Layer derivative
                passthrough = &self.weights[cnt].transpose() * d_activation;
                assert_eq!(passthrough.nrows(), self.neurons[cnt]);

                der_weights.push(d_weights);
                der_bias.push(d_bias);

            }

            der_weights.reverse();
            der_bias.reverse();

            (der_weights, der_bias)
            
        }


        fn adam(&mut self, beta1:f32, beta2:f32, epsilon:f32){
            self.itteration += 1;

            
            for i in 0..self.layers{
                self.momentum_weights[i] = beta1 * &self.momentum_weights[i] + (1.0 - beta1) * &self.accum_grad_weights[i];
                self.momentum_bias[i] = beta1 * &self.momentum_bias[i] + (1.0 - beta1) * &self.accum_grad_bias[i];

                let momentum_weights_hat = &self.momentum_weights[i] / (1.0 - beta1.powf(self.itteration as f32));
                let momentum_bias_hat = &self.momentum_bias[i] / (1.0 - beta1.powf(self.itteration as f32));

                self.velocity_weights[i] = beta2 * &self.velocity_weights[i] + (1.0 - beta2) * &self.accum_grad_weights[i].map(|x| x.powi(2));
                self.velocity_bias[i] = beta2 * &self.velocity_bias[i] + (1.0 - beta2) * &self.accum_grad_bias[i].map(|x| x.powi(2));

                let velocity_weights_hat = &self.velocity_weights[i] / (1.0 - beta2.powf(self.itteration as f32));
                let velocity_bias_hat = &self.velocity_bias[i] / (1.0 - beta2.powf(self.itteration as f32));

                self.weights[i] -= self.alpha * &momentum_weights_hat.zip_map(&velocity_weights_hat, |m, v| m / (v.sqrt() + epsilon));
                self.bias[i] -= self.alpha * &momentum_bias_hat.zip_map(&velocity_bias_hat, |m, v| m / (v.sqrt() + epsilon));

            }

            
            
        }

        
        fn rmsprop(&mut self, beta:f32){

            for i in 0..self.layers {
                self.velocity_weights[i] = beta * &self.velocity_weights[i] + (1.0 - beta) * &self.accum_grad_weights[i];
                self.velocity_bias[i] = beta * &self.velocity_bias[i] + (1.0 - beta) * &self.accum_grad_bias[i];

                self.weights[i] -= self.alpha * &self.accum_grad_weights[i].component_div(&(&self.velocity_weights[i]).map(|x| (x+ 1e-8).sqrt()));
                self.bias[i] -= self.alpha * &self.accum_grad_bias[i].component_div(&(&self.velocity_bias[i]).map(|x| (x + 1e-8).sqrt()));
                
            }
        }

        fn sgd(&mut self){
            for i in 0..self.layers {
                self.weights[i] -= self.alpha * &self.accum_grad_weights[i];
                self.bias[i] -= self.alpha * &self.accum_grad_bias[i];
            }
        }
    
}



/*
    pub fn backpropagation_recersive(&mut self, 
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

            let (d_weights_prev, d_bias_prev, d_layer_prev) = self.backpropagation_recersive(activations, d_layer.clone(), input.clone(), truth.clone(),der_weights, der_bias, layer - 1);
            

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

            let (d_weights_prev, d_bias_prev, d_layer_prev) = self.backpropagation_recersive(activations, d_layer.clone(), input.clone(), truth.clone(), der_weights, der_bias, layer - 1);

            //self.weights[current_layer] -= self.alpha * d_weights.clone();
            //self.bias[current_layer] -= self.alpha * d_bias.clone();
            

            return (d_weights, d_bias, d_layer);
        }
    }
*/
