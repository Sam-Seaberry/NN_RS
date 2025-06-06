use nalgebra::{DMatrix, RowDVector, DVector};
use rand::prelude::*;
use std::env;
use rayon::prelude::*;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::task;
use crate::preprocess;

pub struct NeuralNetwork<'a> {
    weights: Vec<DMatrix<f32>>,
    bias: Vec<DMatrix<f32>>,
    accum_grad_weights: Vec<DMatrix<f32>>,
    accum_grad_bias: Vec<DMatrix<f32>>,
    momentum_weights: Vec<DMatrix<f32>>,
    momentum_bias: Vec<DMatrix<f32>>,
    velocity_weights: Vec<DMatrix<f32>>,
    velocity_bias: Vec<DMatrix<f32>>,
    activations: Vec<DMatrix<f32>>,
    running_mean: Vec<DVector<f32>>,
    running_var: Vec<DVector<f32>>,
    gamma: Vec<DVector<f32>>,
    beta: Vec<DVector<f32>>,
    neurons: Vec<usize>,
    activation: Option<Activation>,
    cost: Option<Cost>,
    optimizer: Option<Optimizer>,
    layers: usize,
    alpha: f32,
    itteration: usize,
    _phantom: std::marker::PhantomData<&'a ()>,
}
#[derive(Clone)]
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



impl<'a> NeuralNetwork<'a> {
    pub fn new(neurons: Vec<usize>, activation: Activation, cost: Cost, optimizer: Optimizer, alpha: f32) -> Self {
        let layers = neurons.len() - 1;
        let mut weights: Vec<DMatrix<f32>> = Vec::new();
        let mut bias: Vec<DMatrix<f32>> = Vec::new();

        let mut gamma: Vec<DVector<f32>> = Vec::new();
        let mut beta: Vec<DVector<f32>> = Vec::new();

        let mut running_mean: Vec<DVector<f32>> = Vec::new();
        let mut running_var: Vec<DVector<f32>> = Vec::new();

        let mut accumulated_weight_gradients: Vec<DMatrix<f32>> = Vec::new();
        let mut accumulated_bias_gradients: Vec<DMatrix<f32>> = Vec::new();

        let mut momentum_weights: Vec<DMatrix<f32>> = Vec::new();
        let mut momentum_bias: Vec<DMatrix<f32>> = Vec::new();

        let mut velocity_weights: Vec<DMatrix<f32>> = Vec::new();
        let mut velocity_bias: Vec<DMatrix<f32>> = Vec::new();

        let mut activations: Vec<DMatrix<f32>> = Vec::with_capacity(layers);

        // Fill the vector with default values
        activations.extend(vec![DMatrix::zeros(0, 0); layers]);

        NeuralNetwork {
            weights,
            bias,
            accum_grad_weights: accumulated_weight_gradients,
            accum_grad_bias: accumulated_bias_gradients,
            momentum_weights,
            momentum_bias,
            velocity_weights,
            velocity_bias,
            activations,
            running_mean,
            running_var,
            gamma,
            beta,
            neurons,
            activation: Some(activation),
            cost: Some(cost),
            optimizer: Some(optimizer),
            layers,
            alpha,
            itteration: 1,
            _phantom: std::marker::PhantomData,
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


    pub fn train(&mut self, input: Vec<DMatrix<f32>>, truth: Vec<DMatrix<f32>>, test_input: Vec<DMatrix<f32>>, test_truth: Vec<DMatrix<f32>>, epochs: usize, batch: usize) {
        for i in input[0].nrows()-1..self.layers+input[0].nrows()-1 {
            self.weights.push(NeuralNetwork::xavier_init(self.neurons[i + 1], self.neurons[i], &self.activation));
            self.bias.push(NeuralNetwork::xavier_init(self.neurons[i + 1], 1, &self.activation));

            self.gamma.push(DVector::from_element(self.neurons[i+1], 1.0));
            self.beta.push(DVector::from_element(self.neurons[i+1], 0.0));

            self.accum_grad_weights.push(DMatrix::zeros(self.neurons[i + 1], self.neurons[i]));
            self.accum_grad_bias.push(DMatrix::zeros(self.neurons[i + 1], 1));

            self.momentum_weights.push(DMatrix::zeros(self.neurons[i + 1], self.neurons[i]));
            self.momentum_bias.push(DMatrix::zeros(self.neurons[i + 1], 1));

            self.velocity_weights.push(DMatrix::zeros(self.neurons[i + 1], self.neurons[i]));
            self.velocity_bias.push(DMatrix::zeros(self.neurons[i + 1], 1));

            self.running_mean.push(DVector::from_element(self.neurons[i+1], 0.0));
            self.running_var.push(DVector::from_element(self.neurons[i+1], 0.0));

        }
        
        
        println!("Training on {} Images", input.len());


        for cnt in 0..epochs {
            let mut current = 0;
            self.itteration = 1;

            

            while current <= input.len() - 1{
                for layer_index in 0..self.layers {
                    self.accum_grad_weights[layer_index].fill(0.0);
                    self.accum_grad_bias[layer_index].fill(0.0);
                }
                let mut next = current + batch;
                if next >= input.len() {
                    next = input.len() - 1;
                }
                
                /*for j in current..next {
                    
                    let i = input[j].transpose();
                    
                    //NeuralNetwork::show_data(&input[j], &truth[j]);
                    let y = truth[j].clone();
                    self.feedfrwd(&i);
                    let (batch_d_weights, batch_d_bias) = self.backprop_test(&i, &y);
                    

                    for layer_index in 0..self.layers {
                        self.accum_grad_weights[layer_index] += &batch_d_weights[layer_index]; // Accumulate weight gradients
                        self.accum_grad_bias[layer_index] += &batch_d_bias[layer_index]; // Accumulate bias gradients
                        
                        
    
                    
                    }
    
                    //self.print_activation_stats();
    
                    //self.print_grad_stats();
                    
                
                }
                panic!{};*/
                let (mean,xhat, sdev) = self.feedfrwd_batched(&input[current..next].to_vec(), true);

                let (d_gamma, d_beta) = self.backprop_batched(&input[current..next].to_vec(), &truth[current..next].to_vec(), mean, xhat, sdev);
                
    
                //gradient normalization
                //TODO: Change to work for batch size instead of input size

                
                //NeuralNetwork::print_grad_stats_batch(&d_gamma);
                //NeuralNetwork::print_grad_stats_batch(&d_beta);

                
                
                for layer_index in 0..self.layers{
                    println!("Layer index {}", layer_index);
                    //self.accum_grad_weights[layer_index] /= batch as f32;
                    //self.accum_grad_bias[layer_index] /= batch as f32;

                    if layer_index < self.layers -1{

                        let gamma_reg_term = self.gamma[layer_index].map(|g| 0.0001 * g);
                        let beta_reg_term = self.beta[layer_index].map(|b| 0.0001 * b);

                        let d_gamma_layer = &d_gamma[layer_index].column_mean() + &gamma_reg_term;
                        let d_beta_layer = &d_beta[layer_index].column_mean() + &beta_reg_term;

                        self.gamma[layer_index] -= self.alpha * d_gamma_layer;
                        self.beta[layer_index] -= self.alpha * d_beta_layer;
                    }
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
                /*for j in 0..input.len() {
                    error += self.cost(&self.activations[self.layers-1], &truth[j]);
                }*/

                error = self.cost_batched(&self.activations[self.layers - 1], &truth[current..next].to_vec());


                error = error/input.len() as f32;
                
                //if error < 1e-3 {
                  //  println!("Epcoh: {}", cnt);
                    //break;
                //}
                if cnt == 2{

                    panic!();

                }
                self.test_accuracy_batched(&test_input, &test_truth);

                println!("Epoch: {}, Error: {:?}, itteration {}", cnt, error, &self.itteration);

    
                
                if cnt % 1 == 0 {
                    self.print_weight_stats();
                    self.print_activation_stats();
                    self.print_grad_stats();
                    //NeuralNetwork::print_grad_stats_batch(&self.beta);

                    //self.count_zeros();
                    //self.count_zeros_weights();
                    
                }
                
                /*if cnt == epochs/2{
                    self.alpha = self.alpha * 0.5;
                }*/
                //println!("Final Layer Activation: {}", self.activations[self.layers - 1].column(0));
                current += batch;

            }

        }
            
    }

    fn show_data(input: &DMatrix<f32>, truth: &DMatrix<f32>){
        println!("--------------------------------");
        NeuralNetwork::display_mnist_matrix(&preprocess::create_ndmatrix_from_mnist_image(&&preprocess::convert_to_mnist_image(input.clone(), vec![0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]), vec![28, 28]));
        println!("Truth {:?}", truth);
        println!("--------------------------------");
    }

    //output should be small for gradients but not tiny if too small increase learning rate 
    // or introduce gradient clipping
    fn print_grad_stats(&self) {
        for (i, grad) in self.accum_grad_weights.iter().enumerate() {
            let mean = grad.mean();
            let max = grad.max();
            let min = grad.min();
            println!("Layer {} - Grad Mean: {:.6}, Max: {:.6}, Min: {:.6}", i, mean, max, min);
        }
    }
    fn print_grad_stats_batch(grads: &Vec<DMatrix<f32>>) {
        for (i, grad) in grads.iter().enumerate() {
            let mean = grad.mean();
            let max = grad.max();
            let min = grad.min();
            println!("Layer {} - Grad Mean: {:.6}, Max: {:.6}, Min: {:.6}", i, mean, max, min);
        }
    }
    //These values should not change massively or be very large/small
    fn print_weight_stats(&self) {
        for (i, w) in self.weights.iter().enumerate() {
            println!("Layer {} - Weight Norm: {:.6}", i, w.norm());
        }
    }
    // Max/Mean should not be zero 
    fn print_activation_stats(&self) {
        for (i, act) in self.activations.iter().enumerate() {
            let mean = act.mean();
            let max = act.max();
            let min = act.min();
            println!("Layer {} - Activation Mean: {:.6}, Max: {:.6}, Min: {:.6}", i, mean, max, min);
        }
    }

    fn count_zeros(&self) {
        for (i, act) in self.activations.iter().enumerate() {
            let zeros = act.iter().filter(|&&x| x==0.0).count();
            let total = act.len();
            println!("Layer {} - Activation Dead Neuron Count {} of {}", i, zeros, total);
        }
    } 

    fn count_zeros_weights(&self){
        for (i, act) in self.weights.iter().enumerate(){
            let zeros = act.iter().filter(|&&x| x==0.0).count();
            println!("Weights Layer {} - Dead Neuron Count {} of {}", i, zeros, act.len());
        }
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

    fn test_accuracy(&mut self, test_input: &Vec<DMatrix<f32>>, test_truth: &Vec<DMatrix<f32>>){
        let results = self.predict(test_input.to_vec());
        let mut truthcnt = 0;
        let mut last_guess = [0,0,0,0,0,0,0,0,0,0];
        let mut correct_guess = [0,0,0,0,0,0,0,0,0,0];

        for i in 0..results.len() {
            let (max_value, max_index) = NeuralNetwork::find_max_and_index(&results[i]);
            let (_max_value_truth, max_index_truth) = NeuralNetwork::find_max_and_index(&test_truth[i]);
            println!("---------------------------------------------------");
            last_guess[max_index.0] += 1;
            if max_index.0 as i32 == max_index_truth.0 as i32{
                truthcnt += 1;
                correct_guess[max_index.0] += 1;
                println!("Correct");
            }else{
                println!("Incorrect");
            }
            

            println!("Guess {}", results[i]);
            //println!("Max index output: {:?}, max index truth {:?}", max_index.0, max_index_truth.0);
            //println!("Output: {}, Value: {:.3}, Index: {:?}", i, max_value, max_index);
            println!("Truth: {:?}", test_truth[i]);
            println!("--------------------------------------------------- \n\n");
        }
        println!("Accuracy: {}, Guess Count: {:?}, Correct Count: {:?}", truthcnt as f32 / results.len() as f32, last_guess, correct_guess);
    }

    fn test_accuracy_batched(&mut self, test_input: &Vec<DMatrix<f32>>, test_truth: &Vec<DMatrix<f32>>){
        let results = self.predict_batched(&test_input.to_vec());
        let mut truthcnt = 0;
        let mut last_guess = [0,0,0,0,0,0,0,0,0,0];
        let mut correct_guess = [0,0,0,0,0,0,0,0,0,0];

        for (cnt, i) in results.column_iter().enumerate() {
            //let (max_value, max_index) = NeuralNetwork::find_max_and_index(i);
            if let Some((max_index, &max_value)) = i.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()) {
                let (_max_value_truth, max_index_truth) = NeuralNetwork::find_max_and_index(&test_truth[cnt]);
                //println!("---------------------------------------------------");
                //println!("guess {}, \n truth {}", i, test_truth[cnt]);
                last_guess[max_index] += 1;
                if max_index as i32 == max_index_truth.0 as i32{
                    truthcnt += 1;
                    correct_guess[max_index] += 1;
                    //println!("Correct");
                }else{
                    //println!("Incorrect");
                }
            } else {
                println!("Vector is empty.");
            }
            
            

            
            //println!("Max index output: {:?}, max index truth {:?}", max_index.0, max_index_truth.0);
            //println!("Output: {}, Value: {:.3}, Index: {:?}", i, max_value, max_index);
            //println!("Truth: {:?}", test_truth[i]);
            //println!("--------------------------------------------------- \n\n");
        }
        println!("Accuracy: {}, Guess Count: {:?}, Correct Count: {:?}", truthcnt as f32 / results.len() as f32, last_guess, correct_guess);
    }



    pub fn xavier_init(rows: usize, cols: usize, activation: &Option<Activation>) -> DMatrix<f32> {
        let mut rng = rand::thread_rng();
        let std_dev = match activation{
            Some(Activation::Sigmoid) | Some(Activation::Softmax) => {
                (2.0 / (rows + cols) as f32).sqrt()
            }
            Some(Activation::ReLU) => {
                (6.0 / (cols) as f32).sqrt()
                
            }
            _ => {
                (2.0 / (rows + cols) as f32).sqrt()
            }
        };
        DMatrix::from_fn(rows, cols, |_, _| rng.gen_range(-std_dev..std_dev))
    }
    
    pub fn predict(&mut self, input: Vec<DMatrix<f32>>) -> Vec<DMatrix<f32>> {
        let mut output: Vec<DMatrix<f32>> = Vec::new();
        for j in 0..input.len() {
            let i = input[j].transpose();
            self.feedfrwd(&i);
            output.push(self.activations[self.layers - 1].clone());
        }

        output
    }   

    pub fn predict_batched(&mut self, input: &Vec<DMatrix<f32>>)->DMatrix<f32>{
        let mut output: Vec<DMatrix<f32>> = Vec::new();
        self.feedfrwd_batched(&input, false);
        assert_eq!(self.activations[self.layers - 1].ncols(), input.len());

        self.activations[self.layers -1].clone()
    }
    
    fn cost(&self, y_hat: &DMatrix<f32>, y: &DMatrix<f32>) -> f32 {
        assert_eq!(y_hat.nrows(), y.nrows());
    
        let epsilon = 1e-10; // Avoid log(0)
        let y_hat = y_hat.map(|x| x.max(epsilon).min(1.0 - epsilon)); // Clipping values

        let losses = match self.activation {

            Some(Activation::Softmax) => y.zip_map(&y_hat, |yi, y_hat_i| -(yi * y_hat_i.ln())),

            Some(Activation::Sigmoid) => y.zip_map(&y_hat, |yi, y_hat_i| { -(yi * y_hat_i.ln() + (1.0 - yi) * (1.0 - y_hat_i).ln())}),

            _ => y.zip_map(&y_hat, |yi, y_hat_i| -(yi * y_hat_i.ln() + (1.0 - yi) * (1.0 - y_hat_i).ln())),

        };
       
        let sum_of_losses: f32 = losses.sum();
        let num_elements = y.nrows() * y.ncols();
        sum_of_losses / num_elements as f32
        
    }
    fn cost_batched(&self, y_hat: &DMatrix<f32>, y: &Vec<DMatrix<f32>>) -> f32 {
        assert_eq!(y_hat.ncols(), y.len());
    
        let epsilon = 1e-10; // Avoid log(0)
        let y_hat:DMatrix<f32> = DMatrix::from_columns(&y_hat.column_iter().map(|x| x.map(|z| z.max(epsilon).min(1.0 - epsilon))).collect::<Vec<_>>()); // Clipping values   
        //println!("Yhat shape {:?}, turth shape {:?}", y_hat.column(0).shape(), y[0].shape());     

        let losses = match self.activation {

            Some(Activation::Softmax) => DMatrix::from_columns(&y_hat.column_iter().enumerate().map(|(cnt, val)| -(&y[cnt].component_mul(&val.map(|x| x.ln())))).collect::<Vec<_>>()),

            Some(Activation::Sigmoid) => DMatrix::from_columns(&y_hat.column_iter().enumerate().map(|(cnt, val)| -(&y[cnt].component_mul(&val.map(|x| x.ln())) + (y[cnt].map(|x| 1.0 + x)).component_mul(&val.map(|x| (1.0 - x).ln())))).collect::<Vec<_>>()),

            _ => DMatrix::from_columns(&y_hat.column_iter().enumerate().map(|(cnt, val)| -(&y[cnt].component_mul(&val.map(|x| x.ln())) + (y[cnt].map(|x| 1.0 + x)).component_mul(&val.map(|x| (1.0 - x).ln())))).collect::<Vec<_>>()),

        };
       
        let sum_of_losses = losses.sum();
        let num_of_elements = y.len();
        //println!("sum of cols {:?}, shape {:?}", sum_of_losses.len(), sum_of_losses.shape());
        sum_of_losses / num_of_elements as f32
        
    }
    
    fn sigmoid(arr: &mut DMatrix<f32>) {
        arr.iter_mut().for_each(|x| *x = 1.0 / (1.0 + (-*x).exp()));
    }

    fn sigmoid_prime_return(mut arr: DMatrix<f32>)->DMatrix<f32>{
        let met = arr.map(|x| x * (1.0 - x));
        met
    }
    
    fn relu(arr: &mut DMatrix<f32>) {
        arr.iter_mut().for_each(|x| *x = x.max(0.0));
    }

    fn leakyrelu(arr: &mut DMatrix<f32>) {
        arr.iter_mut().for_each(|x| *x = x.max(*x*0.01));
    }

    fn leakyrelu_prime(mut arr: DMatrix<f32>)->DMatrix<f32> {
        let met = arr.map(|x| if x > 0.0 {1.0} else {0.01});
        met
    }

    fn relu_prime(arr: &mut DMatrix<f32>) {
        arr.iter_mut().for_each(|x| *x = if *x > 0.0 { 1.0 } else { 0.0 });
    }

    fn relu_prime_return(mut arr: DMatrix<f32>)->DMatrix<f32>{
        let met = arr.map(|x| if x > 0.0 { 1.0 } else { 0.0 });
        met
    }

    //log-sum-exp trick
    fn softmax(arr: &mut DMatrix<f32>) {
        let max_val = arr.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)); 
        let exp_values: Vec<f32> = arr.iter().map(|x| (x - max_val).exp()).collect();
        let sum_exp = exp_values.iter().sum::<f32>();
        arr.iter_mut().zip(exp_values.iter()).for_each(|(x, &exp_x)| *x = exp_x / sum_exp);
    }

    fn softmax_batched(arr: &mut DMatrix<f32>) {
        for mut i in arr.column_iter_mut(){
            let max_val = i.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)); 
            let exp_values: Vec<f32> = i.iter().map(|x| (x - max_val).exp()).collect();
            let sum_exp = exp_values.iter().sum::<f32>();
            i.iter_mut().zip(exp_values.iter()).for_each(|(x, &exp_x)| *x = exp_x / sum_exp);

        }
        
    }

    fn softmax_prime_return(mut arr: DMatrix<f32>)->DMatrix<f32>{
        let met = arr.map(|x| x * (1.0 - x));
        met
    }
    
    fn feedfrwd_concurrent(&mut self, input: DMatrix<f32>, weights: &Vec<DMatrix<f32>>, bias: &Vec<DMatrix<f32>>) {
        let mut input = input;
        assert_eq!(weights.len(), bias.len());

        for i in 0..weights.len() {
            assert_eq!(weights[i].ncols(), input.nrows());
            assert_eq!(bias[i].nrows(), weights[i].nrows());

            let mut z = &weights[i] * &input + NeuralNetwork::repeat_bias(&bias[i], input.ncols());

            if i == weights.len() - 1 {
                NeuralNetwork::softmax(&mut z);
            } else {
                // Hidden layer activation
                NeuralNetwork::leakyrelu(&mut z);
            }

            // Store activation
            self.activations[i] = z.clone();

            // Update input for next layer
            input = z;
        }
    }
    
    fn feedfrwd(&mut self, input: &DMatrix<f32>) {
        let mut input = input.clone();
        assert_eq!(self.weights.len(), self.layers);
        assert_eq!(self.bias.len(), self.layers);

        /*if self.itteration > 1 {
            self.activations.clear();  // Ensure we start fresh for each forward pass

        } */
        for matrix in self.activations.iter_mut() {
            matrix.fill(0.0);
        }

        for i in 0..self.layers {
            assert_eq!(self.weights[i].ncols(), input.nrows());
            assert_eq!(self.bias[i].nrows(), self.weights[i].nrows());

            //let mut z = &self.weights[i] * &input + NeuralNetwork::repeat_bias(&self.bias[i], input.ncols());
            let mut z = &self.weights[i] * &input + &self.bias[i]; 

            //insert batch norm here

            if i == self.layers - 1 {
                match self.activation {
                    Some(Activation::Softmax) => {
                        NeuralNetwork::softmax(&mut z);
                        
                    }
                    Some(Activation::Sigmoid) => {
                        NeuralNetwork::sigmoid(&mut z);
                    }
                    _=>{
                        NeuralNetwork::softmax(&mut z);
                    }
                }
            } else {
                // Hidden layer activation
                NeuralNetwork::leakyrelu(&mut z);
            }

            // Store activation
            self.activations[i] = z.clone();
            println!("Activation sizes: Layer: {}, Shape{:?}", i, z.shape());

            // Update input for next layer
            input = z;
        }
        
    }
    
    fn feedfrwd_batched(&mut self, input: &Vec<DMatrix<f32>>, training: bool)->(Vec<DMatrix<f32>>, Vec<DMatrix<f32>>, Vec<DVector<f32>>) {
        let mut input_mut = input.clone();
        assert_eq!(self.weights.len(), self.layers);
        assert_eq!(self.bias.len(), self.layers);

        /*if self.itteration > 1 {
            self.activations.clear();  // Ensure we start fresh for each forward pass

        } */
        //for matrix in self.activations.iter_mut() {
        //    matrix.fill(0.0);
        //}
        let mut centered: Vec<DMatrix<f32>> = Vec::new();
        let mut x_hat: Vec<DMatrix<f32>> = Vec::new();
        let mut sdev: Vec<DVector<f32>> = Vec::new();
        let mut mean: Vec<DVector<f32>> = Vec::new();

        

        for i in 0..self.layers{

            let mut xi:Vec<DMatrix<f32>>= Vec::new();
            let mut stacked:DMatrix<f32> = DMatrix::zeros(0, 0);

            println!("Weigths Mean {:?}, Max {:?}, MIN {:?}",self.weights[i].mean(), self.weights[i].max(), self.weights[i].min());

            if i>0{
                stacked = DMatrix::from_columns(&input_mut[0].column_iter().map(|m| &self.weights[i] * m + &self.bias[i]).collect::<Vec<_>>());
                //println!("Stacked shape {:?} layer{}", stacked.shape(), i);
            }else{
                xi = input_mut.iter()
                    .map(|x| &self.weights[i] * x.transpose() + &self.bias[i])
                    .collect();
                
                stacked= DMatrix::from_columns(&xi.iter().map(|m| m.column(0)).collect::<Vec<_>>());
                //println!("Stacked shape {:?} layer{}", stacked.shape(), i);

            }
            // Step 1: Compute pre-activations for all batch inputs
            //matrix in from batch size x neurons. where each new matrix is a column contaning the activations for each neuron
            println!("-----------------------------------------------------------------------------------------------");
            println!("Layer {}", i);
            println!("stacked shape {:?}", stacked.shape());
            // Step 2: Stack into a matrix: [features x batch_size]
            //turns Vec<DMatrix<f32>> into a DMatrix<f32>. essentially just combining the columns of each matrix into a single matrix
            println!("Stacked Mean {:?}, Max {:?}, MIN {:?}",stacked.mean(), stacked.max(), stacked.min());

            mean.push(if training {
                                                    stacked.column_mean()
                                                }else{
                                                    self.running_mean[i].clone()
                                                });

            sdev.push(if training {
                                            stacked.column_variance()
                                        }else{
                                            self.running_var[i].clone()
                                        });

            if training{
                self.running_var[i] = self.running_var[i].map(|x| x * 0.9) + &sdev[i].map(|y| y * 0.1);
                self.running_mean[i] = self.running_mean[i].map(|x| x * 0.9) + &mean[i].map(|y| y * 0.1);

            }

            println!("sdev Mean {:?}, Max {:?}, MIN {:?}",sdev[i].mean(), sdev[i].max(), sdev[i].min());

            let result = DMatrix::from_columns(&stacked.column_iter().map(|y| (y - &mean[i]).component_div(&sdev[i])).collect::<Vec<_>>());
            let mut batched = DMatrix::from_columns(&result.column_iter().map(|x| self.gamma[i].component_mul(&x) + &self.beta[i]).collect::<Vec<_>>());

            //println!("mean Mean {:?}, Max {:?}, MIN {:?}",mean.mean(), mean.max(), mean.min());
            //println!("mean shape {:?}", mean.shape());

            //centered.push(DMatrix::from_fn(stacked.nrows(), stacked.ncols(), |t, j| stacked[(t, j)] - mean[j])); 
            centered.push(DMatrix::from_columns(&stacked.column_iter().map(|x| x - &mean[i]).collect::<Vec<_>>()));
            //println!("centered Mean {:?}, Max {:?}, MIN {:?}",centered[i].mean(), centered[i].max(), centered[i].min());
            //println!("CEntered shape {:?}", centered[i].shape());

            //var.push(centered[i].component_mul(&centered[i]).column_sum().map(|x| x/input[0].ncols() as f32));
            let var = sdev[i].map(|x| x + 1e-8);

            //println!("Var Mean {:?}, Max {:?}, MIN {:?}",var[i].mean(), var[i].max(), var[i].min());
            
            //println!("Var shape {:?}, LEN {}", var[i].shape(), var.len());
           

            let sqvar = var.map(|v| v.sqrt());
            //println!("sqvar Mean {:?}, Max {:?}, MIN {:?}",sqvar[i].mean(), sqvar[i].max(), sqvar[i].min());
            

            let ivar = sqvar.map(|v| 1.0 / v);
            //println!("ivar Mean {:?}, Max {:?}, MIN {:?}",ivar[i].mean(), ivar[i].max(), ivar[i].min());
            

            x_hat.push(DMatrix::from_columns(&centered[i].column_iter().map(|x| x.component_mul(&ivar)).collect::<Vec<_>>()));
            //println!("xhat Mean {:?}, Max {:?}, MIN {:?}",x_hat[i].mean(), x_hat[i].max(), x_hat[i].min());
            
            //let mut yi = DMatrix::from_columns(&x_hat[i].column_iter().map(|col| col.component_mul(&self.gamma[i]) + &self.beta[i]).collect::<Vec<_>>());

            //let yi = DMatrix::from_columns(&x_hat[i].column_iter().map(|col| col.clone() * self.gamma[i] + self.beta[i]).collect::<Vec<_>>());
            
            //let mut yt = DMatrix::from_row_slice(yi.nrows(), 1, &yi.column_sum().as_slice());
            //yt /= input.len() as f32;
            

            /*for mut t in yi.clone(){
                if i == self.layers - 1 {
                    match self.activation {
                        Some(Activation::Softmax) => {
                            NeuralNetwork::softmax(  &mut t);
                        }
                        Some(Activation::Sigmoid) => {
                            NeuralNetwork::sigmoid(&mut t);
                        }
                        _=>{
                            NeuralNetwork::softmax(&mut t);
                        }
                    };
                } else {
                    // Hidden layer activation
                    NeuralNetwork::leakyrelu(&mut t);
                }
            }*/
            
            if i == self.layers - 1 {
                    match self.activation {
                        Some(Activation::Softmax) => {
                            NeuralNetwork::softmax_batched(  &mut x_hat[i]);
                            self.activations[i] = x_hat[i].clone();
                            //println!("activations sizes {:?}, len {}", self.activations[i].shape(), self.activations.len());
                            //println!("ACtivation mean {:?}, max {:?}, min {:?}", self.activations[i].mean(), self.activations[i].max(), self.activations[i].min());

                            input_mut = vec![x_hat[i].clone()];
                        }
                        Some(Activation::Sigmoid) => {
                            NeuralNetwork::softmax_batched(  &mut x_hat[i]);
                            self.activations[i] = x_hat[i].clone();
                           // println!("activations sizes {:?}, len {}", self.activations[i].shape(), self.activations.len());
                            //println!("ACtivation mean {:?}, max {:?}, min {:?}", self.activations[i].mean(), self.activations[i].max(), self.activations[i].min());

                            input_mut = vec![x_hat[i].clone()];
                        }
                        _=>{
                            NeuralNetwork::softmax_batched(&mut x_hat[i]);
                            self.activations[i] = x_hat[i].clone();
                            //println!("activations sizes {:?}, len {}", self.activations[i].shape(), self.activations.len());
                            //println!("ACtivation mean {:?}, max {:?}, min {:?}", self.activations[i].mean(), self.activations[i].max(), self.activations[i].min());

                            input_mut = vec![x_hat[i].clone()];
                        }
                    };
            } else {
                // Hidden layer activation
                NeuralNetwork::leakyrelu(&mut batched);
                self.activations[i] = batched.clone();
                //println!("ACtivation mean {:?}, max {:?}, min {:?}", self.activations[i].mean(), self.activations[i].max(), self.activations[i].min());

                //println!("activations sizes {:?}, len {}", self.activations[i].shape(), self.activations.len());
                input_mut = vec![batched.clone()];
            }

            //println!("-----------------------------------------------------------------------------------------------\n");

            
            
            //println!("Results shape {:?}", result.shape());
            

            

            
            
        }

        (centered, x_hat, sdev)
        
        
    }
    
    // Custom function to repeat bias across columns
    pub fn repeat_bias(bias: &DMatrix<f32>, num_cols: usize) -> DMatrix<f32> {
        DMatrix::from_fn(bias.nrows(), num_cols, |i, _| bias[(i, 0)])
    }


    fn backprop_test(&mut self, 
        input: &DMatrix<f32>,
        truth: &DMatrix<f32>)-> (Vec<DMatrix<f32>>, Vec<DMatrix<f32>>) {
            let mut der_weights = Vec::new();
            let mut der_bias = Vec::new();
            let mut passthrough: DMatrix<f32> = DMatrix::zeros(self.neurons[self.layers - 1], truth.ncols());
            

            for cnt in (0..self.layers).rev(){
                if cnt == self.layers -1 {
                    // Output layer
                    //Activation functions assume BCE with one-hot encoding
                    let d_activation = match self.activation {
                        Some(Activation::Softmax) => {
                            self.activations[cnt].clone() - truth //.component_mul(&NeuralNetwork::softmax_prime_return(self.activations[cnt].clone()));
                        }
                        Some(Activation::Sigmoid) => {
                            self.activations[cnt].clone() - truth
                        }
                        _=>{
                            self.activations[cnt].clone() - truth
                        }
                        
                    };
                    
                    // Weights derivative
                    let d_weights = (&d_activation * &self.activations[cnt - 1].transpose()).map(|x| x / d_activation.nrows() as f32);
                
                    // Bias derivative
                    let d_bias = DMatrix::from_row_slice(d_weights.nrows(), 1, &(0..d_weights.nrows())
                        .map(|i| d_activation.row(i).sum() / d_activation.nrows() as f32)
                        .collect::<Vec<f32>>());

                    // Layer derivative
                    passthrough = &self.weights[cnt].transpose() * &d_activation;
                    //println!("dweignts {:?}", d_weights.shape());

        
                    der_weights.push(d_weights);
                    der_bias.push(d_bias);



                }else if cnt == 0{
                    
                    let d_activation = NeuralNetwork::leakyrelu_prime(self.activations[cnt].clone()).component_mul(&passthrough);
                    println!("input shape {:?} len {}", input.shape(), input.len());
                    let d_weights = (&d_activation * input.transpose()).map(|x| x / d_activation.nrows() as f32);
                    println!("Weights shape {:?}", d_weights.shape());
                    

                    let d_bias = DMatrix::from_row_slice(d_weights.nrows(), 1, &(0..d_weights.nrows())
                        .map(|i| d_activation.row(i).sum() / d_activation.nrows() as f32)
                        .collect::<Vec<f32>>());

                    der_weights.push(d_weights);
                    der_bias.push(d_bias);



                }else{

                    // dc/dz = dc/da * da/dz
                    let d_activation = NeuralNetwork::leakyrelu_prime(self.activations[cnt].clone()).component_mul(&passthrough);

                    // Weights derivative
                    let d_weights = (&d_activation * &self.activations[cnt - 1].transpose()).map(|x| x / d_activation.nrows() as f32);
                
                    // Bias derivative
                    let d_bias = DMatrix::from_row_slice(d_weights.nrows(), 1, &(0..d_weights.nrows())
                        .map(|i| d_activation.row(i).sum() / d_activation.nrows() as f32)
                        .collect::<Vec<f32>>());
                
                    // Layer derivative
                    passthrough = &self.weights[cnt].transpose() * &d_activation;

                    der_weights.push(d_weights);
                    der_bias.push(d_bias);

                }

            }

            der_weights.reverse();
            der_bias.reverse();

            (der_weights, der_bias)
            
        }

//(centered, var, x_hat, std_inv1, std_inv)
        fn backprop_batched(&mut self, 
            input: &Vec<DMatrix<f32>>,
            truth: &Vec<DMatrix<f32>>, 
            mean: Vec<DMatrix<f32>>, 
            xhat: Vec<DMatrix<f32>>,
            sdev: Vec<DVector<f32>>)->(Vec<DVector<f32>>, Vec<DVector<f32>>){
                
                    
                let mut passthrough: DMatrix<f32> = DMatrix::zeros(self.neurons[self.layers - 1], truth[0].ncols());
                let mut grad_gamma: Vec<DVector<f32>> = Vec::new();
                let mut grad_beta: Vec<DVector<f32>> = Vec::new();
                
                for cnt in (0..self.layers).rev(){
                    if cnt == self.layers -1 {

                        // Output layer
                        //Activation functions assume BCE with one-hot encoding

                        let d_activation:DMatrix<f32> = match self.activation {
                            Some(Activation::Softmax) => {
                                //.component_mul(&NeuralNetwork::softmax_prime_return(self.activations[cnt].clone()));
                                DMatrix::from_columns(&self.activations[cnt].column_iter().enumerate().map(|(j, col)| col - &truth[j]).collect::<Vec<_>>())
                            }

                            Some(Activation::Sigmoid) => {
                                DMatrix::from_columns(&self.activations[cnt].column_iter().enumerate().map(|(j, col)| col - &truth[j]).collect::<Vec<_>>())
                            }
                            _=>{
                                DMatrix::from_columns(&self.activations[cnt].column_iter().enumerate().map(|(j, col)| col - &truth[j]).collect::<Vec<_>>())
                            }
                            
                        };
                        /*println!("d_activation L2 max {:?}, min {:?} mean {:?}", d_activation.max(), d_activation.min(), d_activation.mean());

                        println!("D_activations shape {:?}", d_activation.shape());

                        let mut rho = &mean[cnt].map(|x| 1.0/(x*x + 0.001).sqrt());

                        let tau = &rho.map(|x| 1.0/x);

                        let d_beta  = DMatrix::from_row_slice(self.neurons[cnt + 1], 1, &(0..self.neurons[cnt + 1])
                        .map(|i| d_activation.row(i).sum()/d_activation.ncols() as f32)
                        .collect::<Vec<f32>>());
                        println!("D_beta shape {:?}", d_beta.shape());


                        grad_beta.insert(0, d_beta.clone());
                        println!("Adding layer {}" , cnt);

                        let d_gamma:DMatrix<f32> = DMatrix::from_columns(&xhat[cnt].column_iter().map(|m| m.component_mul(&d_beta)).collect::<Vec<_>>());
                        grad_gamma.insert(0,d_gamma.clone());

                        let d_xhat = DMatrix::from_columns(&d_gamma.column_iter().map(|m| m.component_mul(&self.gamma[cnt])).collect::<Vec<_>>());

                        let d_ivar = (&d_xhat.component_mul(&mean[cnt])).column_sum();

                        let d_stdev1 = &d_xhat.component_mul(tau);


                        //let d_sqvar = d_ivar * (rho[cnt].map(|x| x = 1.0/x).iter().collect());
                        //let d_sqvar = d_ivar.component_mul(&rho.map(|x| 1.0 / x));
                        let rho = &rho.map(|x| 1.0 / x);

                        let d_sqvar = DMatrix::from_columns(&rho.column_iter().map(|m| m.component_mul(&d_ivar)).collect::<Vec<_>>());

                        let d_var = var[cnt].map(|x| 0.5 * 1.0 / (x - 0.000001).sqrt());

                        let d_var:DMatrix<f32> = DMatrix::from_columns(&d_sqvar.column_iter().enumerate().map(|(val, x)| x.map(|y| y * d_var[val])).collect::<Vec<_>>());

                        let d_sq = d_var.map(|x| x/input.len() as f32);


                        let d_stdev2 = 2.0*&mean[cnt].component_mul(&d_sq);

                        let d_x1 = (d_stdev1 + &d_stdev2);

                        let d_dev = &d_stdev1.zip_map(&d_stdev2, |x, y| -1.0/(x+y));

                        let d_x2 = d_dev.map(|x| x/input.len() as f32);

                        let d_x = d_x1 + d_x2;

                        let d_activation = d_x;*/

                        //println!("d_activation L1 after bN max {:?}, min {:?} mean {:?}", d_activation.max(), d_activation.min(), d_activation.mean());

                        
                        // Weights derivative
                        //let d_weights = (&d_activation * &self.activations[cnt - 1].transpose()).map(|x| x / d_activation.nrows() as f32);
                        let d_weights = &d_activation.column_iter().enumerate().map(|(val, x)| (x * &self.activations[cnt - 1].column(val).transpose()).map(|z|z / d_activation.nrows() as f32)).collect::<Vec<_>>();

                    
                        // Bias derivative
                        //let d_bias = DMatrix::from_row_slice(d_weights[0].nrows(), 1, &(0..d_weights[0].nrows())
                          //  .map(|i| d_activation.row(i).sum() / d_activation.nrows() as f32)
                        //    .collect::<Vec<f32>>());
                        let d_bias: DVector<f32> = d_activation.column_sum().map(|x| x / d_activation.ncols() as f32);

                        // Layer derivative
                        passthrough = &self.weights[cnt].transpose() * &d_activation;

                        //println!("d_weights shape {:?} lne {}", d_weights[0].shape(), d_weights.len());
                        for i in d_weights{
                            self.accum_grad_weights[cnt] += i;
                        }
                        self.accum_grad_bias[cnt] += d_bias.clone();
    
    
    
                    }else if cnt == 0{
                        
                         // dc/dz = dc/da * da/dz
                        //println!("Passthrough max {:?}, min {:?} mean {:?}", passthrough.max(), passthrough.min(), passthrough.mean());
                        // dc/dz = dc/da * da/dz
                        let d_activation = NeuralNetwork::leakyrelu_prime(self.activations[cnt].clone()).component_mul(&passthrough);
                        
                    

                        //let d_beta  = DMatrix::from_row_slice(self.neurons[cnt + 1], 1, &(0..self.neurons[cnt + 1])
                        //.map(|i| d_activation.row(i).sum() / d_activation.nrows() as f32)
                        //.collect::<Vec<f32>>());
                        let d_beta: DVector<f32> = d_activation.column_sum().map(|x| x / d_activation.ncols() as f32);


                        grad_beta.insert(0,d_beta.clone());
                        //println!("Adding layer {}" , cnt);


                        println!("XHAT {:?}, len {}", xhat[cnt].shape(), xhat.len());
                        println!("activation {:?}", d_activation.shape());
                        let d_gamma:DVector<f32> = DMatrix::from_columns(&xhat[cnt].column_iter().enumerate().map(|(cnt, m)| m.component_mul(&d_activation.column(cnt))).collect::<Vec<_>>()).column_sum();
                        //let d_gamma  = &xhat[cnt]*&d_activation;
                        grad_gamma.insert(0,d_gamma.clone());

                        let d_xhat = DMatrix::from_columns(&d_activation.column_iter().map(|m| m.component_mul(&self.gamma[cnt])).collect::<Vec<_>>());

                        let d_ivar = (&d_xhat.component_mul(&mean[cnt])).column_sum();


                        let d_stdev1 = DMatrix::from_columns(&d_xhat.column_iter().map(|x| x + &sdev[cnt].map(|z| 1.0/((z + 1e-8).sqrt()))).collect::<Vec<_>>());


                        //let d_sqvar = d_ivar * (rho[cnt].map(|x| x = 1.0/x).iter().collect());
                        //let d_sqvar = d_ivar.component_mul(&rho.map(|x| 1.0 / x));
                        let d_sqvar = DMatrix::from_columns(&sdev[cnt].map(|z| z.sqrt()).column_iter().map(|m| m.map(|x| -1.0/x*x)).collect::<Vec<_>>());
                        //println!("Dsqvar shape {:?}", d_sqvar.shape());
                        //println!("ivar shape {:?}", d_ivar.shape());

                        let d_sqvar: DMatrix<f32> = DMatrix::from_columns(&d_sqvar.column_iter().map(|x| x.component_mul(&d_ivar)).collect::<Vec<_>>());
                        //println!("Dsqvar shape {:?}", d_sqvar.shape());
                        //println!("var shape {:?}", var[cnt].shape());
                        let d_var = sdev[cnt].map(|x| 0.5 * 1.0 / (x + 1e-8).sqrt());
                        //println!("Dvar shape {:?}", d_var.shape());

                        let d_var:DMatrix<f32> = DMatrix::from_columns(&d_sqvar.column_iter().enumerate().map(|(val, x)|x * d_var[cnt]).collect::<Vec<_>>());
                        //println!("Dsqvar shape {:?}", d_var.shape());
                        let d_sq = d_var.map(|x| x/input.len() as f32);
                        //println!("dsq shape {:?}", d_sq.shape());
                        //println!("Mean shape {:?}", mean[cnt].shape());
                        let d_stdev2 = 2.0*DMatrix::from_columns(&mean[cnt].column_iter().map(|x| x.component_mul(&d_sq)).collect::<Vec<_>>());
                        //println!("Dstdev2 shape {:?}", d_stdev2.shape());
                        let d_x1 = &d_stdev1 + &d_stdev2;
                        ///println!("Dx1 shape {:?}", d_x1.shape());
                        let d_dev = (d_stdev1 + d_stdev2).column_sum() * -1.0;
                        //println!("Ddev shape {:?}", d_dev.shape());

                        //let d_x2:DMatrix<f32> = DMatrix::from_columns(&DMatrix::<f32>::zeros(self.activations[cnt].nrows(), self.activations[cnt].ncols()).column_iter().map(|x| (x.component_mul(&d_dev)).map(|t| t/input.len() as f32)).collect::<Vec<_>>());
                        let d_x2:DMatrix<f32> = DMatrix::from_columns(&d_dev.column_iter().map(|x|(x.component_mul(&DVector::from_element(x.nrows(), 1.0))).map(|t|t/input.len() as f32)).collect::<Vec<_>>());

                        //println!("Dx2 shape {:?}", d_x2.shape());
                        let d_x:DMatrix<f32> = DMatrix::from_columns(&d_x1.column_iter().map(|x| x + &d_x2).collect::<Vec<_>>());

                        let d_activation = d_x;

                        //println!("d_activation shape {:?}", d_activation.shape());
                        println!("INput shape {:?} len {}", input[0].shape(), input.len());
                        let d_weights_cols = &d_activation.column_iter().enumerate().map(|(val, x)| (x * &input[val]).map(|z|z / d_activation.nrows() as f32)).collect::<Vec<_>>();
                        //let d_weights = &d_activation.column_iter().enumerate().map(|(val, x)| (x * &self.activations[cnt - 1].column(val).transpose()).map(|z|z / d_activation.nrows() as f32)).collect::<Vec<_>>();
                        //println!("dweights shape {:?} len {}", d_weights_cols[0].shape(), d_weights_cols.len());
                        //let d_weights = DMatrix::from_columns(d_weights_cols);
                        //let d_weights = (&d_activation * i.transpose()).map(|x| x / d_activation.nrows() as f32);



                        //let d_bias = DMatrix::from_row_slice(d_weights_cols[0].nrows(), 1, &(0..d_weights_cols[0].nrows())
                          //  .map(|i| d_activation.row(i).sum() / d_activation.nrows() as f32)
                            //.collect::<Vec<f32>>());
                        let d_bias: DVector<f32> = d_activation.column_sum().map(|x| x / d_activation.ncols() as f32);

                        for i in d_weights_cols{
                            self.accum_grad_weights[cnt] += i;
                        }
                        //self.accum_grad_weights[cnt] += d_weights.clone();
                        self.accum_grad_bias[cnt] += d_bias.clone();
    
    
    
                    }else{
                        //println!("Passthrough L1 max {:?}, min {:?} mean {:?}", passthrough.max(), passthrough.min(), passthrough.mean());
    
                        // dc/dz = dc/da * da/dz
                        let d_activation = NeuralNetwork::leakyrelu_prime(self.activations[cnt].clone()).component_mul(&passthrough);
                        
                    

                        //let d_beta  = DMatrix::from_row_slice(self.neurons[cnt + 1], 1, &(0..self.neurons[cnt + 1])
                        //.map(|i| d_activation.row(i).sum() / d_activation.nrows() as f32)
                        //.collect::<Vec<f32>>());
                        let d_beta: DVector<f32> = d_activation.column_sum().map(|x| x / d_activation.ncols() as f32);


                        grad_beta.insert(0,d_beta.clone());
                        //println!("Adding layer {}" , cnt);


                        println!("XHAT {:?}, len {}", xhat[cnt].shape(), xhat.len());
                        println!("activation {:?}", d_activation.shape());
                        let d_gamma:DVector<f32> = DMatrix::from_columns(&xhat[cnt].column_iter().enumerate().map(|(cnt, m)| m.component_mul(&d_activation.column(cnt))).collect::<Vec<_>>()).column_sum();
                        //let d_gamma  = &xhat[cnt]*&d_activation;
                        grad_gamma.insert(0,d_gamma.clone());

                        let d_xhat = DMatrix::from_columns(&d_activation.column_iter().map(|m| m.component_mul(&self.gamma[cnt])).collect::<Vec<_>>());

                        let d_ivar = (&d_xhat.component_mul(&mean[cnt])).column_sum();


                        let d_stdev1 = DMatrix::from_columns(&d_xhat.column_iter().map(|x| x + &sdev[cnt].map(|z| 1.0/((z + 1e-8).sqrt()))).collect::<Vec<_>>());


                        //let d_sqvar = d_ivar * (rho[cnt].map(|x| x = 1.0/x).iter().collect());
                        //let d_sqvar = d_ivar.component_mul(&rho.map(|x| 1.0 / x));
                        let d_sqvar = DMatrix::from_columns(&sdev[cnt].map(|z| z.sqrt()).column_iter().map(|m| m.map(|x| -1.0/x*x)).collect::<Vec<_>>());
                        //println!("Dsqvar shape {:?}", d_sqvar.shape());
                        //println!("ivar shape {:?}", d_ivar.shape());

                        let d_sqvar: DMatrix<f32> = DMatrix::from_columns(&d_sqvar.column_iter().map(|x| x.component_mul(&d_ivar)).collect::<Vec<_>>());
                        //println!("Dsqvar shape {:?}", d_sqvar.shape());
                        //println!("var shape {:?}", var[cnt].shape());
                        let d_var = sdev[cnt].map(|x| 0.5 * 1.0 / (x - 1e+8).sqrt());
                        //println!("Dvar shape {:?}", d_var.shape());

                        let d_var:DMatrix<f32> = DMatrix::from_columns(&d_sqvar.column_iter().enumerate().map(|(val, x)|x * d_var[cnt]).collect::<Vec<_>>());
                        //println!("Dsqvar shape {:?}", d_var.shape());
                        let d_sq = d_var.map(|x| x/input.len() as f32);
                        //println!("dsq shape {:?}", d_sq.shape());
                        //println!("Mean shape {:?}", mean[cnt].shape());
                        let d_stdev2 = 2.0*DMatrix::from_columns(&mean[cnt].column_iter().map(|x| x.component_mul(&d_sq)).collect::<Vec<_>>());
                        //println!("Dstdev2 shape {:?}", d_stdev2.shape());
                        let d_x1 = &d_stdev1 + &d_stdev2;
                        ///println!("Dx1 shape {:?}", d_x1.shape());
                        let d_dev = (d_stdev1 + d_stdev2).column_sum() * -1.0;
                        //println!("Ddev shape {:?}", d_dev.shape());
                        //let d_x2:DMatrix<f32> = DMatrix::from_columns(&DMatrix::<f32>::zeros(self.activations[cnt].nrows(), self.activations[cnt].ncols()).column_iter().map(|x| (x.component_mul(&d_dev)).map(|t| t/input.len() as f32)).collect::<Vec<_>>());
                        let d_x2:DMatrix<f32> = DMatrix::from_columns(&d_dev.column_iter().map(|x|(x.component_mul(&DVector::from_element(x.nrows(), 1.0))).map(|t|t/input.len() as f32)).collect::<Vec<_>>());
                       
                        //println!("Dx2 shape {:?}", d_x2.shape());
                        let d_x:DMatrix<f32> = DMatrix::from_columns(&d_x1.column_iter().map(|x| x + &d_x2).collect::<Vec<_>>());

                        let d_activation = d_x;

                        passthrough = &self.weights[cnt].transpose() * &d_activation;

                        //println!("Self.activation shape {:?}",self.activations[cnt -1].shape());
                        // Weights derivative
                        //let d_weights = (&d_activation * &self.activations[cnt - 1].transpose()).map(|x| x / d_activation.nrows() as f32);
                        let d_weights = &d_activation.column_iter().enumerate().map(|(val, x)| (x * &self.activations[cnt - 1].column(val).transpose()).map(|z|z / d_activation.nrows() as f32)).collect::<Vec<_>>();

                    
                        // Bias derivative
                        //let d_bias = DMatrix::from_row_slice(d_weights[0].nrows(), 1, &(0..d_weights[0].nrows())
                          //  .map(|i| d_activation.row(i).sum() / d_activation.nrows() as f32)
                            //.collect::<Vec<f32>>());
                        let d_bias: DVector<f32> = d_activation.column_sum().map(|x| x / d_activation.ncols() as f32);

                        //println!("Weights L1 max {:?}, min {:?} mean {:?}", self.weights[cnt].max(), self.weights[cnt].min(), self.weights[cnt].mean());
                        //println!("d_activation L1 max {:?}, min {:?} mean {:?}", d_activation.max(), d_activation.min(), d_activation.mean());
                        // Layer derivative
                        //passthrough = &self.weights[cnt].transpose() * &d_activation;

    
                        for i in d_weights{
                            self.accum_grad_weights[cnt] += i;
                        }
                        self.accum_grad_bias[cnt] += d_bias.clone();
    
                    }
    
                }
    
            //self.print_grad_stats();

            for layer_index in 0..self.layers{
                self.accum_grad_weights[layer_index] /= input.len() as f32;
                self.accum_grad_bias[layer_index] /= input.len() as f32;
            }
            /*for (cnt, i) in grad_gamma.iter().enumerate(){
                println!("grad shape {:?}, num {}", i.shape(), cnt);
            }*/
        

            (grad_gamma, grad_beta)
                    
        }
    





        fn adam(&mut self, beta1:f32, beta2:f32, epsilon:f32){
            for i in 0..self.layers{
                self.momentum_weights[i] = beta1 * &self.momentum_weights[i] + (1.0 - beta1) * &self.accum_grad_weights[i];
                self.momentum_bias[i] = beta1 * &self.momentum_bias[i] + (1.0 - beta1) * &self.accum_grad_bias[i];

                let momentum_weights_hat = &self.momentum_weights[i] / (1.0 - beta1.powf(self.itteration as f32));
                let momentum_bias_hat = &self.momentum_bias[i] / (1.0 - beta1.powf(self.itteration as f32));

                self.velocity_weights[i] = beta2 * &self.velocity_weights[i] +  &self.accum_grad_weights[i].map(|x| (1.0 - beta2)*x.powf(2.0));
                //self.velocity_weights[i] = beta2 * &self.velocity_weights[i] + ((1.0 - beta2) * &self.accum_grad_weights[i].pow(2));
                self.velocity_bias[i] = beta2 * &self.velocity_bias[i] + &self.accum_grad_bias[i].map(|x| (1.0 - beta2) * x.powf(2.0));

                let velocity_weights_hat = &self.velocity_weights[i] / (1.0 - beta2.powf(self.itteration as f32));
                let velocity_bias_hat = &self.velocity_bias[i] / (1.0 - beta2.powf(self.itteration as f32));

                self.weights[i] -= self.alpha * &momentum_weights_hat.zip_map(&velocity_weights_hat, |m, v| m / (v.sqrt() + epsilon));
                self.bias[i] -= self.alpha * &momentum_bias_hat.zip_map(&velocity_bias_hat, |m, v| m / (v.sqrt() + epsilon));

            }
            self.itteration += 1;
            
        }

        
        fn rmsprop(&mut self, beta:f32){

            for i in 0..self.layers {
                self.velocity_weights[i] = beta * &self.velocity_weights[i] + (1.0 - beta) * &self.accum_grad_weights[i] * (&self.accum_grad_weights[i]);
                self.velocity_bias[i] = beta * &self.velocity_bias[i] + (1.0 - beta) * &self.accum_grad_bias[i]*(&self.accum_grad_bias[i]);

                self.weights[i] -= self.alpha * &self.accum_grad_weights[i].component_div(&(&self.velocity_weights[i]).map(|x| (x+ 1e-8).sqrt()));
                self.bias[i] -= self.alpha * &self.accum_grad_bias[i].component_div(&(&self.velocity_bias[i]).map(|x| (x + 1e-8).sqrt()));
                
            }
        }

        fn sgd(&mut self){
            for i in 0..self.layers {
                self.weights[i] -=  self.alpha * &self.accum_grad_weights[i];
                self.bias[i] -=  self.alpha * &self.accum_grad_bias[i];

                //println!("Grad Weights: {:?}", &self.accum_grad_weights[i]);
                if i == self.layers - 1{
                    println!("Grad Weights: {:?}", &self.accum_grad_weights[i].mean());
                }

            }
            //println!("Bias: {:?}", self.bias);
            //println!("Weights: {:?}", self.weights);
            //println!("\n\n ---------------------------------------- \n\n");
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

 /*pub async fn train_concurrent(
        &mut self,
        input: Vec<DMatrix<f32>>,
        truth: Vec<DMatrix<f32>>,
        epochs: usize,
        batch_size: usize,
        max_threads: usize,
    ) {
        println!("Training on {} Images", input.len());

        let mut activations: Vec<DMatrix<f32>> = Vec::with_capacity(self.layers);
        // Fill the vector with default values
        activations.extend(vec![DMatrix::zeros(0, 0); self.layers]);

        for layer_index in 0..self.layers {
            self.accum_grad_weights[layer_index].fill(0.0);
            self.accum_grad_bias[layer_index].fill(0.0);
        }

        for cnt in 0..epochs {

            let mut accum_grad_weights_batch = self.accum_grad_weights.clone();
            let accum_grad_bias_batch = Arc::new(Mutex::new(self.accum_grad_bias.clone()));
            let weights = self.weights.clone();
            let bias = self.bias.clone();
            let layers = self.layers;
            let neurons = self.neurons.clone();

            // Create a shared structure with owned data
            let shared_data_backwards = Arc::new(Mutex::new((
                accum_grad_weights_batch.clone(),
                accum_grad_bias_batch.clone(),
                weights,
                bias,
            )));


            let mut handles = Vec::new();
            let num_batches = (input.len() + batch_size - 1) / batch_size;

            for batch_index in 0..num_batches {
                let start = batch_index * batch_size;
                let end = usize::min(start + batch_size, input.len());
                let batch_input = input[start..end].to_vec();
                let batch_truth = truth[start..end].to_vec();

                let shared_data_backwards_clone = Arc::clone(&shared_data_backwards);
                let neurons_clone = neurons.clone();

                // Spawn a task for the current batch
                let handle = tokio::spawn(async move {
                    let lock = shared_data_backwards_clone.lock().await;

                    // Extract the locked data
                    let (mut accum_grad_weights_batch, mut accum_grad_bias_batchs, weights, bias) = lock.clone();

                    let mut batch_activations: Vec<DMatrix<f32>> = Vec::with_capacity(weights.len());

                    for (i, input_matrix) in batch_input.iter().enumerate() {
                        // Perform forward pass
                        NeuralNetwork::feedfrwd_concurrent(
                            input_matrix.clone(),
                            &mut batch_activations,
                            &weights,
                            &bias,
                        );

                        // Perform backpropagation
                        let (d_weights, d_bias) = NeuralNetwork::backprop_concurrent(
                            input_matrix.clone(),
                            batch_truth[i].clone(),
                            &batch_activations,
                            &weights,
                            layers,
                            &neurons_clone,
                            
                        );

                        // Accumulate gradients for this batch
                        for layer_index in 0..weights.len() {
                            accum_grad_weights_batch[layer_index] += &d_weights[layer_index];
                            accum_grad_bias_batch[layer_index] += &d_bias[layer_index];
                        }
                    }
                });

                handles.push(handle);
            }

            // Wait for all tasks to complete
            for handle in handles {
                handle.await.unwrap();
            }

            self.accum_grad_weights = accum_grad_weights_batch.clone();
            self.accum_grad_bias = Arc::try_unwrap(accum_grad_bias_batch)
                .expect("Failed to unwrap Arc")
                .into_inner()
                .expect("Failed to lock Mutex");

            // Gradient normalization after all tasks complete
            for layer_index in 0..self.layers {
                self.accum_grad_weights[layer_index] /= input.len() as f32;
                self.accum_grad_bias[layer_index] /= input.len() as f32;
            }

            // Update the shared data with the accumulated gradients
            
            

            // Perform optimization step based on selected optimizer
            match self.optimizer {
                Some(Optimizer::SGD) => {
                    self.sgd();
                }
                Some(Optimizer::Adam) => {
                    self.adam(0.9, 0.999, 0.000008);
                }
                Some(Optimizer::RMSprop) => {
                    self.rmsprop(0.9);
                }
                _ => {
                    for i in 0..self.layers {
                        self.weights[i] -= self.alpha * &self.accum_grad_weights[i];
                        self.bias[i] -= self.alpha * &self.accum_grad_bias[i];
                    }
                }
            }

            // Compute error for this epoch
            let mut error = 0.0;
            for j in 0..input.len() {
                error += self.cost(&activations[self.layers - 1], &truth[j]);
            }
            error /= input.len() as f32;

            if error < 1e-3 {
                println!("Epoch: {} - Early stopping, error below threshold.", cnt);
                break;
            }

            if cnt % 100 == 0 {
                println!("Epoch: {}, Error: {}", cnt, error);
            }
        }
    }*/

    /*
        fn backprop_concurrent(&mut self, input: DMatrix<f32>, truth: DMatrix<f32>, weights: &Vec<DMatrix<f32>>, layers: usize, neurons: &Vec<usize>) -> (Vec<DMatrix<f32>>, Vec<DMatrix<f32>>) {
            let mut der_weights = Vec::new();
            let mut der_bias = Vec::new();
            let mut passthrough: DMatrix<f32> = DMatrix::zeros(neurons[layers - 1], truth.ncols());

            for cnt in (0..layers).rev(){
                if cnt == layers -1 {
                    // Output layer
                    //NeuralNetwork::softmax_prime(&mut activations[cnt].clone());
                    let mut d_activation = &self.activations[cnt] - &truth;
                    assert_eq!(d_activation.nrows(), neurons[cnt + 1]);
                    assert_eq!(d_activation.ncols(), truth.ncols());
        
                    // Weights derivative
                    let d_weights = if cnt > 0 {
                        &d_activation * &self.activations[cnt - 1].transpose()
                    } else {
                        &d_activation * &input.transpose()
                    };

                    assert_eq!(d_weights.nrows(), neurons[cnt + 1]);
                    assert_eq!(d_weights.ncols(), neurons[cnt]);
        
                    // Bias derivative
                    let d_bias = DMatrix::from_row_slice(d_weights.nrows(), 1, &(0..d_weights.nrows())
                        .map(|i| d_activation.row(i).sum())
                        .collect::<Vec<f32>>());
                    assert_eq!(d_bias.nrows(), neurons[cnt + 1]);
                    assert_eq!(d_bias.ncols(), 1);
        
                    // Layer derivative
                    passthrough  = &weights[cnt].transpose() * &d_activation;
                    assert_eq!(passthrough.nrows(), neurons[cnt]);
                    assert_eq!(passthrough.ncols(), truth.ncols());
        
                    der_weights.push(d_weights);
                    der_bias.push(d_bias);


                    continue;


                }
                else if cnt == 0{
                    // Input layer
                    assert_eq!(input.nrows(), neurons[0]);
                    //sigmoid prime
                    //let d_activations = &activations[cnt].component_mul(&(&activations[cnt].map(|x| x * (1.0 - x))));
                    //relu prime
                    let mut d_activations = self.activations[cnt].clone();
                    NeuralNetwork::relu_prime(&mut d_activations);
                    
                    assert_eq!(d_activations.nrows(), neurons[cnt + 1]);
                    
                    // dc/dz = dc/da * da/dz
                    let d_activation = &d_activations*(&passthrough);
                    assert_eq!(d_activation.nrows(), neurons[cnt + 1]);
                    assert_eq!(d_activation.ncols(), passthrough.ncols());

                    let d_weights = d_activation * &input.transpose();
                    assert_eq!(d_weights.nrows(), neurons[cnt + 1]);
                    assert_eq!(d_weights.ncols(), neurons[cnt]);

                    let d_bias = DMatrix::from_row_slice(d_weights.nrows(), 1, &(0..d_weights.nrows())
                        .map(|i| d_activation.row(i).sum())
                        .collect::<Vec<f32>>());

                    assert_eq!(d_bias.nrows(), neurons[cnt + 1]);
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
                let mut d_activations = self.activations[cnt].clone();
                NeuralNetwork::relu_prime(&mut d_activations);
                
                    
                assert_eq!(d_activations.nrows(), neurons[cnt + 1]);
                // dc/dz = dc/da * da/dz
                let d_activation = d_activations*(&passthrough);
                assert_eq!(d_activation.nrows(), neurons[cnt + 1]);
                assert_eq!(d_activation.ncols(), passthrough.ncols());

                // Weights derivative
                let d_weights = &d_activation * &self.activations[cnt - 1].transpose();
                assert_eq!(d_weights.nrows(), neurons[cnt + 1]);
                assert_eq!(d_weights.ncols(), neurons[cnt]);

                // Bias derivative
                let d_bias = DMatrix::from_row_slice(d_weights.nrows(), 1, &(0..d_weights.nrows())
                    .map(|i| d_activation.row(i).sum())
                    .collect::<Vec<f32>>());
                assert_eq!(d_bias.nrows(), neurons[cnt + 1]);
                assert_eq!(d_bias.ncols(), 1);

                // Layer derivative
                passthrough = &weights[cnt].transpose() * d_activation;
                assert_eq!(passthrough.nrows(), neurons[cnt]);

                der_weights.push(d_weights);
                der_bias.push(d_bias);

            }

            der_weights.reverse();
            der_bias.reverse();

            (der_weights, der_bias)

            
        }


            fn backprop(&mut self, 
        input: DMatrix<f32>,
        truth: DMatrix<f32>)-> (Vec<DMatrix<f32>>, Vec<DMatrix<f32>>) {
            let mut der_weights = Vec::new();
            let mut der_bias = Vec::new();
            let mut passthrough: DMatrix<f32> = DMatrix::zeros(self.neurons[self.layers - 1], truth.ncols());

            for cnt in (0..self.layers).rev(){
                if cnt == self.layers -1 {
                    // Output layer
                    //NeuralNetwork::softmax_prime(&mut activations[cnt].clone());
                    //forward pass uses softmax so we need to use the derivative of softmax
                    //d_softmax = y_hat * (1 - y_hat)
                    let mut d_activation = self.activations[cnt].clone();
                    NeuralNetwork::softmax_prime(&mut d_activation);
                    //d_activation = d_activation.zip_map(&truth, |x, y| x - y);
                    d_activation = (self.activations[cnt].clone() - &truth).component_mul(&d_activation);
                    
                    assert_eq!(d_activation.nrows(), self.neurons[cnt + 1]);
                    assert_eq!(d_activation.ncols(), truth.ncols());
        
                    // Weights derivative
                    let d_weights = if cnt > 0 {
                        &d_activation * &self.activations[cnt - 1].transpose()
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
                    //passthrough  = self.weights[cnt].transpose().zip_map(&d_activation, |x, y| x* y);
                    passthrough = &self.weights[cnt].transpose() * &d_activation;
                    
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
                    let mut d_activations = self.activations[cnt].clone();
                    for x in d_activations.iter_mut() {
                        *x = if *x > 0.0 { 1.0 } else { 0.0 };
                    }

                    //let d_activations = self.activations[cnt].clone();
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
                let mut d_activations = self.activations[cnt].clone();
                for x in d_activations.iter_mut() {
                    *x = if *x > 0.0 { 1.0 } else { 0.0 };
                }
                //relu prime
                //NeuralNetwork::relu_prime(&mut self.activations[cnt].clone());
                //let d_activations = self.activations[cnt].clone();
                    
                assert_eq!(d_activations.nrows(), self.neurons[cnt + 1]);
                // dc/dz = dc/da * da/dz
                let d_activation = d_activations.component_mul(&passthrough);
                assert_eq!(d_activation.nrows(), self.neurons[cnt + 1]);
                assert_eq!(d_activation.ncols(), passthrough.ncols());

                // Weights derivative
                let d_weights = &d_activation * &self.activations[cnt - 1].transpose();
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
        */
    
