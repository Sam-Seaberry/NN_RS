use nalgebra::{DMatrix, U28, Dynamic};
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

        let mut accumulated_weight_gradients: Vec<DMatrix<f32>> = Vec::new();
        let mut accumulated_bias_gradients: Vec<DMatrix<f32>> = Vec::new();

        let mut momentum_weights: Vec<DMatrix<f32>> = Vec::new();
        let mut momentum_bias: Vec<DMatrix<f32>> = Vec::new();

        let mut velocity_weights: Vec<DMatrix<f32>> = Vec::new();
        let mut velocity_bias: Vec<DMatrix<f32>> = Vec::new();

        let mut activations: Vec<DMatrix<f32>> = Vec::with_capacity(layers);

        // Fill the vector with default values
        activations.extend(vec![DMatrix::zeros(0, 0); layers]);

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
            activations,
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
                
                for j in current..next {
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
                current += batch;

                
    
                //gradient normalization
                //TODO: Change to work for batch size instead of input size
                
                for layer_index in 0..self.layers{
                    self.accum_grad_weights[layer_index] /= batch as f32;
                    self.accum_grad_bias[layer_index] /= batch as f32;
                }
    
                match self.optimizer {
                    Some(Optimizer::SGD) => {
                        self.sgd();
    
                    }
                    Some(Optimizer::Adam) => {
                        self.adam( 0.9, 0.999, 0.00000008);
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
                    error += self.cost(&self.activations[self.layers-1], &truth[j]);
                }
                error = error/input.len() as f32;
                
                if error < 1e-3 {
                    println!("Epcoh: {}", cnt);
                    break;
                }
                println!("Epoch: {}, Error: {:?}, itteration {}", cnt, error, &self.itteration);
    
                self.test_accuracy(&test_input, &test_truth);
                
                if cnt % 1 == 0 {
                    self.print_weight_stats();
                    self.print_activation_stats();
                    self.print_grad_stats();
                    //self.count_zeros();
                    self.count_zeros_weights();
                    
                }
                
                /*if cnt == epochs/2{
                    self.alpha = self.alpha * 0.5;
                }*/
                println!("Final Layer Activation: {:?}", self.activations[self.layers - 1]);
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
            //println!("---------------------------------------------------");
            last_guess[max_index.0] += 1;
            if max_index.0 as i32 == max_index_truth.0 as i32{
                truthcnt += 1;
                correct_guess[max_index.0] += 1;
                //println!("Correct");
            }else{
                //println!("Incorrect");
            }
            

            
            //println!("Max index output: {:?}, max index truth {:?}", max_index.0, max_index_truth.0);
            //println!("Output: {}, Value: {:.3}, Index: {:?}", i, max_value, max_index);
            //println!("Truth: {:?}", test_truth[i]);
            //println!("--------------------------------------------------- \n\n");
        }
        println!("Accuracy: {}, Guess Count: {:?}, Correct Count: {:?}", truthcnt as f32 / results.len() as f32, last_guess, correct_guess);
    }



    pub fn xavier_init(rows: usize, cols: usize, activation: Option<&Activation>) -> DMatrix<f32> {
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
    
    pub fn cost(&self, y_hat: &DMatrix<f32>, y: &DMatrix<f32>) -> f32 {
        assert_eq!(y_hat.nrows(), y.nrows());
    
        let epsilon = 1e-10; // Avoid log(0)
        let y_hat = y_hat.map(|x| x.max(epsilon).min(1.0 - epsilon)); // Clipping values
        //let losses = y.zip_map(&y_hat, |yi, y_hat_i| {
       //     -(yi * y_hat_i.ln() + (1.0 - yi) * (1.0 - y_hat_i).ln())
        //});
        let losses = y.zip_map(&y_hat, |yi, y_hat_i| -(yi * y_hat_i.ln()));
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

    fn softmax_prime(arr: &mut DMatrix<f32>) {
        arr.iter_mut().for_each(|x| *x = *x * (1.0 - *x));
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
    
    pub fn feedfrwd(&mut self, input: &DMatrix<f32>) {
        let mut input = input.clone();
        assert_eq!(self.weights.len(), self.layers);
        assert_eq!(self.bias.len(), self.layers);

        // Store input activation (Important for backpropagation)
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

            if i == self.layers - 1 {
                match self.activation {
                    Some(Activation::Softmax) => {
                        NeuralNetwork::softmax(&mut z);
                        
                    }
                    _ => {
                        NeuralNetwork::softmax(&mut z);
                    }
                }
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
    
    
    // Custom function to repeat bias across columns
    pub fn repeat_bias(bias: &DMatrix<f32>, num_cols: usize) -> DMatrix<f32> {
        DMatrix::from_fn(bias.nrows(), num_cols, |i, _| bias[(i, 0)])
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

        fn backprop_test(&mut self, 
            input: &DMatrix<f32>,
            truth: &DMatrix<f32>)-> (Vec<DMatrix<f32>>, Vec<DMatrix<f32>>) {
                let mut der_weights = Vec::new();
                let mut der_bias = Vec::new();
                let mut passthrough: DMatrix<f32> = DMatrix::zeros(self.neurons[self.layers - 1], truth.ncols());
                
    
                for cnt in (0..self.layers).rev(){
                    if cnt == self.layers -1 {
                        // Output layer
                        let d_activation = self.activations[cnt].clone() - truth; //.component_mul(&NeuralNetwork::softmax_prime_return(self.activations[cnt].clone()));
                        
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

    
    
                    }else if cnt == 0{
                        
                        let d_activation = NeuralNetwork::leakyrelu_prime(self.activations[cnt].clone()).component_mul(&passthrough);
    
                        let d_weights = (&d_activation * input.transpose()).map(|x| x / d_activation.nrows() as f32);

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


        fn adam_parallel(&mut self, beta1: f32, beta2: f32, epsilon: f32) {
            self.itteration += 1;
            let iter_float = self.itteration as f32;

            self.weights
                .par_iter_mut()
                .zip(self.bias.par_iter_mut())
                .zip(self.momentum_weights.par_iter_mut())
                .zip(self.momentum_bias.par_iter_mut())
                .zip(self.velocity_weights.par_iter_mut())
                .zip(self.velocity_bias.par_iter_mut())
                .zip(self.accum_grad_weights.par_iter())
                .zip(self.accum_grad_bias.par_iter())
                .for_each(|(((((((weights, bias), momentum_weights,),momentum_bias,), velocity_weights), velocity_bias), accum_grad_weights), accum_grad_bias)| {
                        // Update momentum
                        *momentum_weights = beta1 * &*momentum_weights + (1.0 - beta1) * accum_grad_weights;
                        *momentum_bias = beta1 * &*momentum_bias + (1.0 - beta1) * accum_grad_bias;

                        // Bias-corrected momentum estimates
                        let momentum_weights_hat = &*momentum_weights / (1.0 - beta1.powf(iter_float));
                        let momentum_bias_hat = &*momentum_bias / (1.0 - beta1.powf(iter_float));

                        // Update velocity
                        *velocity_weights = beta2 * &*velocity_weights + (1.0 - beta2) * accum_grad_weights.map(|x| x.powi(2));
                        *velocity_bias = beta2 * &*velocity_bias + (1.0 - beta2) * accum_grad_bias.map(|x| x.powi(2));

                        // Bias-corrected velocity estimates
                        let velocity_weights_hat = &*velocity_weights / (1.0 - beta2.powf(iter_float));
                        let velocity_bias_hat = &*velocity_bias / (1.0 - beta2.powf(iter_float));

                        // Update parameters
                        *weights -= self.alpha
                            * momentum_weights_hat.zip_map(&velocity_weights_hat, |m, v| m / (v.sqrt() + epsilon));
                        *bias -= self.alpha
                            * momentum_bias_hat.zip_map(&velocity_bias_hat, |m, v| m / (v.sqrt() + epsilon));
                    },
                );
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
        */
    
