use candid::CandidType;
use ic_cdk::{query, update};
use ic_cdk_macros::*;
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::{rand, RandomExt};
use ndarray::prelude::*;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use getrandom::register_custom_getrandom;
use std::sync::Mutex;
use ndarray_rand::rand_distr::Uniform;

fn custom_getrandom(_buf: &mut [u8]) -> Result<(), getrandom::Error> { Ok(()) }
register_custom_getrandom!(custom_getrandom);
#[derive(CandidType, Deserialize)]
struct BatchData {
    preprocessed: Vec<Vec<f64>>,
    labels: Vec<f64>,
}

#[derive(CandidType, Deserialize, Clone)]
struct ModelUpdate {
    dense2_weights: Vec<Vec<f64>>,
    dense2_bias: Vec<f64>,
}

struct ServerModel {
    dense1: Dense,
    dense2: Dense,
}

impl ServerModel {
    fn new() -> Self {

        ServerModel {
            dense1: Dense::new(64 * 5 * 5, 128),
            dense2: Dense::new(128, 10),
        }
    }
    fn forward(&self, input: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let dense1_output = self.dense1.forward(input);
        let dense1_activated = leaky_relu_1d(&dense1_output, 0.01);
        let dense2_output = self.dense2.forward(&dense1_activated);
        (dense1_output, dense1_activated, dense2_output)
    }

    fn backward(&mut self, dense1_output: &[f64], dense1_activated: &[f64], dense2_output: &[f64],input:&[f64], target: &[f64], learning_rate: f64) -> (Vec<f64>, f32) {
        // Calcul du softmax et de la perte
        let softmax_output = softmax(dense2_output);
        let grad = softmax_cross_entropy_grad(&softmax_output, target);
        let loss = cross_entropy_loss2(&softmax_output, target);
        ic_cdk::println!("dense2back");
        // Rétropropagation pour la deuxième couche dense
        let (grad_dense1, grad_weights2, grad_bias2) = self.dense2.backward(&dense1_activated, &grad);
        self.dense2.update(&grad_weights2, &grad_bias2, learning_rate);
        ic_cdk::println!("lreluback");
        // Rétropropagation pour la fonction d'activation Leaky ReLU
        let grad_dense1_activated = leaky_relu_backward_1d(&dense1_output, &grad_dense1, 0.01);

        // Rétropropagation pour la première couche dense
        let (grad_intermediate, grad_weights1, grad_bias1) = self.dense1.backward(&input, &grad_dense1_activated);
        self.dense1.update(&grad_weights1, &grad_bias1, learning_rate);

        // Retourne les gradients intermédiaires et la perte
        (grad_intermediate.to_vec(), loss)
    }
    /*fn train(&mut self, batch_data: &BatchData, learning_rate: f64) -> f64 {
        let batch_size = batch_data.preprocessed.len();
        let mut total_loss = 0.0;

        for (preprocessed, label) in batch_data.preprocessed.iter()
            .zip(batch_data.labels.chunks(10))
        {

            let input = Array1::from_vec(preprocessed.to_vec());
            ic_cdk::println!("in:{:?}",&input);
            let output = self.dense2.forward(&input);
            ic_cdk::println!("out:{:?}",&output);
            let target = Array1::from_vec(label.to_vec());

            let loss = cross_entropy_loss(&output, &target);
            ic_cdk::println!("loss:{:?}",&loss);
            total_loss += loss;

            let grad_output = softmax_grad(&output, &target);
            self.dense2.backward(&input, &grad_output, learning_rate);
        }

        total_loss / batch_size as f64
    }*/

    fn get_model_update(&self) -> ModelUpdate {
        ic_cdk::println!("weights:{:?}", self.dense2.weights);
        ic_cdk::println!("bias:{:?}", self.dense2.bias);
        ModelUpdate {
            dense2_weights: self.dense2.weights.clone().rows().into_iter()
                .map(|row| row.to_vec())
                .collect(),
            dense2_bias: self.dense2.bias.clone().to_vec(),
        }
    }
}
struct Dense {
    weights: Array2<f64>,
    bias: Array1<f64>,
}

impl Dense {
    fn new(in_features: usize, out_features: usize) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, (2.0 / (in_features + out_features) as f64).sqrt()).unwrap();
        let weights = Array2::random((out_features, in_features), normal);
        let bias = Array1::from_elem(out_features, 0.01);
        Dense { weights, bias }
    }

    fn forward(&self, input: &[f64]) -> Vec<f64> {
        let input = Array1::from_vec(input.to_vec());
        (self.weights.dot(&input) + &self.bias)
            .mapv(|x| if x.is_finite() { x } else { 0.0 })
            .to_vec()
    }

    fn backward(&self, input: &[f64], grad_output: &[f64]) -> (Vec<f64>, Array2<f64>, Array1<f64>) {
        let input = Array1::from_vec(input.to_vec());
        let grad_output = Array1::from_vec(grad_output.to_vec());

        let grad_input = self.weights.t().dot(&grad_output);

        let grad_weights = Array2::from_shape_fn(
            (grad_output.len(), input.len()),
            |(i, j)| grad_output[i] * input[j]
        );

        let grad_bias = grad_output.to_owned();

        (grad_input.to_vec(), grad_weights, grad_bias)
    }


    fn update(&mut self, grad_weights: &Array2<f64>, grad_bias: &Array1<f64>, learning_rate: f64) {
        self.weights -= &(grad_weights * learning_rate);
        self.bias -= &(grad_bias * learning_rate);
    }
}
fn clip_gradients(gradients: &mut Array2<f64>, max_norm: f64) {
    let norm = gradients.mapv(|x| x.powi(2)).sum().sqrt();
    if norm > max_norm {
        *gradients *= max_norm / norm;
    }
}
fn cross_entropy_loss(output: &Array1<f64>, target: &Array1<f64>) -> f32 {
    let epsilon = 1e-15; // Pour éviter le log(0)
    let loss: f64 = -target.iter().zip(output.iter())
        .map(|(&t, &o)| t * (o.max(epsilon).ln()))
        .sum::<f64>();
    loss as f32
}


fn softmax_grad(output: &Array1<f64>, target: &Array1<f64>) -> Array1<f64> {
    output - target
}

fn relu_1d(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&v| v.max(0.0)).collect()
}
fn leaky_relu_1d(x: &[f64], alpha: f64) -> Vec<f64> {
    x.iter().map(|&v| if v > 0.0 { v } else { alpha * v }).collect()
}
fn leaky_relu_backward_1d(input: &[f64], grad_output: &[f64], alpha: f64) -> Vec<f64> {
    input.iter().zip(grad_output.iter())
        .map(|(&x, &grad)| if x > 0.0 { grad } else { alpha * grad })
        .collect()
}
fn softmax_cross_entropy_grad(output: &[f64], target: &[f64]) -> Vec<f64> {
    output.iter().zip(target.iter()).map(|(&o, &t)| o - t).collect()
}

fn softmax(x: &[f64]) -> Vec<f64> {
    let max_x = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_x: Vec<f64> = x.iter().map(|&xi| (xi - max_x).exp()).collect();
    let sum_exp_x: f64 = exp_x.iter().sum();
    exp_x.into_iter().map(|xi| xi / sum_exp_x).collect()
}
fn relu_backward_1d(input: &[f64], grad_output: &[f64]) -> Vec<f64> {
    input.iter().zip(grad_output.iter())
        .map(|(&x, &grad)| if x > 0.0 { grad } else { 0.0 })
        .collect()
}

fn cross_entropy_loss2(output: &Vec<f64>, target: &[f64]) -> f32 {
    let epsilon = 1e-15; // Pour éviter le log(0)
    let loss: f64 = -target.iter().zip(output.iter())
        .map(|(&t, &o)| t * (o.max(epsilon).ln()))
        .sum::<f64>();
    loss.abs() as f32
}

thread_local! {
    static SERVER_MODEL: RefCell<ServerModel> = RefCell::new(ServerModel::new());
}
/*
#[update]
async fn train(batch_data: BatchData, learning_rate: f64) -> f64 {    SERVER_MODEL.with(|model| {
        let mut model = model.borrow_mut();
        model.train(&batch_data, learning_rate)
    })
}*/
#[derive(CandidType, Deserialize, Clone)]
struct Train2Result {
    loss: f32,
    server_gradients: Vec<Vec<f32>>,
}
#[derive(CandidType, Deserialize,Clone)]
struct ImagesBatch {
    tensor: Vec<Vec<u8>>,
    labels: Vec<Vec<u8>>,
}
#[update]
async fn train2(images_batch: ImagesBatch, learning_rate: f32) -> Train2Result {
    ic_cdk::println!("start train");
    let ImagesBatch { tensor, labels:labs } = images_batch;
    // Convertir les données quantisées en f64
    let encoded_batch: Vec<Vec<f64>> = tensor.into_iter()
        .map(|img| img.into_iter().map(|x| x as f64 / 255.0).collect())
        .collect();
    let labels: Vec<Vec<f64>> = labs.into_iter()
        .map(|lbl| lbl.into_iter().map(|x| x as f64 ).collect())
        .collect();

    SERVER_MODEL.with(|model| {
        let mut model = model.borrow_mut();
        let batch_size = encoded_batch.len();
        let mut total_loss = 0.0;
        let mut all_gradients = Vec::with_capacity(batch_size);
       for (encoded, label) in encoded_batch.iter().zip(labels.iter()) {
           // Propagation avant
           ic_cdk::println!("forward");
           let (dense1_output, dense1_activated, output) = model.forward(encoded);
           ic_cdk::println!("backward");
           // Propagation arrière
           let (gradients, loss) = model.backward(&dense1_output, &dense1_activated, &output ,encoded, label, learning_rate.into());

           total_loss+=loss;
           // Quantisation des gradients
           let quantized_gradients: Vec<f32> = quantize_gradients(gradients);
           //ic_cdk::println!("grad: {:?}", &quantize_gradients);
           all_gradients.push(quantized_gradients);
       }
        ic_cdk::println!("Batch size: {}", batch_size);
        ic_cdk::println!("Total loss: {}", total_loss);
        ic_cdk::println!("Gradients size: {}", all_gradients.len());
        let avg_loss = total_loss / batch_size as f32;
        ic_cdk::println!("Average loss: {}", avg_loss);
       Train2Result {
           loss: avg_loss,
           server_gradients: all_gradients,
       }
    })
}
#[query]
async fn predict_forward(intermediate_activations: Vec<f64>) -> Vec<f64> {
    SERVER_MODEL.with(|model| {
        let model = model.borrow();
        let (_,_,pred) =model.forward(&intermediate_activations);
        pred
    })
}
fn quantize_gradients(grad: Vec<f64>) -> Vec<f32> {
    grad.iter().map(|&x| {
        x as f32
/*        (if x < -1.0 {
            -1.0
        } else if x > 1.0 {
            1.0
        } else {
            x
        } ) as f32*/
    }).collect()
}
#[query]
fn get_model_update() -> ModelUpdate {
    SERVER_MODEL.with(|model| model.borrow().get_model_update())
}


ic_cdk_macros::export_candid!();
