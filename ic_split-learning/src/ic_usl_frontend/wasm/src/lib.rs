

use ndarray::prelude::*;
use ndarray_rand::RandomExt;

use web_sys::console;
use core::arch::wasm32::*;
use std::ops::AddAssign;

use wasm_bindgen::prelude::*;
// Utilitaire pour le débogage

use wasm_bindgen::prelude::*;
use ndarray::{Array1, Array2, Array3, Array4, Axis, ViewRepr};
use rand::distributions::Uniform;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use web_sys::js_sys::{Float64Array, Uint8Array};
#[derive(Deserialize)]
struct ModelUpdate {
    dense2_weights: Vec<Vec<f64>>,
    dense2_bias: Vec<f64>,
}
// Structures internes utilisant ndarray
struct InternalClientModel {
    conv1: Conv2D,
    conv2: Conv2D,
    conv1_output: Option<Array3<f64>>,
    conv2_input: Option<Array3<f64>>,
  //  dense1: Dense,
   // dense2: Dense,
}

struct Conv2D {
    filters: Array4<f64>,
    bias: Array1<f64>,
    input: Option<Array3<f64>>,
    output: Option<Array3<f64>>,
}

struct Dense {
    weights: Array2<f64>,
    bias: Array1<f64>,
}

// Structures sérialisables pour wasm-bindgen
#[derive(Serialize, Deserialize)]
struct SerializableConv2D {
    filters: Vec<Vec<Vec<Vec<f64>>>>,
    bias: Vec<f64>,
}

#[derive(Serialize, Deserialize)]
struct SerializableDense {
    weights: Vec<Vec<f64>>,
    bias: Vec<f64>,
}

#[derive(Serialize, Deserialize)]
struct SerializableClientModel {
    conv1: SerializableConv2D,
    conv2: SerializableConv2D,
  //  dense1: SerializableDense,
  //  dense2: SerializableDense,
}

#[wasm_bindgen]
pub struct ClientModel {
    internal: InternalClientModel,
}

#[wasm_bindgen]
impl ClientModel {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        let uniform = Uniform::new(-0.1, 0.1);
        ClientModel {
            internal: InternalClientModel {
                conv1: Conv2D::new(1, 32, 3, uniform),
                conv2: Conv2D::new(32, 64, 3, uniform),
                conv1_output: None,
                conv2_input: None,
                // dense1: Dense::new(64 * 5 * 5, 128, normal),
                //dense2: Dense::new(128, 10, normal),
            }
        }
    }
    // Fonction de quantisation
    fn quantize(values: Vec<f64>) -> Vec<u8> {
        values.into_iter().map(|x| x  as u8).collect()
    }
    fn quantize_and_prune(values: Vec<f64>, threshold: f64) -> Vec<u8> {
        let max_value = values.iter().cloned().fold(f64::MIN, f64::max);
        values.into_iter().map(|x| { ((x / max_value) * 255.0) as u8
            /*if x.abs() < threshold {
                0 // Prune values below the threshold
            } else {
                let clamped = if x < 0.0 {
                    0.0
                } else if x > 1.0 {
                    1.0
                } else {
                    x
                };
                (clamped * 255.0).round() as u8
            }*/
        }).collect()
    }
    #[wasm_bindgen]
    pub fn forward(&mut self, input: &[f64], batch_size: usize) -> Result<JsValue, JsValue> {
        let image_size = 28 * 28;
        let mut batch_outputs = Vec::with_capacity(batch_size);
        if input.len() % image_size != 0 {
            return Err(JsValue::from_str("Input size is not a multiple of 28x28"));
        }
        for i in 0..batch_size {
            let start = i * image_size;
            let end = start + image_size;
            if end > input.len() {
                break; // Sortir de la boucle si nous avons dépassé la fin de l'entrée
            }
            let single_input = &input[start..end];
            //console::log_1(&JsValue::from_str(&format!("input: {:?}",&single_input)));
            let mut x = Array3::from_shape_vec((1, 28, 28), single_input.to_vec()).unwrap();

            // Conv1
            x = self.internal.conv1.forward(&x);
            //console::log_1(&JsValue::from_str(&format!("Après conv1: {:?}", &x)));
            x = relu(&x);
            //console::log_1(&JsValue::from_str(&format!("Après ReLU1: {:?}", &x)));
            x = max_pool2d(&x, 2);
            //console::log_1(&JsValue::from_str(&format!("Après max_pool2d1: {:?}", &x)));
            // Conv2
            x = self.internal.conv2.forward(&x);
            //console::log_1(&JsValue::from_str(&format!("Après conv2: {:?}", &x)));
            x = relu(&x);
            //console::log_1(&JsValue::from_str(&format!("Après ReLU2: {:?}", &x)));
            x = max_pool2d(&x, 2);
            //console::log_1(&JsValue::from_str(&format!("Après max_pool2d2: {:?}", &x)));
            let quantized_output = Self::quantize_and_prune(x.into_raw_vec(),0.01);
            //console::log_1(&JsValue::from_str(&format!("conv2: {:?}",&quantized_output)));
            batch_outputs.push(quantized_output);
        }

        // Sérialiser le Vec<Vec<f64>> en JsValue
        serde_wasm_bindgen::to_value(&batch_outputs).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))



    }
    #[wasm_bindgen]
    pub fn backward(&mut self, input: &[f64], grad_output: &[f64], learning_rate: f64, batch_size: usize) {
        let image_size = 28 * 28;
        let grad_size = 64 * 5 * 5; // Assumant que le gradient de sortie est de cette taille

        // Vérification des dimensions de l'entrée
        if input.len() != batch_size * image_size {
            panic!("Dimensions incompatibles : taille de l'entrée ({}) ne correspond pas à la taille du batch ({}) x image_size ({})", input.len(), batch_size, image_size);
        }

        // Vérification des dimensions du gradient de sortie
        if grad_output.len() != batch_size * grad_size {
            panic!("Dimensions incompatibles : taille du gradient de sortie ({}) ne correspond pas à la taille du batch ({}) x grad_size ({})", grad_output.len(), batch_size, grad_size);
        }

        for i in 0..batch_size {
            let input_start = i * image_size;
            let input_end = input_start + image_size;
            let grad_start = i * grad_size;
            let grad_end = grad_start + grad_size;

            let single_input = &input[input_start..input_end];
            let single_grad_output = &grad_output[grad_start..grad_end];

            let x = Array3::from_shape_vec((1, 28, 28), single_input.to_vec()).unwrap();
            let mut grad = Array3::from_shape_vec((64, 5, 5), single_grad_output.to_vec()).unwrap();

            // Initialisation des gradients accumulés
            let mut grad_filters_conv2_accum = Array4::<f64>::zeros(self.internal.conv2.filters.dim());
            let mut grad_bias_conv2_accum = Array1::<f64>::zeros(self.internal.conv2.bias.dim());
            let mut grad_filters_conv1_accum = Array4::<f64>::zeros(self.internal.conv1.filters.dim());
            let mut grad_bias_conv1_accum = Array1::<f64>::zeros(self.internal.conv1.bias.dim());

            // Backward pass for conv2
            if let (Some(conv2_input), Some(conv2_output)) = (&self.internal.conv2_input, &self.internal.conv2.output) {
                grad = max_pool2d_backward(conv2_output, &grad, 2);
                grad = relu_backward(conv2_input, &grad);
                let (grad_conv1, grad_filters_conv2, grad_bias_conv2) = self.internal.conv2.backward(conv2_input, &grad);
                grad_filters_conv2_accum += &grad_filters_conv2;
                grad_bias_conv2_accum += &grad_bias_conv2;

                // Backward pass for conv1
                if let Some(conv1_output) = &self.internal.conv1_output {
                    grad = max_pool2d_backward(conv1_output, &grad_conv1, 2);
                    grad = relu_backward(&self.internal.conv1.input.as_ref().unwrap(), &grad);
                    let (_, grad_filters_conv1, grad_bias_conv1) = self.internal.conv1.backward(&x, &grad);
                    grad_filters_conv1_accum += &grad_filters_conv1;
                    grad_bias_conv1_accum += &grad_bias_conv1;
                }
            }

            // Mise à jour des poids après l'accumulation des gradients
            self.internal.conv2.update(&grad_filters_conv2_accum, &grad_bias_conv2_accum, learning_rate);
            self.internal.conv1.update(&grad_filters_conv1_accum, &grad_bias_conv1_accum, learning_rate);
        }
    }
   /* pub fn preprocess_batch(&self, images: &[f64], labels: &[f64], batch_size: usize) -> JsValue {
        console::log_1(&JsValue::from_str(&format!("Images length: {}, Labels length: {}, Batch size: {}", images.len(), labels.len(), batch_size)));

        let image_size = 28 * 28;
        let num_images = images.len() / image_size;

        if num_images != batch_size {
            console::log_1(&JsValue::from_str(&format!("Warning: Number of images ({}) doesn't match batch size ({})", num_images, batch_size)));
        }

        let preprocessed = (0..num_images)
            .map(|i| {
                let start = i * image_size;
                let end = start + image_size;
                let image = Array3::from_shape_vec((1, 28, 28), images[start..end].to_vec()).unwrap();

                let x = self.internal.conv1.forward(&image);
                let x = relu(&x);
                let x = max_pool2d(&x, 2);
                let x = self.internal.conv2.forward(&x);
                let x = relu(&x);
                let x = max_pool2d(&x, 2);
                 x.into_raw_vec()
              //  self.internal.dense1.forward(&x)
            })
            .collect::<Vec<_>>();

        let batch_data = BatchData {
            preprocessed,
            labels: labels.to_vec(),
        };

        serde_wasm_bindgen::to_value(&batch_data).unwrap()
    }*/

    pub fn serialize(&self) -> JsValue {
        let serializable = SerializableClientModel {
            conv1: self.internal.conv1.to_serializable(),
            conv2: self.internal.conv2.to_serializable(),
       //     dense1: self.internal.dense1.to_serializable(),
       //     dense2: self.internal.dense2.to_serializable(),
        };
        serde_wasm_bindgen::to_value(&serializable).unwrap()
    }

    pub fn deserialize(&mut self, js_value: JsValue) -> Result<(), JsValue> {
        let serializable: SerializableClientModel = serde_wasm_bindgen::from_value(js_value)?;
        self.internal.conv1 = Conv2D::from_serializable(serializable.conv1);
        self.internal.conv2 = Conv2D::from_serializable(serializable.conv2);
      //  self.internal.dense1 = Dense::from_serializable(serializable.dense1);
      //  self.internal.dense2 = Dense::from_serializable(serializable.dense2);
        Ok(())
    }
    #[wasm_bindgen]
    pub fn load_model(&mut self, weights: &[f64], bias: &[f64]) -> Result<(), JsValue> {
        let weights_shape = (bias.len(), weights.len() / bias.len());

       /* self.internal.dense2.weights = Array2::from_shape_vec(weights_shape, weights.to_vec())
            .map_err(|e| JsValue::from_str(&format!("Failed to create weights array: {}", e)))?;

        self.internal.dense2.bias = Array1::from(bias.to_vec());
*/
        Ok(())
    }

}

impl Conv2D {
    fn new(in_channels: usize, out_channels: usize, kernel_size: usize, uniform: Uniform<f64>) -> Self {
        Conv2D {
            filters: Array4::random((out_channels, in_channels, kernel_size, kernel_size), uniform),
            bias: Array1::zeros(out_channels),
            input: None,
            output: None,
        }
    }

    fn forward(&mut self, input: &Array3<f64>) -> Array3<f64> {
        self.input = Some(input.clone());
        let (num_filters, in_channels, kernel_height, kernel_width) = self.filters.dim();
        let (_, input_height, input_width) = input.dim();
        let out_height = input_height - kernel_height + 1;
        let out_width = input_width - kernel_width + 1;

        let mut output = Array3::<f64>::zeros((num_filters, out_height, out_width));
        let mut flat_view = vec![0.0; in_channels * kernel_height * kernel_width];

        for f in 0..num_filters {
            for h in 0..out_height {
                for w in 0..out_width {
                    flat_view.clear();
                    for c in 0..in_channels {
                        let view = input.slice(s![c, h..h+kernel_height, w..w+kernel_width]);
                        flat_view.extend(view.iter().cloned());
                    }

                    let filter_slice = self.filters.slice(s![f, .., .., ..]);
                    let sum: f64 = filter_slice.iter().zip(flat_view.iter())
                        .map(|(&a, &b)| a * b)
                        .sum();

                    output[[f, h, w]] = sum + self.bias[f];
                }
            }
        }
        self.output = Some(output.clone());
        output
    }


    fn backward(&self, input: &Array3<f64>, grad_output: &Array3<f64>) -> (Array3<f64>, Array4<f64>, Array1<f64>) {
        let (num_filters, _, kernel_height, kernel_width) = self.filters.dim();
        let (in_channels, input_height, input_width) = input.dim();
        let (_, out_height, out_width) = grad_output.dim();

        // Initialiser les gradients
        let mut grad_input = Array3::<f64>::zeros((in_channels, input_height, input_width));
        let mut grad_filters = Array4::<f64>::zeros(self.filters.dim());
        let mut grad_bias = Array1::<f64>::zeros(num_filters);

        // Calculer les gradients
        for f in 0..num_filters {
            for h in 0..out_height {
                for w in 0..out_width {
                    let h_start = h;
                    let h_end = h + kernel_height;
                    let w_start = w;
                    let w_end = w + kernel_width;

                    let input_slice = input.slice(s![.., h_start..h_end, w_start..w_end]);
                    let grad_output_value = grad_output[[f, h, w]];

                    // Gradient des filtres
                    grad_filters.slice_mut(s![f, .., .., ..])
                        .add_assign(&(&input_slice * grad_output_value));

                    // Gradient de l'entrée
                    grad_input.slice_mut(s![.., h_start..h_end, w_start..w_end])
                        .add_assign(&(&self.filters.slice(s![f, .., .., ..]) * grad_output_value));

                    // Gradient du biais
                    grad_bias[f] += grad_output_value;
                }
            }
        }

        (grad_input, grad_filters, grad_bias)
    }

    fn update(&mut self, grad_filters: &Array4<f64>, grad_bias: &Array1<f64>, learning_rate: f64) {
        // Mise à jour des filtres
        self.filters -= &(grad_filters * learning_rate);

        // Mise à jour des biais
        self.bias -= &(grad_bias * learning_rate);
    }
    fn to_serializable(&self) -> SerializableConv2D {
        SerializableConv2D {
            filters: self.filters.outer_iter()
                .map(|f| f.outer_iter()
                    .map(|c| c.rows().into_iter()
                        .map(|r| r.to_vec())
                        .collect())
                    .collect())
                .collect(),
            bias: self.bias.to_vec(),
        }
    }

    fn from_serializable(serializable: SerializableConv2D) -> Self {
        Conv2D {
            filters: Array4::from_shape_vec(
                (serializable.filters.len(),
                 serializable.filters[0].len(),
                 serializable.filters[0][0].len(),
                 serializable.filters[0][0][0].len()),
                serializable.filters.into_iter().flatten().flatten().flatten().collect()
            ).unwrap(),
            bias: Array1::from(serializable.bias),
            input: None,
            output: None,
        }
    }
}

impl Dense {
    fn new(in_features: usize, out_features: usize, normal: Normal<f64>) -> Self {
        Dense {
            weights: Array2::random((out_features, in_features), normal),
            bias: Array1::zeros(out_features),
        }
    }

    fn forward(&self, input: &[f64]) -> Vec<f64> {
        let input = Array1::from_vec(input.to_vec());
        (self.weights.dot(&input) + &self.bias).to_vec()
    }

    fn to_serializable(&self) -> SerializableDense {
        SerializableDense {
            weights: self.weights.outer_iter().map(|row| row.to_vec()).collect(),
            bias: self.bias.to_vec(),
        }
    }

    fn from_serializable(serializable: SerializableDense) -> Self {
        Dense {
            weights: Array2::from_shape_vec(
                (serializable.weights.len(), serializable.weights[0].len()),
                serializable.weights.into_iter().flatten().collect()
            ).unwrap(),
            bias: Array1::from(serializable.bias),
        }
    }
}

#[derive(Serialize, Deserialize)]
struct BatchData {
    preprocessed: Vec<Vec<f64>>,
    labels: Vec<f64>,
}


fn relu(x: &Array3<f64>) -> Array3<f64> {
    x.mapv(|a| a.max(0.0))
}

fn relu_1d(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&a| a.max(0.0)).collect()
}

fn max_pool2d(x: &Array3<f64>, kernel_size: usize) -> Array3<f64> {
    let (channels, height, width) = x.dim();
    let out_height = height / kernel_size;
    let out_width = width / kernel_size;

    let mut output = Array3::<f64>::zeros((channels, out_height, out_width));

    for c in 0..channels {
        for h in 0..out_height {
            for w in 0..out_width {
                let pool_region = x.slice(s![c,
                    h*kernel_size..(h+1)*kernel_size,
                    w*kernel_size..(w+1)*kernel_size
                ]);
                output[[c, h, w]] = pool_region.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            }
        }
    }

    output
}

fn max_pool2d_backward(input: &Array3<f64>, grad_output: &Array3<f64>, kernel_size: usize) -> Array3<f64> {
    let (channels, height, width) = input.dim();
    let (_, out_height, out_width) = grad_output.dim();
    let mut grad_input = Array3::<f64>::zeros((channels, height, width));

    for c in 0..channels {
        for h in 0..out_height {
            for w in 0..out_width {
                let h_start = h * kernel_size;
                let h_end = (h + 1) * kernel_size;
                let w_start = w * kernel_size;
                let w_end = (w + 1) * kernel_size;

                let pool_region = input.slice(s![c, h_start..h_end, w_start..w_end]);
                let (max_index, _) = pool_region.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap();

                let max_h = max_index / kernel_size;
                let max_w = max_index % kernel_size;

                grad_input[[c, h_start + max_h, w_start + max_w]] = grad_output[[c, h, w]];
            }
        }
    }

    grad_input
}

fn relu_backward(input: &Array3<f64>, grad_output: &Array3<f64>) -> Array3<f64> {
    input.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }) * grad_output
}

// Version 1D de relu_backward pour les couches denses
fn relu_backward_1d(input: &[f64], grad_output: &[f64]) -> Vec<f64> {
    input.iter()
        .zip(grad_output.iter())
        .map(|(&x, &grad)| if x > 0.0 { grad } else { 0.0 })
        .collect()
}

fn prune_weights(weights: &mut [f64], threshold: f64) {
    for weight in weights.iter_mut() {
        if weight.abs() < threshold {
            *weight = 0.0;
        }
    }
}