extern crate cfg_if;
extern crate wasm_bindgen;

use std::io::Read;
use cfg_if::cfg_if;
use wasm_bindgen::prelude::*;
use ndarray::{Array1, Array2};

cfg_if! {
    if #[cfg(feature = "wee_alloc")] {
        extern crate wee_alloc;
        #[global_allocator]
        static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;
    }
}

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

#[wasm_bindgen]
pub struct MLP {
    weights: Vec<Array2<f32>>,
    biases: Vec<Array1<f32>>,
}

#[wasm_bindgen]
impl MLP {
    #[wasm_bindgen(constructor)]
    pub fn new(data: &[u8]) -> Self {
        let (weights, biases) = Self::parse_weights_and_biases(data);
        MLP { weights, biases }
    }

    fn parse_weights_and_biases(data: &[u8]) -> (Vec<Array2<f32>>, Vec<Array1<f32>>) {
        let mut cursor = std::io::Cursor::new(data);

        let mut num_layers = [0; 4];
        cursor.read_exact(&mut num_layers).unwrap();
        let num_layers = i32::from_le_bytes(num_layers);

        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for _ in 0..num_layers {
            let mut rows = [0; 4];
            cursor.read_exact(&mut rows).unwrap();
            let rows = i32::from_le_bytes(rows);

            let mut cols = [0; 4];
            cursor.read_exact(&mut cols).unwrap();
            let cols = i32::from_le_bytes(cols);

            let mut weight_matrix = Array2::<f32>::zeros((rows as usize, cols as usize));
            for i in 0..(rows * cols) {
                let mut weight = [0; 4];
                cursor.read_exact(&mut weight).unwrap();
                weight_matrix[((i / cols) as usize, (i % cols) as usize)] = f32::from_le_bytes(weight);
            }
            weights.push(weight_matrix);

            let mut bias_size = [0; 4];
            cursor.read_exact(&mut bias_size).unwrap();
            let bias_size = i32::from_le_bytes(bias_size);

            let mut bias_vector = Array1::<f32>::zeros(bias_size as usize);
            for i in 0..bias_size {
                let mut bias = [0; 4];
                cursor.read_exact(&mut bias).unwrap();
                bias_vector[i as usize] = f32::from_le_bytes(bias);
            }
            biases.push(bias_vector);
        }

        (weights, biases)
    }

    #[wasm_bindgen]
    pub fn predict(&self, image_data: &[f32]) -> Vec<f32> {
        let input_image = Array1::from_vec(image_data.to_vec());
        let mut layer_output = input_image.to_shape((1, 784)).unwrap().into_owned();

        for (weight_matrix, bias_vector) in self.weights.iter().zip(self.biases.iter()) {
            layer_output = layer_output.dot(weight_matrix) + bias_vector.clone();
            layer_output.mapv_inplace(relu);
        }

        let logits = layer_output.to_shape(10).unwrap();
        let probabilities = softmax(&logits.to_owned());

        probabilities.to_vec()
    }
}

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

fn softmax(logits: &Array1<f32>) -> Array1<f32> {
    let max_logit = logits.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_logits: Array1<f32> = logits.mapv(|x| (x - max_logit).exp());
    let sum_exp_logits = exp_logits.sum();
    exp_logits / sum_exp_logits
}
