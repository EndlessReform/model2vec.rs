use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Model2VecConfig {
    pub tokenizer_name: String,
    pub apply_pca: Option<usize>,
    pub apply_zipf: bool,
    pub normalize: bool,
    pub hidden_dim: usize,
    pub seq_length: usize,
}
