use candle_core::{Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct Model2VecConfig {
    pub tokenizer_name: String,
    pub apply_pca: Option<usize>,
    pub apply_zipf: bool,
    pub normalize: bool,
    pub hidden_dim: usize,
    pub seq_length: usize,
}

pub struct Model2Vec {
    emb: Embedding,
}

impl Model2Vec {
    pub fn load(vb: VarBuilder, vocab_size: usize, dim: usize) -> Result<Self> {
        Ok(Self {
            emb: Embedding::new(vb.get((vocab_size, dim), "embeddings")?, dim),
        })
    }
}

impl Module for Model2Vec {
    fn forward(&self, xs: &candle_core::Tensor) -> Result<Tensor> {
        let embs = self.emb.forward(xs)?;
        embs.mean(0)
    }
}
