use candle_core::{Result, Tensor, D};
use candle_nn::{Embedding, Module, VarBuilder};

pub struct Model2Vec {
    emb: Embedding,
}

impl Model2Vec {
    pub fn new(emb: Embedding) -> Self {
        Self { emb }
    }
    pub fn load(vb: VarBuilder, vocab_size: usize, dim: usize) -> Result<Self> {
        Ok(Self {
            emb: Embedding::new(vb.get((vocab_size, dim), "embeddings")?, dim),
        })
    }
}

impl Module for Model2Vec {
    fn forward(&self, xs: &candle_core::Tensor) -> Result<Tensor> {
        let embs = self.emb.forward(xs)?;
        embs.mean(D::Minus2)
    }
}
