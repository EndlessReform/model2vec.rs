use super::config::Model2VecConfig;
use super::model::Model2Vec;
use candle_core::{error::Error as E, Device, Result, Tensor, D};
use candle_nn::{Embedding, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config as BertConfig, DTYPE};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

const BATCH_SIZE: usize = 1024;

fn get_bos_eos(tokenizer: &Tokenizer) -> tokenizers::Result<(u32, u32)> {
    let enc = tokenizer.encode("a", false)?;
    let ids = enc.get_ids().to_vec();
    match ids.len() {
        3 => Ok((ids[0], ids[2])),
        _ => Err("Expected BOS and EOS".into()),
    }
}

pub fn create_output_embs_from_model(model: &BertModel, tokenizer: &Tokenizer) -> Result<Tensor> {
    let vocab_size = tokenizer.get_vocab_size(false);
    let (bos_id, eos_id) = get_bos_eos(tokenizer).map_err(|e| {
        candle_core::Error::Msg(format!("Error getting EOS and BOS tokens: {:?}", e))
    })?;

    let all_ids = Tensor::arange::<u32>(0, vocab_size as u32, &model.device)?;
    let bos_ids = Tensor::full(bos_id, all_ids.shape(), &model.device)?;
    let eos_ids = Tensor::full(eos_id, all_ids.shape(), &model.device)?;
    let stacked = Tensor::stack(&[bos_ids, all_ids, eos_ids], 1)?;

    let n_chunks = (vocab_size + BATCH_SIZE - 1) / BATCH_SIZE;
    let batches = stacked.chunk(n_chunks, 0)?;

    let embeds: Result<Vec<Tensor>> = batches
        .iter()
        .map(|b| model.forward(b, &b.zeros_like().unwrap(), None))
        .map(|b| b.and_then(|t| t.squeeze(0)))
        .collect();

    let x = Tensor::cat(&embeds?, 0)?;
    println!("Shape: {:?}", x.shape());
    x.mean(D::Minus2)
}

pub fn distill_from_model(
    model: &BertModel,
    tokenizer: &Tokenizer,
    model_id: &str,
    pca_dim: Option<usize>,
    apply_zipf: bool,
) -> Result<(Model2Vec, Model2VecConfig)> {
    let embs = create_output_embs_from_model(model, tokenizer)?;
    // Ignore unused tokens for now
    let hidden_dim = embs.dim(D::Minus1)?;

    let config = Model2VecConfig {
        tokenizer_name: model_id.into(),
        apply_pca: pca_dim,
        apply_zipf,
        // Setting to arbitrary length
        seq_length: 1000000,
        normalize: false,
        hidden_dim,
    };
    let model = Model2Vec::new(Embedding::new(embs, hidden_dim));
    Ok((model, config))
}

pub fn distill(
    model_id: &str,
    pca_dim: Option<usize>,
    apply_zipf: bool,
    device: &Device,
) -> anyhow::Result<(Model2Vec, Model2VecConfig)> {
    let (config_filename, tokenizer_filename, weights_filename) = {
        let api = Api::new()?;
        let api = api.model(model_id.to_string());
        let config = api.get("config.json")?;
        let tokenizer = api.get("tokenizer.json")?;
        let weights = api.get("model.safetensors")?;
        (config, tokenizer, weights)
    };
    let config = std::fs::read_to_string(config_filename)?;
    let config: BertConfig = serde_json::from_str(&config)?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename)
        .map_err(|e| E::Msg(format!("{:?}", e).to_string()))?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };
    let model = BertModel::load(vb, &config)?;
    let res = distill_from_model(&model, &tokenizer, model_id, pca_dim, apply_zipf)?;
    Ok(res)
}
