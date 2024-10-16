use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};
use hf_hub::api::sync::Api;
use model2vec_infer::{Model2Vec, Model2VecConfig};
use std::fs;
use tokenizers::tokenizer::Tokenizer;

const MODEL_NAME: &str = "minishlab/M2V_base_output";

fn main() -> Result<()> {
    // Download model
    let tokenizer = Tokenizer::from_pretrained(MODEL_NAME, None).unwrap();

    let api = Api::new()?;
    let repo = api.model(MODEL_NAME.to_string());
    let model_fname = repo.get("model.safetensors")?;
    let config_file = fs::File::open(repo.get("config.json")?)?;

    let cfg: Model2VecConfig = serde_json::from_reader(config_file)?;

    // TODO: Add acceleration
    let device = Device::Cpu;
    let dtype = DType::F32;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_fname], dtype, &device)? };
    let model = Model2Vec::load(vb, tokenizer.get_vocab_size(false), cfg.hidden_dim)?;

    // TODO: Customize input!
    let encoding = tokenizer.encode("Hello world", false).unwrap();
    let input_tensor = Tensor::from_slice(encoding.get_ids(), encoding.len(), &device)?;
    let emb = model.forward(&input_tensor)?;

    emb.write_npy("emb_rust.npy")?;
    Ok(())
}
