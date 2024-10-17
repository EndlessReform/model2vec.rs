use anyhow::Result;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::{
    extract::State,
    response::Json,
    routing::{get, post},
    Router,
};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};
use hf_hub::api::sync::Api;
use model2vec_infer::{Model2Vec, Model2VecConfig};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs;
use std::net::SocketAddr;
use std::sync::Arc;
use tokenizers::tokenizer::Tokenizer;

const MODEL_NAME: &str = "minishlab/M2V_base_output";

// Support either a String or Vec<String> for the `input` field
#[derive(Deserialize)]
#[serde(untagged)]
enum EmbeddingInputType {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Deserialize)]
struct EmbeddingRequestBody {
    input: EmbeddingInputType,
    model: String,
}

#[derive(Serialize)]
struct EmbeddingObject {
    object: String,
    embedding: Vec<f32>,
    index: usize,
}

impl EmbeddingObject {
    pub fn from_tensor(index: usize, xs: &Tensor) -> Result<Self> {
        Ok(Self {
            object: "embedding".to_string(),
            embedding: xs.to_vec1::<f32>()?,
            index,
        })
    }
}

#[derive(Clone)]
struct AppState {
    tokenizer: Arc<Tokenizer>,
    model: Arc<Model2Vec>,
    device: Arc<Device>,
}

async fn embedding_handler(
    State(state): State<AppState>,
    Json(payload): Json<EmbeddingRequestBody>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    if let EmbeddingInputType::Multiple(ref texts) = payload.input {
        if texts.is_empty() {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "Input vector cannot be empty"})),
            ));
        }
    }
    let token_ids = match payload.input {
        EmbeddingInputType::Single(s) => state.tokenizer.encode_fast(s, false).map(|e| vec![e]),
        EmbeddingInputType::Multiple(inputs) => state.tokenizer.encode_batch_fast(inputs, false),
    }
    .unwrap();
    let tokens_used: usize = token_ids.iter().map(|t| t.len()).sum();

    // TODO: Real batching with padding, not simulated
    let input_tensors: candle_core::Result<Vec<Tensor>> = token_ids
        .iter()
        .map(|e| Tensor::from_slice(e.get_ids(), e.len(), &state.device))
        .collect();
    let input_tensors = input_tensors.map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": "Failed to load IDs to hardware accelerator"})),
        )
    })?;
    let output_embs: candle_core::Result<Vec<Tensor>> = input_tensors
        .iter()
        .map(|input| state.model.forward(input))
        .collect();
    let output_embs = output_embs.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("Failed to embed inputs: {:?}", e)})),
        )
    })?;
    // yeah this is never going to fail
    let output_data: Vec<EmbeddingObject> = output_embs
        .iter()
        .enumerate()
        .map(|(i, e)| EmbeddingObject::from_tensor(i, e).unwrap())
        .collect();

    Ok(Json(json!({
        "object": "list",
        "model": MODEL_NAME,
        "usage": {
            "prompt_tokens": tokens_used,
            "total_tokens": tokens_used,
        },
        "data": output_data
    })))
}

// basic handler that responds with a static string
async fn root() -> &'static str {
    "System is healthy"
}

#[tokio::main]
async fn main() -> Result<()> {
    // Download model if not already existing
    let tokenizer = Tokenizer::from_pretrained(MODEL_NAME, None).unwrap();

    let hf_api = Api::new()?;
    let repo = hf_api.model(MODEL_NAME.to_string());
    let model_fname = repo.get("model.safetensors")?;
    let config_file = fs::File::open(repo.get("config.json")?)?;

    let cfg: Model2VecConfig = serde_json::from_reader(config_file)?;

    // TODO: Add hardware acceleration
    let device = Device::Cpu;
    let dtype = DType::F32;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_fname], dtype, &device)? };
    let model = Model2Vec::load(vb, tokenizer.get_vocab_size(false), cfg.hidden_dim)?;

    let shared_state = AppState {
        tokenizer: Arc::new(tokenizer),
        model: Arc::new(model),
        device: Arc::new(device),
    };

    let app: Router<()> = Router::new()
        .route("/", get(root))
        .route("/v1/embeddings", post(embedding_handler))
        .with_state(shared_state);

    // TODO: Make port configurable
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("Embedding server listening on http://{}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
