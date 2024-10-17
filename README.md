# model2vec.rs

![model2vec](./docs/model2vec.png)

An OpenAI-compatible embeddings server using Minish Labs' [Model2Vec](https://github.com/MinishLab/model2vec).

Features:
- Single binary, less than 10MB. (The model is only 30MB!)

to [candle.rs](https://github.com/huggingface/candle).


## What is Model2Vec, anyway?

Model2Vec is a **static embedding model**.
Similar to [Word2Vec](https://en.wikipedia.org/wiki/Word2vec), it uses mean-pooled bag-of-word embeddings.
However, each token embedding is that token's output from a sentence transformer model (like SBERT or Nomic Embeddings).
Surprisingly enough, the authors find that this has MTEB classification performance very close to the original model (though clustering and reranking are predictably much lower).

This approach is very fast!

For a full explanation, please read Minish Labs' [official deep dive](https://huggingface.co/blog/Pringled/model2vec).

## Usage

Run server, building from source:

```bash
cargo run --bin model2vec_server
```

## Roadmap

- [X] OpenAI-compatible inference
- [ ] Batching
- [ ] Distillation
- [ ] Custom vocabulary
- [ ] Hardware acceleration from Candle.rs
- [ ] WASM bindings
