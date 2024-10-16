from model2vec import StaticModel
import numpy as np


def main():
    print("Downloading model")
    model_name = "minishlab/M2V_base_output"
    model = StaticModel.from_pretrained(model_name)

    embeddings = model.encode("Hello world")
    reference_embedding = np.load("model2vec_infer/emb_rust.npy")

    print(np.allclose(embeddings, reference_embedding, atol=1e-5))


if __name__ == "__main__":
    main()
