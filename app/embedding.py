# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("all-MiniLM-L6-v2")

# def generate_embedding(text: str):
#     embedding = model.encode(text)
#     return embedding.tolist()

from sentence_transformers import SentenceTransformer
import torch

# ✅ Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device=device
)

def generate_embedding(text: str):
    embedding = model.encode(
        text,
        normalize_embeddings=True,   # better for cosine
        convert_to_numpy=True,
        show_progress_bar=False
    )
    return embedding.tolist()
