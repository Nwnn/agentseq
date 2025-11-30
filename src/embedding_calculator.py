from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingCalculator:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def compute_embeddings(self, texts):
        """
        Compute embeddings for a list of texts.

        Args:
            texts (list): List of text strings.

        Returns:
            np.ndarray: Embeddings array of shape (len(texts), embedding_dim)
        """
        return self.model.encode(texts, convert_to_numpy=True)

    def compute_distance(self, emb1, emb2):
        """
        Compute cosine distance between two embeddings.

        Args:
            emb1 (np.ndarray): Embedding 1
            emb2 (np.ndarray): Embedding 2

        Returns:
            float: Cosine distance (1 - cosine_similarity)
        """
        sim = cosine_similarity([emb1], [emb2])[0][0]
        return 1 - sim

    def compute_mean_embedding(self, embeddings):
        """
        Compute mean embedding.

        Args:
            embeddings (np.ndarray): Array of embeddings.

        Returns:
            np.ndarray: Mean embedding.
        """
        return np.mean(embeddings, axis=0)