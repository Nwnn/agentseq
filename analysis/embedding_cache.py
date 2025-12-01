import pickle
import os
import numpy as np

CACHE_FILE = 'analysis/embeddings_cache.pkl'

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)

def get_embedding(text, emb_calc, cache):
    if text in cache:
        return cache[text]
    emb = emb_calc.compute_embeddings([text])[0]
    cache[text] = emb
    return emb