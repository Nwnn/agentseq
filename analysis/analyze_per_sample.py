import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
import yaml
import os
import sys
sys.path.insert(0, '../src')
from embedding_calculator import EmbeddingCalculator
from embedding_cache import load_cache, save_cache, get_embedding

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def analyze_per_sample(logs_path, config_path):
    # Load logs and config
    logs = pd.read_csv(logs_path)
    config = load_config(config_path)

    # Initialize embedding calculator and cache
    emb_calc = EmbeddingCalculator(config['experiment']['embedding_model'])
    cache = load_cache()

    # Calculate TaskScore for each pipeline
    pipelines = logs.groupby(['extract_id', 'summarize_id', 'classify_id'])
    task_scores = {}
    for pipeline, group in pipelines:
        acc = group['is_correct'].mean()
        task_scores[pipeline] = acc

    # Find best E2 for each E3
    best_e2_for_e3 = {}
    for e3 in ['E3-A', 'E3-B', 'E3-C']:
        best_acc = -1
        best_e2 = None
        for e2 in ['E2-A', 'E2-B', 'E2-C']:
            avg_acc = np.mean([task_scores.get((e1, e2, e3), 0) for e1 in ['E1-A','E1-B','E1-C']])
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_e2 = e2
        best_e2_for_e3[e3] = best_e2

    # Calculate μ_in for E3
    embeddings_in = {}
    for e3, best_e2 in best_e2_for_e3.items():
        subset = logs[(logs['summarize_id'] == best_e2) & (logs['classify_id'] == e3)]
        texts = subset['summarize_text'].tolist()
        embs = [get_embedding(text, emb_calc, cache) for text in texts if pd.notna(text) and isinstance(text, str) and not text.startswith('Error')]
        if embs:
            embeddings_in[e3] = np.mean(embs, axis=0)

    # Calculate per-sample distances
    distances = []
    for idx, row in logs.iterrows():
        e3 = row['classify_id']
        if e3 in embeddings_in and pd.notna(row['summarize_text']) and isinstance(row['summarize_text'], str) and not row['summarize_text'].startswith('Error'):
            y2_emb = get_embedding(row['summarize_text'], emb_calc, cache)
            dist = cosine_distances([y2_emb], [embeddings_in[e3]])[0][0]
            distances.append({
                'sample_id': row['sample_id'],
                'pipeline': (row['extract_id'], row['summarize_id'], row['classify_id']),
                'distance': dist,
                'is_correct': row['is_correct']
            })

    distances_df = pd.DataFrame(distances)

    # Plot distribution
    correct_distances = distances_df[distances_df['is_correct'] == 1]['distance']
    incorrect_distances = distances_df[distances_df['is_correct'] == 0]['distance']

    plt.figure(figsize=(10, 5))
    plt.hist(correct_distances, alpha=0.5, label='Correct', bins=50)
    plt.hist(incorrect_distances, alpha=0.5, label='Incorrect', bins=50)
    plt.xlabel('Cosine Distance to μ_in(E3)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Per-Sample Cosine Distances')
    plt.legend()
    plt.savefig('analysis/per_sample_distances.png')
    plt.show()

    # Statistics
    print("Correct predictions:")
    print(f"Mean distance: {correct_distances.mean():.4f}")
    print(f"Std distance: {correct_distances.std():.4f}")
    print("Incorrect predictions:")
    print(f"Mean distance: {incorrect_distances.mean():.4f}")
    print(f"Std distance: {incorrect_distances.std():.4f}")

    save_cache(cache)

if __name__ == "__main__":
    analyze_per_sample("../logs/experiment_logs.csv", "../config.yaml")