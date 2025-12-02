import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

def analyze_3d_distances(logs_path, config_path):
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

    # Find best pairs
    best_e1_for_e2 = {}
    for e2 in ['E2-A', 'E2-B', 'E2-C']:
        best_acc = -1
        best_e1 = None
        for e1 in ['E1-A', 'E1-B', 'E1-C']:
            avg_acc = np.mean([task_scores.get((e1, e2, e3), 0) for e3 in ['E3-A','E3-B','E3-C']])
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_e1 = e1
        best_e1_for_e2[e2] = best_e1

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

    # Calculate μ_in
    embeddings_in = {'summarize': {}, 'classify': {}}
    for e2, best_e1 in best_e1_for_e2.items():
        subset = logs[(logs['extract_id'] == best_e1) & (logs['summarize_id'] == e2)]
        texts = subset['extract_text'].tolist()
        embs = [get_embedding(text, emb_calc, cache) for text in texts if pd.notna(text) and isinstance(text, str) and not text.startswith('Error')]
        if embs:
            embeddings_in['summarize'][e2] = np.mean(embs, axis=0)

    for e3, best_e2 in best_e2_for_e3.items():
        subset = logs[(logs['summarize_id'] == best_e2) & (logs['classify_id'] == e3)]
        texts = subset['summarize_text'].tolist()
        embs = [get_embedding(text, emb_calc, cache) for text in texts if pd.notna(text) and isinstance(text, str) and not text.startswith('Error')]
        if embs:
            embeddings_in['classify'][e3] = np.mean(embs, axis=0)

    # Calculate μ_out
    embeddings_out = {'extract': {}, 'summarize': {}, 'classify': {}}
    for step in ['extract', 'summarize', 'classify']:
        step_texts = logs.groupby(f'{step}_id')[f'{step}_text'].apply(list)
        for agent_id, texts in step_texts.items():
            embs = [get_embedding(text, emb_calc, cache) for text in texts if pd.notna(text) and isinstance(text, str) and not text.startswith('Error')]
            if embs:
                embeddings_out[step][agent_id] = np.mean(embs, axis=0)

    # Calculate distances for each sample
    distances = []
    for idx, row in logs.iterrows():
        e1, e2, e3 = row['extract_id'], row['summarize_id'], row['classify_id']
        if (pd.notna(row['extract_text']) and isinstance(row['extract_text'], str) and not row['extract_text'].startswith('Error') and
            pd.notna(row['summarize_text']) and isinstance(row['summarize_text'], str) and not row['summarize_text'].startswith('Error') and
            e2 in embeddings_in['summarize'] and e3 in embeddings_in['classify']):
            y1_emb = get_embedding(row['extract_text'], emb_calc, cache)
            y2_emb = get_embedding(row['summarize_text'], emb_calc, cache)
            dist12 = cosine_distances([y1_emb], [embeddings_in['summarize'][e2]])[0][0]
            dist23 = cosine_distances([y2_emb], [embeddings_in['classify'][e3]])[0][0]
            total_dist = dist12 + dist23
            distances.append({
                'sample_id': row['sample_id'],
                'dist12': dist12,
                'dist23': dist23,
                'total_dist': total_dist,
                'is_correct': row['is_correct']
            })

    distances_df = pd.DataFrame(distances)

    # 2D plot: Dist12 vs Dist23, colored by is_correct
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(distances_df['dist12'], distances_df['dist23'], c=distances_df['is_correct'], cmap='coolwarm', alpha=0.5, s=1)
    plt.xlabel('Dist12')
    plt.ylabel('Dist23')
    plt.colorbar(sc, label='Is Correct')
    plt.title('Dist12 vs Dist23 (Per-Sample)')
    plt.grid(True)
    plt.savefig('analysis/dist12_dist23_2d.png')
    plt.show()

    save_cache(cache)

if __name__ == "__main__":
    analyze_3d_distances("../logs/experiment_logs.csv", "../config.yaml")