import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
from umap import UMAP
import yaml
import os
import sys
sys.path.insert(0, '../src')
from embedding_calculator import EmbeddingCalculator
from embedding_cache import load_cache, save_cache, get_embedding

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def analyze_results(logs_path, config_path):
    # Load logs and config
    logs = pd.read_csv(logs_path)
    config = load_config(config_path)

    # Initialize embedding calculator and cache
    emb_calc = EmbeddingCalculator(config['experiment']['embedding_model'])
    cache = load_cache()

    # Group by pipeline
    pipelines = logs.groupby(['extract_id', 'summarize_id', 'classify_id'])

    # Calculate TaskScore for each pipeline
    task_scores = {}
    for pipeline, group in pipelines:
        acc = group['is_correct'].mean()
        task_scores[pipeline] = acc

    # Calculate TaskScore for each agent pair
    task_score_e1 = {}
    task_score_e2 = {}
    task_score_e3 = {}
    for e1 in ['E1-A', 'E1-B', 'E1-C']:
        accs = [task_scores[(e1, e2, e3)] for e2 in ['E2-A','E2-B','E2-C'] for e3 in ['E3-A','E3-B','E3-C']]
        task_score_e1[e1] = np.mean(accs)
    for e2 in ['E2-A', 'E2-B', 'E2-C']:
        accs = [task_scores[(e1, e2, e3)] for e1 in ['E1-A','E1-B','E1-C'] for e3 in ['E3-A','E3-B','E3-C']]
        task_score_e2[e2] = np.mean(accs)
    for e3 in ['E3-A', 'E3-B', 'E3-C']:
        accs = [task_scores[(e1, e2, e3)] for e1 in ['E1-A','E1-B','E1-C'] for e2 in ['E2-A','E2-B','E2-C']]
        task_score_e3[e3] = np.mean(accs)

    # Find best E1 for each E2
    best_e1_for_e2 = {}
    for e2 in ['E2-A', 'E2-B', 'E2-C']:
        best_acc = -1
        best_e1 = None
        for e1 in ['E1-A', 'E1-B', 'E1-C']:
            avg_acc = np.mean([task_scores[(e1, e2, e3)] for e3 in ['E3-A','E3-B','E3-C']])
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_e1 = e1
        best_e1_for_e2[e2] = best_e1

    # Find best E2 for each E3
    best_e2_for_e3 = {}
    for e3 in ['E3-A', 'E3-B', 'E3-C']:
        best_acc = -1
        best_e2 = None
        for e2 in ['E2-A', 'E2-B', 'E2-C']:
            avg_acc = np.mean([task_scores[(e1, e2, e3)] for e1 in ['E1-A','E1-B','E1-C']])
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_e2 = e2
        best_e2_for_e3[e3] = best_e2

    # Calculate μ_out for each agent
    embeddings_out = {'extract': {}, 'summarize': {}, 'classify': {}}
    for step in ['extract', 'summarize', 'classify']:
        step_texts = logs.groupby(f'{step}_id')[f'{step}_text'].apply(list)
        for agent_id, texts in step_texts.items():
            embs = [get_embedding(text, emb_calc, cache) for text in texts if pd.notna(text) and isinstance(text, str) and not text.startswith('Error')]
            if embs:
                embeddings_out[step][agent_id] = np.mean(embs, axis=0)

    # Calculate μ_in for E2 and E3
    embeddings_in = {'summarize': {}, 'classify': {}}
    for e2, best_e1 in best_e1_for_e2.items():
        # y1 texts where extract_id == best_e1 and summarize_id == e2
        subset = logs[(logs['extract_id'] == best_e1) & (logs['summarize_id'] == e2)]
        texts = subset['extract_text'].tolist()
        embs = [get_embedding(text, emb_calc, cache) for text in texts if pd.notna(text) and isinstance(text, str) and not text.startswith('Error')]
        if embs:
            embeddings_in['summarize'][e2] = np.mean(embs, axis=0)

    for e3, best_e2 in best_e2_for_e3.items():
        # y2 texts where summarize_id == best_e2 and classify_id == e3
        subset = logs[(logs['summarize_id'] == best_e2) & (logs['classify_id'] == e3)]
        texts = subset['summarize_text'].tolist()
        embs = [get_embedding(text, emb_calc, cache) for text in texts if pd.notna(text) and isinstance(text, str) and not text.startswith('Error')]
        if embs:
            embeddings_in['classify'][e3] = np.mean(embs, axis=0)

    # Calculate distances
    dist12 = {}
    dist23 = {}
    for e1 in embeddings_out['extract']:
        for e2 in embeddings_in['summarize']:
            if e1 in embeddings_out['extract'] and e2 in embeddings_in['summarize']:
                dist = cosine_distances([embeddings_out['extract'][e1]], [embeddings_in['summarize'][e2]])[0][0]
                dist12[(e1, e2)] = dist

    for e2 in embeddings_out['summarize']:
        for e3 in embeddings_in['classify']:
            if e2 in embeddings_out['summarize'] and e3 in embeddings_in['classify']:
                dist = cosine_distances([embeddings_out['summarize'][e2]], [embeddings_in['classify'][e3]])[0][0]
                dist23[(e2, e3)] = dist

    # Plot TaskScore vs Dist12
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    x = [dist12[(p[0], p[1])] for p in task_scores if (p[0], p[1]) in dist12]
    y = [task_scores[p] for p in task_scores if (p[0], p[1]) in dist12]
    plt.scatter(x, y)
    plt.xlabel('Dist12')
    plt.ylabel('TaskScore')
    plt.title('TaskScore vs Dist12')

    # Plot TaskScore vs Dist23
    plt.subplot(1, 2, 2)
    x = [dist23[(p[1], p[2])] for p in task_scores if (p[1], p[2]) in dist23]
    y = [task_scores[p] for p in task_scores if (p[1], p[2]) in dist23]
    plt.scatter(x, y)
    plt.xlabel('Dist23')
    plt.ylabel('TaskScore')
    plt.title('TaskScore vs Dist23')
    plt.tight_layout()
    plt.savefig('analysis/scatter_plots.png')
    plt.show()

    # PCA and UMAP visualization
    all_embs = []
    labels = []
    for step, embs in embeddings.items():
        for agent, emb in embs.items():
            all_embs.append(emb)
            labels.append(f'{step}_{agent}')

    if all_embs:
        # PCA
        pca = PCA(n_components=2)
        reduced_pca = pca.fit_transform(all_embs)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        for i, label in enumerate(labels):
            plt.scatter(reduced_pca[i, 0], reduced_pca[i, 1], label=label)
        plt.legend()
        plt.title('PCA of Agent Embeddings')

        # UMAP
        umap = UMAP(n_components=2, random_state=42)
        reduced_umap = umap.fit_transform(all_embs)
        plt.subplot(1, 2, 2)
        for i, label in enumerate(labels):
            plt.scatter(reduced_umap[i, 0], reduced_umap[i, 1], label=label)
        plt.legend()
        plt.title('UMAP of Agent Embeddings')

        plt.tight_layout()
        plt.savefig('analysis/embedding_visualization.png')
        plt.show()

    save_cache(cache)

if __name__ == "__main__":
    analyze_results("../logs/experiment_logs.csv", "../config.yaml")