import pandas as pd
import numpy as np
import yaml
import os
import sys

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def analyze_variance_decomposition(logs_path, config_path):
    # Load logs and config
    logs = pd.read_csv(logs_path)
    config = load_config(config_path)

    # Calculate TaskScore for each pipeline
    pipelines = logs.groupby(['extract_id', 'summarize_id', 'classify_id'])
    task_scores = {}
    for pipeline, group in pipelines:
        acc = group['is_correct'].mean()
        task_scores[pipeline] = acc

    # Prepare data: list of (e1, e2, e3, score)
    data = []
    for pipeline, score in task_scores.items():
        e1, e2, e3 = pipeline
        data.append({'e1': e1, 'e2': e2, 'e3': e3, 'score': score})

    df = pd.DataFrame(data)
    y = df['score'].values
    mu = np.mean(y)
    n_total = len(y)

    # Function to compute R^2 for a factor
    def compute_r2(group_col):
        groups = df.groupby(group_col)
        ss_between = 0
        for name, group in groups:
            n_k = len(group)
            mu_k = np.mean(group['score'])
            ss_between += n_k * (mu_k - mu)**2
        ss_total = np.sum((y - mu)**2)
        r2 = ss_between / ss_total if ss_total > 0 else 0
        return r2, ss_between, ss_total

    # Compute for E1, E2, E3
    r2_e1, ss_e1, ss_total = compute_r2('e1')
    r2_e2, ss_e2, _ = compute_r2('e2')
    r2_e3, ss_e3, _ = compute_r2('e3')

    print("Variance Decomposition:")
    print(f"Total SS: {ss_total:.6f}")
    print(f"E1 SS: {ss_e1:.6f}, R^2: {r2_e1:.4f}")
    print(f"E2 SS: {ss_e2:.6f}, R^2: {r2_e2:.4f}")
    print(f"E3 SS: {ss_e3:.6f}, R^2: {r2_e3:.4f}")

if __name__ == "__main__":
    analyze_variance_decomposition("../logs/experiment_logs.csv", "../config.yaml")