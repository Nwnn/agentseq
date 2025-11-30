import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
import yaml
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
load_dotenv()

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def analyze_results(results_path, config_path):
    config = load_config(config_path)
    results = pd.read_csv(results_path)

    # Normalize scores if needed
    results['task_score_norm'] = (results['task_score'] - results['task_score'].min()) / (results['task_score'].max() - results['task_score'].min())
    results['total_dist_norm'] = (results['total_dist'] - results['total_dist'].min()) / (results['total_dist'].max() - results['total_dist'].min())

    # Local best strategy: max task_score
    local_best = results.loc[results['task_score'].idxmax()]
    print(f"Local Best: {local_best['pipeline']}, Acc: {local_best['final_acc']}, Task Score: {local_best['task_score']}, Dist: {local_best['total_dist']}")

    # Graph best strategy: max(task_score - lambda * total_dist)
    lambda_val = config['experiment']['lambda']
    results['graph_score'] = results['task_score'] - lambda_val * results['total_dist']
    graph_best = results.loc[results['graph_score'].idxmax()]
    print(f"Graph Best: {graph_best['pipeline']}, Acc: {graph_best['final_acc']}, Task Score: {graph_best['task_score']}, Dist: {graph_best['total_dist']}")

    # Pareto frontier
    points = results[['total_dist_norm', 'task_score_norm']].values

    # 1. 距離で昇順ソート
    order = np.argsort(points[:, 0])
    points_sorted = points[order]

    # 2. パレートフロンティア抽出（距離↓・スコア↑）
    pareto_mask = []
    best_score = -np.inf
    for d, s in points_sorted:
        if s > best_score + 1e-9:  # 数値誤差よけ
            pareto_mask.append(True)
            best_score = s
        else:
            pareto_mask.append(False)

    pareto_points = points_sorted[np.array(pareto_mask)]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(results['total_dist_norm'], results['task_score_norm'], alpha=0.5, label='Pipelines')
    plt.plot(pareto_points[:, 0], pareto_points[:, 1], 'r-', label='Pareto Frontier')
    plt.scatter(local_best['total_dist_norm'], local_best['task_score_norm'], color='blue', s=100, label='Local Best')
    plt.scatter(graph_best['total_dist_norm'], graph_best['task_score_norm'], color='green', s=100, label='Graph Best')
    plt.xlabel('Normalized Total Distance')
    plt.ylabel('Normalized Task Score')
    plt.title('Task Score vs Distance Trade-off')
    plt.legend()
    plt.savefig('analysis/tradeoff_plot.png')
    plt.show()

if __name__ == "__main__":
    analyze_results("analysis/aggregated_results.csv", "../config.yaml")