import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import sys
sys.path.insert(0, '../src')

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def analyze_agent_scores(logs_path, config_path):
    # Load logs and config
    logs = pd.read_csv(logs_path)
    config = load_config(config_path)

    # Group by pipeline
    pipelines = logs.groupby(['extract_id', 'summarize_id', 'classify_id'])

    # Calculate TaskScore for each pipeline
    task_scores = {}
    for pipeline, group in pipelines:
        acc = group['is_correct'].mean()
        task_scores[pipeline] = acc

    # Calculate TaskScore for each agent
    agent_scores = {'extract': {}, 'summarize': {}, 'classify': {}}
    for step_idx, step in enumerate(['extract', 'summarize', 'classify']):
        agent_ids = [agent["id"] for agent in config['steps'][step_idx]['agents']]
        for agent_id in agent_ids:
            scores = []
            for pipeline, score in task_scores.items():
                if pipeline[step_idx] == agent_id:
                    scores.append(score)
            agent_scores[step][agent_id] = scores

    # Plot distributions as histograms for E3 only
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    step = 'classify'
    agents_list = list(agent_scores[step].keys())
    colors = ['blue', 'green', 'red']  # Colors for A, B, C
    for j, agent in enumerate(agents_list):
        scores = agent_scores[step][agent]
        if scores:
            ax.hist(scores, bins=10, alpha=0.5, color=colors[j], label=agent, edgecolor='black')
    ax.set_title('E3 Agent Task Scores')
    ax.set_xlabel('Task Score (Accuracy)')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.savefig('analysis/e3_agent_scores_histogram.png')
    plt.show()

    # Print statistics for E3 only
    print(f"\nClassify Agents (E3):")
    for agent, scores in agent_scores['classify'].items():
        if scores:
            print(f"  {agent}: Mean={np.mean(scores):.4f}, Std={np.std(scores):.4f}, Min={np.min(scores):.4f}, Max={np.max(scores):.4f}")
        else:
            print(f"  {agent}: No data")

if __name__ == "__main__":
    analyze_agent_scores("../logs/experiment_logs.csv", "../config.yaml")