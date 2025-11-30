import pandas as pd
import numpy as np
from dotenv import load_dotenv
import yaml
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.embedding_calculator import EmbeddingCalculator

load_dotenv()

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def aggregate_logs(log_path, config_path):
    config = load_config(config_path)
    logs = pd.read_csv(log_path)

    step_names = [step['name'] for step in config['steps']]
    agent_ids_per_step = [[agent['id'] for agent in step['agents']] for step in config['steps']]

    # Embedding calculator
    emb_calc = EmbeddingCalculator(config['experiment']['embedding_model'])

    # Compute final accuracies
    final_accs = {}
    for pipeline in logs.groupby([f'{step}_id' for step in step_names]):
        pipeline_ids = pipeline[0]
        df = pipeline[1]
        acc = df['is_correct'].mean()
        final_accs[tuple(pipeline_ids)] = acc

    # Task scores
    task_scores = {}
    for i, step_name in enumerate(step_names):
        for agent_id in agent_ids_per_step[i]:
            # Average over other steps
            others = [s for s in step_names if s != step_name]
            other_ids = [f'{s}_id' for s in others]
            avg_acc = logs[logs[f'{step_name}_id'] == agent_id].groupby(other_ids)['is_correct'].mean().mean()
            task_scores[(step_name, agent_id)] = avg_acc

    # Total task score for pipelines
    pipeline_task_scores = {}
    for pipeline in final_accs:
        score = sum(task_scores[(step_names[i], pipeline[i])] for i in range(len(step_names)))
        pipeline_task_scores[pipeline] = score

    # Embedding distances
    # Collect outputs
    outputs = {}
    for step_name in step_names[:-1]:  # Up to second last
        outputs[step_name] = {}
        for agent_id in agent_ids_per_step[step_names.index(step_name)]:
            texts = logs[logs[f'{step_name}_id'] == agent_id][f'{step_name}_text'].tolist()
            embeddings = emb_calc.compute_embeddings(texts)
            outputs[step_name][agent_id] = emb_calc.compute_mean_embedding(embeddings)

    # For last step input (μ_in for last step)
    last_step_input = {}
    for agent_id in agent_ids_per_step[-1]:  # Last step agents
        # Find best partner from previous step
        best_acc = 0
        best_partner = None
        for partner_id in agent_ids_per_step[-2]:
            pipeline_key = (partner_id, agent_id) if len(step_names) == 2 else (agent_ids_per_step[0][0], partner_id, agent_id)[:len(step_names)]  # Adjust for n steps
            acc = final_accs.get(pipeline_key, 0)
            if acc > best_acc:
                best_acc = acc
                best_partner = partner_id
        # Collect y_{n-1} for that pair
        mask = (logs[f'{step_names[-2]}_id'] == best_partner) & (logs[f'{step_names[-1]}_id'] == agent_id)
        texts = logs[mask][f'{step_names[-2]}_text'].tolist()
        if texts:  # Check if not empty
            embeddings = emb_calc.compute_embeddings(texts)
            last_step_input[agent_id] = emb_calc.compute_mean_embedding(embeddings)
        else:
            last_step_input[agent_id] = np.zeros(emb_calc.model.get_sentence_embedding_dimension())  # Fallback

    # Distances
    distances = {}
    for i in range(len(step_names)-1):
        step1, step2 = step_names[i], step_names[i+1]
        for a1 in agent_ids_per_step[i]:
            for a2 in agent_ids_per_step[i+1]:
                if i == len(step_names)-2:  # Last transition
                    emb1 = outputs[step1][a1]
                    emb2 = last_step_input[a2] if step2 == step_names[-1] else outputs[step2][a2]
                else:
                    emb1 = outputs[step1][a1]
                    emb2 = outputs[step2][a2]
                dist = emb_calc.compute_distance(emb1, emb2)
                distances[(a1, a2)] = dist

    # Total distances for pipelines
    pipeline_distances = {}
    for pipeline in final_accs:
        total_dist = sum(distances[(pipeline[i], pipeline[i+1])] for i in range(len(pipeline)-1))
        pipeline_distances[pipeline] = total_dist

    # Save results
    results = []
    for pipeline in final_accs:
        results.append({
            'pipeline': '_'.join(pipeline),
            'final_acc': final_accs[pipeline],
            'task_score': pipeline_task_scores[pipeline],
            'total_dist': pipeline_distances[pipeline]
        })
    os.makedirs('analysis', exist_ok=True)
    pd.DataFrame(results).to_csv('analysis/aggregated_results.csv', index=False)

if __name__ == "__main__":
    aggregate_logs("../logs/experiment_logs.csv", "../config.yaml")