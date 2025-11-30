import yaml
import os
import pandas as pd
from itertools import product
from dataset_loader import load_dataset
from llm_agent import LLMAgent
from embedding_calculator import EmbeddingCalculator

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_experiment(config_path):
    config = load_config(config_path)

    # Load dataset
    df, categories = load_dataset(config['experiment']['dataset'], config['experiment']['num_samples'])

    # API key
    api_key = os.getenv(config['experiment']['openrouter_api_key_env'])
    if not api_key:
        raise ValueError("API key not found in environment variables")

    # Embedding calculator
    emb_calc = EmbeddingCalculator(config['experiment']['embedding_model'])

    # Create agents for each step
    agents = {}
    for step in config['steps']:
        agents[step['name']] = {}
        for agent_config in step['agents']:
            agents[step['name']][agent_config['id']] = LLMAgent(agent_config['model'], api_key)

    # Generate all pipeline combinations
    step_names = [step['name'] for step in config['steps']]
    agent_ids_per_step = [[agent['id'] for agent in step['agents']] for step in config['steps']]
    pipelines = list(product(*agent_ids_per_step))

    # Logs
    logs = []

    for idx, row in df.iterrows():
        input_text = row['text']
        gold_label = row['label']

        for pipeline in pipelines:
            outputs = {}
            current_input = input_text

            for i, step_name in enumerate(step_names):
                agent_id = pipeline[i]
                agent = agents[step_name][agent_id]
                prompt = config['steps'][i]['agents'][[a['id'] for a in config['steps'][i]['agents']].index(agent_id)]['prompt']

                if step_name == config['steps'][-1]['name']:  # Last step (classify)
                    output = agent.call(prompt, current_input, categories)
                else:
                    output = agent.call(prompt, current_input)

                outputs[step_name] = output
                current_input = output

            # Determine predicted label (simple parsing for now)
            pred_label = parse_label(outputs[config['steps'][-1]['name']], categories)
            is_correct = 1 if pred_label == gold_label else 0

            log_entry = {
                'sample_id': idx,
                **{f'{step}_id': pid for step, pid in zip(step_names, pipeline)},
                **{f'{step}_text': outputs[step] for step in step_names},
                'pred_label': pred_label,
                'gold_label': gold_label,
                'is_correct': is_correct
            }
            logs.append(log_entry)

    # Save logs
    log_df = pd.DataFrame(logs)
    log_df.to_csv('logs/experiment_logs.csv', index=False)

def parse_label(output, categories):
    # Simple parsing: assume the output contains the category name
    for i, cat in enumerate(categories):
        if cat.lower() in output.lower():
            return i
    return -1  # Unknown

if __name__ == "__main__":
    run_experiment("../config.yaml")