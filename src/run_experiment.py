import yaml
import os
import pandas as pd
from itertools import product
from dotenv import load_dotenv
from dataset_loader import load_dataset
from llm_agent import LLMAgent
from embedding_calculator import EmbeddingCalculator

load_dotenv()

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_experiment(config_path):
    config = load_config(config_path)

    print("Loading dataset...")
    # Load dataset
    df, categories = load_dataset(config['experiment']['dataset'], config['experiment']['num_samples'])
    print(f"Loaded {len(df)} samples from {config['experiment']['dataset']}")

    # API key
    api_key = os.getenv(config['experiment']['openrouter_api_key_env'])
    if not api_key:
        raise ValueError("API key not found in environment variables")

    print("Initializing embedding calculator...")
    # Embedding calculator
    emb_calc = EmbeddingCalculator(config['experiment']['embedding_model'])

    print("Creating LLM agents...")
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
    print(f"Generated {len(pipelines)} pipeline combinations")

    # Logs path
    logs_path = '../logs/experiment_logs.csv'

    # Load existing logs if any
    processed = set()
    if os.path.exists(logs_path):
        existing_logs = pd.read_csv(logs_path)
        processed = set(zip(existing_logs['sample_id'], existing_logs['extract_id'], existing_logs['summarize_id'], existing_logs['classify_id']))
        print(f"Resuming from existing logs: {len(processed)} entries already processed")
    else:
        print("Starting new experiment")

    # Logs
    logs = []
    total_samples = len(df)
    total_pipelines = len(pipelines)

    print(f"Starting experiment with {total_samples} samples and {total_pipelines} pipelines...")

    for idx, row in df.iterrows():
        if idx % 50 == 0 or idx == 0:
            print(f"Processing sample {idx+1}/{total_samples}")
        input_text = row['text']
        gold_label = row['label']

        for pipeline_idx, pipeline in enumerate(pipelines):
            pipeline_key = (idx, *pipeline)
            if pipeline_key in processed:
                continue  # Skip already processed

            if pipeline_idx % 5 == 0 or pipeline_idx == 0:
                print(f"  Processing pipeline {pipeline_idx+1}/{total_pipelines} for sample {idx+1}")
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

            # Save incrementally
            pd.DataFrame([log_entry]).to_csv(logs_path, mode='a', header=not os.path.exists(logs_path), index=False)
            processed.add(pipeline_key)

    print(f"Experiment completed. Logs saved to {logs_path}")
    print(f"Total new log entries: {len(logs)}")

def parse_label(output, categories):
    # Simple parsing: assume the output contains the category name
    for i, cat in enumerate(categories):
        if cat.lower() in output.lower():
            return i
    return -1  # Unknown

if __name__ == "__main__":
    run_experiment("../config.yaml")