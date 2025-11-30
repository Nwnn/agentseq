from datasets import load_dataset as hf_load_dataset
import pandas as pd

def load_dataset(dataset_name, num_samples=None):
    """
    Load dataset from Hugging Face datasets.

    Args:
        dataset_name (str): "ag_news" or "yelp_polarity"
        num_samples (int): Number of samples to load. If None, load all.

    Returns:
        pd.DataFrame: DataFrame with columns 'text', 'label', 'categories'
        list: List of category names
    """
    if dataset_name == "ag_news":
        dataset = hf_load_dataset("ag_news", split="test")  # Using test split for simplicity
        categories = ['World', 'Sports', 'Business', 'Sci/Tech']
    elif dataset_name == "yelp_polarity":
        dataset = hf_load_dataset("yelp_polarity", split="test")
        categories = ['negative', 'positive']
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    df = pd.DataFrame({
        'text': dataset['text'],
        'label': dataset['label']
    })

    if num_samples:
        df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)

    return df, categories