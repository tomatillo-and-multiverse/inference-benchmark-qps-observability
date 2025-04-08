import argparse
import json
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import subprocess


def create_pretty_conversation_json(prompts, responses, output_filename="conversations.json"):
    """
    Creates a pretty-printed JSON file from two lists of prompts and responses.
    
    Args:
        prompts (list): List of prompt strings
        responses (list): List of response strings
        output_filename (str): Name of the output JSON file
        
    Returns:
        None
    """
    # Ensure both lists have the same length
    if len(prompts) != len(responses):
        raise ValueError("Prompts and responses lists must have the same length")
    
    # Create the conversation objects
    conversation_objects = []
    
    for i in range(len(prompts)):
        conversation_object = {
            "conversations": [
                {
                    "from": "human",
                    "value": prompts[i]
                },
                {
                    "from": "gpt",
                    "value": responses[i]
                }
            ]
        }
        conversation_objects.append(conversation_object)
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(conversation_objects, f, indent=4, ensure_ascii=False)
  
def calculate_token_stats(dataset, filename, tokenizer_name, max_samples=None, input_column='instruction', output_column = 'generation', count_tokens=False, input_text_prefix = None):
    """
    Calculate token statistics for a dataset's input and output columns.

    Args:
        dataset: Hugging Face dataset
        input_column: Column name containing input text
        output_column: Column name containing output text
        tokenizer_name: Name of the tokenizer to use (e.g., 'gpt2', 'facebook/bart-large')
        max_samples: Maximum number of samples to process (None for all)

    Returns:
        DataFrame with token statistics
    """
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Limit sample size if specified
    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    # Ensure columns exist
    #if input_column not in dataset.column_names:
     #   raise ValueError(f"Input column '{input_column}' not found in dataset. Available columns: {dataset.column_names}")
    #if output_column not in dataset.column_names:
     #   raise ValueError(f"Output column '{output_column}' not found in dataset. Available columns: {dataset.column_names}")

    input_token_counts = []
    output_token_counts = []
    prompts = []
    responses = []
    results, histogram_data, js = None, None, None
    # Process each example
    for example in tqdm(dataset, desc="sampling examples", unit="example"):
      if 'conversations' in example:
        if len(example['conversations']) < 2:
          continue

        input_text =  example['conversations'][0]['value'] if input_text_prefix is None else input_text_prefix + example['conversations'][0]['value']
        output_text = example['conversations'][1]['value']
        #print(input_text)
        #print(output_text)
      elif 'messages' in example:
        if len(example['messages']) < 2:
          continue

        input_text = example['messages'][0]['content'] if input_text_prefix is None else input_text_prefix + example['messages'][0]['content']
        output_text = example['messages'][1]['content']
        #print(input_text)
        #print(output_text)
      else:
        input_text = example[input_column] if input_text_prefix is None else input_text_prefix + example[input_column]
        output_text = example[output_column]
        # Skip examples with missing text
        if not input_text or not output_text:
            continue

        # Count tokens
      if count_tokens:
        input_tokens = tokenizer(input_text, truncation=False, padding=False)
        output_tokens = tokenizer(output_text, truncation=False, padding=False)
        input_token_counts.append(len(input_tokens["input_ids"]))
        output_token_counts.append(len(output_tokens["input_ids"]))
        input_mean = np.mean(input_token_counts)
        input_std = np.std(input_token_counts)
        input_median = np.median(input_token_counts)
        input_min = np.min(input_token_counts)
        input_max = np.max(input_token_counts)

        output_mean = np.mean(output_token_counts)
        output_std = np.std(output_token_counts)
        output_median = np.median(output_token_counts)
        output_min = np.min(output_token_counts)
        output_max = np.max(output_token_counts)

        total_mean = np.mean([i + o for i, o in zip(input_token_counts, output_token_counts)])
        total_std = np.std([i + o for i, o in zip(input_token_counts, output_token_counts)])

        # Create results dataframe
        results = pd.DataFrame({
        "Metric": ["Mean", "Std Dev", "Median", "Min", "Max"],
        "Input Tokens": [input_mean, input_std, input_median, input_min, input_max],
        "Output Tokens": [output_mean, output_std, output_median, output_min, output_max],
        "Total Tokens": [total_mean, total_std, np.median([i + o for i, o in zip(input_token_counts, output_token_counts)]),
                        np.min([i + o for i, o in zip(input_token_counts, output_token_counts)]),
                        np.max([i + o for i, o in zip(input_token_counts, output_token_counts)])]
    })

        # Create histogram data
        histogram_data = {
        "input_tokens": input_token_counts,
        "output_tokens": output_token_counts,
        "total_tokens": [i + o for i, o in zip(input_token_counts, output_token_counts)]
    }

      prompts.append(input_text)
      responses.append(output_text)
 
    create_pretty_conversation_json(prompts, responses, filename)


    return  results, histogram_data

def plot_token_distributions(histogram_data, output_file="token_distributions.png"):
    """Generate histograms for token distributions"""
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # Plot input tokens
    axes[0].hist(histogram_data["input_tokens"], bins=50, alpha=0.7)
    axes[0].set_title("Input Token Distribution")
    axes[0].set_xlabel("Number of Tokens")
    axes[0].set_ylabel("Frequency")

    # Plot output tokens
    axes[1].hist(histogram_data["output_tokens"], bins=50, alpha=0.7)
    axes[1].set_title("Output Token Distribution")
    axes[1].set_xlabel("Number of Tokens")
    axes[1].set_ylabel("Frequency")

    # Plot total tokens
    axes[2].hist(histogram_data["total_tokens"], bins=50, alpha=0.7)
    axes[2].set_title("Total Token Distribution")
    axes[2].set_xlabel("Number of Tokens")
    axes[2].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Token distribution plot saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze token statistics from a Hugging Face dataset")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Tokenizer to use for counting tokens")
    parser.add_argument("--max_samples", type=int, default=90000, help="Maximum number of samples to process")
    parser.add_argument("--hf_token", type=str, help="Hugging Face API token for authentication")
    parser.add_argument("--count_tokens", default=False, action="store_true", help="Count tokens in the dataset")
    
    #datasets_config = [("FiscalNote/billsum", "default"), ("BAAI/Infinity-Instruct", "Gen"), ("chargoddard/WebInstructSub-prometheus", "default"), ("arcee-ai/The-Tome", "default"), ("mlabonne/open-perfectblend","default"), ("HuggingFaceTB/smoltalk","all"), ("mlabonne/orca-agentinstruct-1M-v1-cleaned","default"), ("allenai/tulu-3-sft-mixture" ,"default")]
    datasets_config = [("FiscalNote/billsum", "default", "Summarize text: ", "text", "summary")]
    args = parser.parse_args()
    
    # Login to Hugging Face if token is provided
    if args.hf_token:
        print("Logging in to Hugging Face...")
        try:
            # Use the token to log in
            subprocess.run(f"echo {args.hf_token} | huggingface-cli login", shell=True, check=True)
            print("Successfully logged in to Hugging Face")
        except subprocess.CalledProcessError as e:
            print(f"Error logging in to Hugging Face: {e}")
            return
    else:
        print("No Hugging Face token provided. Some datasets may not be accessible.")
    
    for dataset_name, config, input_text_prefix, input_column, output_column in datasets_config:
        print(f"Loading dataset: {dataset_name}")
        try:
            dataset = load_dataset(dataset_name, config, )
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Make sure you have installed the 'huggingface_hub' package and have proper authentication.")
            print("Run: huggingface-cli login")
            continue  # Try next dataset instead of returning

        print(f"Dataset loaded, containing {len(dataset)} examples")
        print(f"Available columns: {dataset.column_names}")

        # Calculate token statistics
        try:
            results, histogram_data = calculate_token_stats(
            dataset['train'],
            f"{dataset_name.split('/')[1]}_conversations.json",
            args.tokenizer,
            args.max_samples,
            input_column=input_column,
            output_column=output_column, count_tokens=args.count_tokens, input_text_prefix=input_text_prefix
        )
            
            # Print results
        except Exception as e:
            print(f"Error during token statistics calculation: {e}")
            continue  # Try next dataset

if __name__ == "__main__":
    main()