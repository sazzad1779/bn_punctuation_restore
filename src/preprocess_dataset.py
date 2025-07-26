import os
import json
from datasets import DatasetDict, Dataset, load_from_disk
from indicnlp.tokenize import indic_tokenize
from indicnlp import common

# ✅ IMPORTANT: Change this path to where you cloned the indic_nlp_resources repo
INDIC_RESOURCES_PATH = "/Users/jbc/Documents/punc_restoration/indic_nlp_resources"

# Ensure indicnlp resources are set
if not os.path.exists(INDIC_RESOURCES_PATH):
    print(f"Warning: INDIC_RESOURCES_PATH '{INDIC_RESOURCES_PATH}' does not exist.")
    print("Please update INDIC_RESOURCES_PATH to the correct location of your indic_nlp_resources directory.")
    # Exit or handle error if resources are critical for execution
    exit("Exiting: indic_nlp_resources not found.")
else:
    common.set_resources_path(INDIC_RESOURCES_PATH)
    print(f"IndicNLP resources set to: {INDIC_RESOURCES_PATH}")


# ------------ Configuration ------------
# Path to your previously saved DatasetDict (e.g., from .save_to_disk())
# This is where your 'content' field dataset should be.
sample_size_per_split = 500
INPUT_DATASET_PATH = "/Users/jbc/Documents/punc_restoration/bangla_newspaper_dataset" # Adjust if your dataset is saved elsewhere

# Directory where the JSONL files for each split will be saved
OUTPUT_JSONL_DIR = "./bengali_punctuation_jsonl_data1"
LANG = "bn"  # Bengali
# ---------------------------------------

# Define supported punctuation tags
PUNCTUATION_MAP = {
    "।": "PERIOD",
    ",": "COMMA",
    "?": "QUESTION",
    "!": "EXCLAMATION",
    ":": "COLON",
    ";": "SEMICOLON",
    "-": "HYPHEN"
}


def tokenize_and_label_batch(examples):
    """
    Tokenizes content and generates word-level tokens and punctuation tags.
    Designed to be used with Dataset.map(batched=True).
    """
    all_words = []
    all_tags = []

    for text in examples["content"]:
        tokens = indic_tokenize.trivial_tokenize(text, lang=LANG)

        words = []
        tags = []

        i = 0
        while i < len(tokens):
            token = tokens[i]

            # If the token itself is a punctuation, skip it (it's handled by previous word's label)
            if token in PUNCTUATION_MAP:
                i += 1
                continue

            # Look ahead to check if next token is punctuation
            if i + 1 < len(tokens) and tokens[i + 1] in PUNCTUATION_MAP:
                label = PUNCTUATION_MAP[tokens[i + 1]]
                i += 2  # Skip current word + punctuation
            else:
                label = "O" # 'O' for 'Other' or no punctuation
                i += 1

            words.append(token)
            tags.append(label)
        
        all_words.append(words)
        all_tags.append(tags)

    return {"tokens": all_words, "tags": all_tags}

def save_to_jsonl(dataset_split, output_filepath):
    """
    Saves a dataset split (which should already contain 'tokens' and 'tags' fields)
    to a JSONL file.
    """
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, "w", encoding="utf-8") as f:
        for entry in dataset_split:
            # Ensure 'tokens' and 'tags' exist and are not empty
            if entry.get("tokens") and entry.get("tags"):
                json_obj = {"tokens": entry["tokens"], "tags": entry["tags"]}
                f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
            else:
                # Optional: Log if an entry has no tokens/tags
                # print(f"Skipping empty or malformed entry: {entry}")
                pass
    print(f"Saved {len(dataset_split)} entries to {output_filepath}")


def main():
    print(f"Attempting to load dataset from: {INPUT_DATASET_PATH}")
    try:
        # Load the DatasetDict
        dataset = load_from_disk(INPUT_DATASET_PATH)
        print("Dataset loaded successfully.")
        print("Original Dataset Structure:")
        print(dataset)
    except Exception as e:
        print(f"Error loading dataset from {INPUT_DATASET_PATH}: {e}")
        print("Please ensure the path is correct and the dataset was saved using .save_to_disk().")
        print("Exiting.")
        return
    
    sampled_dataset = DatasetDict()
    for split_name, split_data in dataset.items():
        if len(split_data) > sample_size_per_split:
            # Use .select() to get a slice of the dataset
            sampled_dataset[split_name] = split_data.select(range(sample_size_per_split))
        else:
            # If the split is smaller than the requested sample size, take all of it
            sampled_dataset[split_name] = split_data
    print("\nProcessing 'content' field and generating tokens and tags...")
    # Apply the tokenization and labeling function to all splits
    # This will add 'tokens' and 'tags' columns to your dataset
    processed_dataset = sampled_dataset.map(
        tokenize_and_label_batch,
        batched=True,
        remove_columns=[col for col in sampled_dataset['train'].column_names if col not in ['content', 'dataset_name']] # Keep content and dataset_name for now, or remove if not needed
    )
    print("Processing complete.")
    print("Processed Dataset Structure (with 'tokens' and 'tags'):")
    print(processed_dataset)
    print("\nExample entry from 'train' split after processing:")
    if 'train' in processed_dataset and len(processed_dataset['train']) > 0:
        print(processed_dataset['train'][0])
    else:
        print("No 'train' split or it's empty.")


    print(f"\nSaving processed data to JSONL files in: {OUTPUT_JSONL_DIR}...")
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_JSONL_DIR, exist_ok=True)

    # Save each split to a separate JSONL file
    for split_name, split_data in processed_dataset.items():
        if split_name == 'valid':
            split_name= 'validation'
        print(f"Processing split: {split_name}")
        output_filepath = os.path.join(OUTPUT_JSONL_DIR, f"{split_name}.jsonl")
        save_to_jsonl(split_data, output_filepath)

    print("\n✅ All splits processed and saved to JSONL!")

if __name__ == "__main__":
    main()