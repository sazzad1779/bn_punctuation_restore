# prepare_bengali_punctuation_data.py

import os
import pandas as pd
from indicnlp.tokenize import indic_tokenize
from indicnlp import common
from indicnlp.tokenize import indic_tokenize
import json

# ✅ Change this path to where you cloned the repo
INDIC_RESOURCES_PATH = "/Users/jbc/Documents/punc_restoration/indic_nlp_resources"

# common.set_resources_path(INDIC_RESOURCES_PATH)
# text = "আমি ভাত খাই। তুমি কি করো?"

# tokens = indic_tokenize.trivial_tokenize(text, lang='bn')

# with open("output.txt", "w", encoding="utf-8") as f:
#     for item in tokens:
#         f.write(f"{item}\n")


# ------------ Configuration ------------
RAW_DATA_FOLDER = "raw_data"
OUTPUT_CSV = "bengali_punctuation_data.csv"
OUTPUT_JSONL = "bengali_punctuation_data.jsonl"
LANG = "bn"  # Bengali
# ---------------------------------------

# Define supported punctuation labels
PUNCTUATION_MAP = {
    "।": "PERIOD",
    ",": "COMMA",
    "?": "QUESTION",
    "!": "EXCLAMATION",
    ":": "COLON",
    ";": "SEMICOLON",
    "-": "HYPHEN"
}


def load_raw_texts(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                texts.append(file.read())
    return texts


def tokenize_and_label(text):
    tokens = indic_tokenize.trivial_tokenize(text, lang=LANG)
    
    words = []
    labels = []

    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        # If the token itself is a punctuation, skip it
        if token in PUNCTUATION_MAP:
            i += 1
            continue
        
        # Look ahead to check if next token is punctuation
        if i + 1 < len(tokens) and tokens[i + 1] in PUNCTUATION_MAP:
            label = PUNCTUATION_MAP[tokens[i + 1]]
            i += 2  # Skip token + punctuation
        else:
            label = "O"
            i += 1
        
        words.append(token)
        labels.append(label)

    return words, labels


def create_dataset(texts):
    all_data = []

    for text in texts:
        tokens, labels = tokenize_and_label(text)
        for token, label in zip(tokens, labels):
            all_data.append({"token": token, "label": label})

    return pd.DataFrame(all_data)

def create_jsonl(texts, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for text in texts:
            tokens, labels = tokenize_and_label(text)
            if tokens:  # skip empty sentences
                json_obj = {"tokens": tokens, "labels": labels}
                f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

def main():
    print(f"Loading texts from {RAW_DATA_FOLDER}...")
    raw_texts = load_raw_texts(RAW_DATA_FOLDER)

    print("Processing and labeling...")
    df = create_dataset(raw_texts)

    print(f"Saving to {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False)
    print("✅ Done!")

    print("🔄 Tokenizing and labeling...")
    create_jsonl(raw_texts, OUTPUT_JSONL)



if __name__ == "__main__":
    main()
