import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from indicnlp.tokenize import indic_tokenize
from indicnlp import common
import os

# ----- Config -----
MODEL_PATH = "/Users/jbc/Documents/punc_restoration/bengali_punctuation_jsonl_data1/Training_outputs/checkpoint-16"
INDIC_RESOURCES_PATH = "/Users/jbc/Documents/punc_restoration/indic_nlp_resources"
MAX_LEN = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")

# Map index → label
LABEL_MAP = {
    0: "O",
    1: "COMMA",
    2: "PERIOD",
    3: "QUESTION",
    4: "EXCLAMATION",
    5: "COLON",
    6: "SEMICOLON",
    7: "HYPHEN"
}

PUNCTUATION_INSERT = {
    "COMMA": ",",
    "PERIOD": "।",
    "QUESTION": "?",
    "EXCLAMATION": "!",
    "COLON": ":",
    "SEMICOLON": ";",
    "HYPHEN": "-"
}
# -------------------

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

# Set Indic NLP path
common.set_resources_path(INDIC_RESOURCES_PATH)


def restore_punctuation(text):
    word_tokens = indic_tokenize.trivial_tokenize(text, lang='bn')

    # Encode
    encoding = tokenizer(word_tokens,
                         is_split_into_words=True,
                         return_tensors="pt",
                         padding=True,
                         truncation=True,
                         max_length=MAX_LEN)

    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        print("predictions:", predictions)

    # Map back to word level
    restored_text = ""
    word_ids = encoding.word_ids()

    previous_word_idx = None
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx == previous_word_idx:
            continue

        word = word_tokens[word_idx]
        label_id = predictions[0][idx].item()
        label = LABEL_MAP[label_id]

        restored_text += word
        if label != "O":
            restored_text += PUNCTUATION_INSERT[label]
        restored_text += " "

        previous_word_idx = word_idx

    return restored_text.strip()


# Example
if __name__ == "__main__":
    text = "প্রধানমন্ত্রী শেখ হাসিনার নেতৃত্বাধীন নতুন মন্ত্রিসভায় ৪৮ জন মন্ত্রী প্রতিমন্ত্রী ও উপমন্ত্রী"
    output = restore_punctuation(text)
    print("🔹 Original:", text)
    print("✅ Restored:", output)
