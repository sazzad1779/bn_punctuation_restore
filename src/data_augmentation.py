from textattack.transformations import WordSwapRandomCharacterDeletion
from textattack.augmentation import Augmenter
import json

def load_sentences(filename):
    sentences = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            sentence = " ".join(data["tokens"])
            sentences.append(sentence)
    return sentences
augmenter = Augmenter(
    transformation=WordSwapRandomCharacterDeletion(),
    pct_words_to_swap=0.1,
    transformations_per_example=2
)


# Load Bengali sentences
bengali_sentences = load_sentences("/Users/jbc/Documents/punc_restoration/bengali_punctuation_jsonl_data/test.jsonl")
sentences = bengali_sentences[:5]
# Augment
output_file = "augmented_sentences.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for sentence in sentences:
        augmented_versions = augmenter.augment(sentence)  # expects list of str
        f.write(f"Original: {sentence}\n")
        for aug in augmented_versions:
            f.write(f"Augmented: {aug}\n")
        f.write("\n")  # blank line between different sentences

print(f"Augmented sentences written to {output_file}")
