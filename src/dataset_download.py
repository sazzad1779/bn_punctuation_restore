from datasets import DatasetDict, concatenate_datasets, load_dataset
from transformers import AutoTokenizer

ds = load_dataset("zabir-nabil/bangla_newspaper_dataset")

def simplify_dataset(ds_split, dataset_name):
    ds = ds_split.remove_columns([c for c in ds_split.column_names if c != "content"])
    return ds.add_column("dataset_name", [dataset_name] * len(ds_split))

dataset_name = "zabir-nabil/bangla_newspaper_dataset"
train_ds = simplify_dataset(ds["train"], dataset_name)
valid_ds = simplify_dataset(ds["valid"], dataset_name)
test_1_ds = simplify_dataset(ds["test_1"], dataset_name)
test_2_ds = simplify_dataset(ds["test_2"], dataset_name)
test_ds = concatenate_datasets([test_1_ds, test_2_ds])
final_dataset = DatasetDict({"train": train_ds, "valid": valid_ds, "test": test_ds})

# ————————————————————————————————————————————————————————————————

max_length = 256  # maximum number of tokens per chunk
stride = 10      # overlap between chunks (optional, common for BERT-style tasks)

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def chunk_content(example):
    output = tokenizer(
        example["content"],
        truncation=False,
        return_overflowing_tokens=True,
        max_length=max_length,
        stride=stride,
    )
    chunks = []
    for i, token_ids in enumerate(output["input_ids"]):
        text_chunk = tokenizer.decode(
            token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        chunks.append({
            "dataset_name": example["dataset_name"],
            "content": text_chunk,
            "chunk_id": i,
            "original_id": example.get("__index_level_0__", None),
        })
    return chunks
def chunk_examples(examples):
    all_content = []
    all_dataset_name = []

    for content, dataset_name in zip(examples["content"], examples["dataset_name"]):
        tokens = tokenizer.encode(content, add_special_tokens=False)
        chunks = [tokens[i:i + MAX_LEN] for i in range(0, len(tokens), MAX_LEN)]
        texts = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

        all_content.extend(texts)
        all_dataset_name.extend([dataset_name] * len(texts))

    return {"content": all_content, "dataset_name": all_dataset_name}

# Apply chunking to each split
for split in ["train", "valid", "test"]:
    ds_split = final_dataset[split]
    final_dataset[split] = ds_split.map(
        chunk_content,
        batched=False,
        remove_columns=ds_split.column_names
    ).flatten_indices()
