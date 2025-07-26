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
final_dataset.save_to_disk("bangla_newspaper_dataset")
