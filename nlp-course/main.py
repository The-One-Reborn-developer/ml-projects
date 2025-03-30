from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from pprint import pprint


def tokenize_function(example):
    return tokenizer(example['sentence'], truncation=True)


checkpoint = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer)
raw_dataset = load_dataset("glue", "sst2")

tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)
samples = {key: value for key, value in tokenized_dataset['train'][:8].items() if key not in {'sentence', 'idx'}}

batch = data_collator(samples)
print({key: value for key, value in batch.items()})
