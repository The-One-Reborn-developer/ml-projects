from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from pprint import pprint


def tokenize_function(example):
    return tokenizer(example['sentence1'], example['sentence2'], truncation=True)


checkpoint = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer)
raw_dataset = load_dataset("glue", "mrpc")

tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)
samples = {key: value for key, value in tokenized_dataset['train'][:8].items() if key not in {'sentence1', 'sentence2', 'idx'}}
print([len(sample) for sample in samples['input_ids']])

batch = data_collator(samples)
print({key: value.shape for key, value in batch.items()})