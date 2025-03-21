from datasets import load_dataset


raw_dataset = load_dataset("glue", "mrpc")
train_raw_dataset = raw_dataset['train']
print(train_raw_dataset.features)