from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, Trainer
from pprint import pprint


def tokenize_function(example):
    return tokenizer(example['sentence1'], example['sentence2'], truncation=True)


checkpoint = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer)
raw_dataset = load_dataset("glue", "mrpc")
tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)

training_arguments = TrainingArguments()

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_arguments,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    data_collator=data_collator,
    tokenizer=tokenizer
)
trainer.train()