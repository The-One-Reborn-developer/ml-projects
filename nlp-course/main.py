import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


CHECKPOINT = "distilbert-base-uncased-finetuned-sst-2-english"
TOKENIZER = AutoTokenizer.from_pretrained(CHECKPOINT)
MODEL = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT)


sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]
tokens = TOKENIZER(sequences, padding=True, truncation=True, return_tensors='pt')
output = MODEL(**tokens)

probabilities = torch.softmax(output.logits, dim=1)
predictions = torch.argmax(probabilities, dim=1)
print(predictions)