import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


model = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForSequenceClassification.from_pretrained(model)

inputs = [
    "This guy tweaking",
    "I'll have two number nines, a number nine large, a number six with extra dip, two number fourty fives, one with cheese, and a large soda.",
    "Dat's good"
]
tokenized_inputs = tokenizer(inputs, truncation=True, padding=True, return_tensors='pt')

outputs = model(**tokenized_inputs)

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)

classification = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
print(classification(inputs))
