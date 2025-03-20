import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification


CHECKPOINT = "distilbert-base-uncased-finetuned-sst-2-english"
TOKENIZER = AutoTokenizer.from_pretrained(CHECKPOINT)
MODEL = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT)


first_sequence = "I've been waiting for a HuggingFace course my whole life."
second_sequence = "I hate this so much!"
first_ids = TOKENIZER.convert_tokens_to_ids(TOKENIZER.tokenize(first_sequence))
second_ids = TOKENIZER.convert_tokens_to_ids(TOKENIZER.tokenize(second_sequence))
for i in range(max(len(first_ids), len(second_ids)) - min(len(first_ids), len(second_ids))):
    second_ids.append(TOKENIZER.pad_token_id)
first_attention_mask = [1 if id != 0 else 0 for id in first_ids]
second_attention_mask = [1 if id != 0 else 0 for id in second_ids]
batched_ids = []
batched_ids.append(first_ids)
batched_ids.append(second_ids)
batched_attention_masks = []
batched_attention_masks.append(first_attention_mask)
batched_attention_masks.append(second_attention_mask)
output = MODEL(torch.tensor(batched_ids), attention_mask=torch.tensor(batched_attention_masks))
