import torch

from transformers import AutoTokenizer, BertModel


MODEL = BertModel.from_pretrained('bert-base-cased')
TOKENIZER = AutoTokenizer.from_pretrained('bert-base-cased')

input_ids = TOKENIZER(['Hello!', 'Cool.', 'Nice!'])['input_ids']
tensor = torch.tensor(input_ids)

output = MODEL(tensor)
print(output)