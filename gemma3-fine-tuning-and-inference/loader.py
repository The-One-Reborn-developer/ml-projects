import torch

from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig


def load_model_and_processor():
    model_id = "google/gemma-3-12b-pt" # `google/gemma-3-4b/12b/27b-pt`

    model_kwargs = dict(
        attn_implementation="eager", # "flash_attention_2" на новых GPU
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
    '''
    In case Gemma throws error on multi-GPU these lines must be commented out

    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
        bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
    )
    '''
    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained("google/gemma-3-12b-it") # `google/gemma-3-4b/12b/27b-it`
    
    return model, processor