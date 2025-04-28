from pathlib import Path
from peft import PeftModel

from transformers import AutoModelForImageTextToText, AutoProcessor


MODEL_ID = "google/gemma-3-12b-pt" # `google/gemma-3-4b/12b/27b-pt`
MODEL = AutoModelForImageTextToText.from_pretrained(MODEL_ID, low_cpu_mem_usage=True)
PEFT_MODEL = PeftModel.from_pretrained(MODEL, Path('.') / "")  # Saved model path

MERGED_MODEL = PEFT_MODEL.merge_and_unload()
MERGED_MODEL.save_pretrained("", safe_serialization=True, max_shard_size='2GB')  # Merged model path

PROCESSOR = AutoProcessor.from_pretrained(Path('.') / "")  # Saved model path
PROCESSOR.save_pretrained("")  # Merged model path
