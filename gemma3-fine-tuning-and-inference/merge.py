from pathlib import Path
from peft import PeftModel

from transformers import AutoModelForImageTextToText, AutoProcessor


MODEL_ID = "google/gemma-3-4b-pt" # или `google/gemma-3-12b-pt`, `google/gemma-3-27-pt`
MODEL = AutoModelForImageTextToText.from_pretrained(MODEL_ID, low_cpu_mem_usage=True)
PEFT_MODEL = PeftModel.from_pretrained(MODEL, Path('.') / "gemma3-bank-card-recognition")

MERGED_MODEL = PEFT_MODEL.merge_and_unload()
MERGED_MODEL.save_pretrained("gemma3-bank-card-recognition-merged", safe_serialization=True, max_shard_size='2GB')

PROCESSOR = AutoProcessor.from_pretrained(Path('.') / "gemma3-bank-card-recognition")
PROCESSOR.save_pretrained("gemma3-bank-card-recognition-merged")
