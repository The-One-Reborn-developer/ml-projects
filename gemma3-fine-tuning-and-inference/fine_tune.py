import gc
import torch

from pathlib import Path
from PIL import Image
from logging import basicConfig, INFO, getLogger

from huggingface_hub import login
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from loader import load_model_and_processor

torch.set_float32_matmul_precision('high')
basicConfig(level=INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = getLogger(__name__)

login('')
DATASET_PATH = Path('.') / ''
JSONL_PATH = DATASET_PATH / '.jsonl'
MODEL_SAVE_PATH = Path('.')

DATASET = load_dataset("json", data_files=str(JSONL_PATH), split='train')

PEFT_CONFIG = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=[
        "lm_head",
        "embed_tokens",
    ],
)

TRAINING_ARGUMENTS = SFTConfig(
    output_dir=MODEL_SAVE_PATH / "",  # Saved model name
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_torch_fused",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    push_to_hub=False,
    report_to=[],
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
)
TRAINING_ARGUMENTS.remove_unused_columns = False

MODEL, PROCESSOR = load_model_and_processor()


def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    image_inputs = []

    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            content = [content]

        for element in content:
            if isinstance(element, dict) and element.get("type") == "image":
                image_path = element.get('images', None)

                if not image_path:
                    LOGGER.info('Missing image data. Skipping.')

                try:
                    image = Image.open(image_path).convert('RGB')
                    image_inputs.append(image)
                except Exception:
                    LOGGER.exception(f'Error processing image.')
                    continue

    return image_inputs


def collate_fn(examples):
    texts = []
    images = []
    for example in examples:
        image_inputs = process_vision_info(example["messages"])
        text = PROCESSOR.apply_chat_template(
            example["messages"], add_generation_prompt=False, tokenize=False
        )
        texts.append(text.strip())
        images.append(image_inputs)

    batch = PROCESSOR(text=texts, images=images, return_tensors="pt", padding=True)

    labels = batch["input_ids"].clone()

    image_token_id = [
        PROCESSOR.tokenizer.convert_tokens_to_ids(
            PROCESSOR.tokenizer.special_tokens_map["boi_token"]
        )
    ]

    labels[labels == PROCESSOR.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    labels[labels == 262144] = -100

    batch["labels"] = labels
    return batch


if __name__ == '__main__':
    trainer = SFTTrainer(
        model=MODEL,
        args=TRAINING_ARGUMENTS,
        train_dataset=DATASET,
        peft_config=PEFT_CONFIG,
        processing_class=PROCESSOR,
        data_collator=collate_fn
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    trainer.train()
    trainer.save_model()
