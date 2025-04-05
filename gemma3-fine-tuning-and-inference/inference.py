import torch
import time

from argparse import ArgumentParser
from logging import basicConfig, INFO, getLogger
from PIL import Image
from pathlib import Path
from transformers import AutoModelForImageTextToText, AutoProcessor

from prompts import SYSTEM_PROMPT


TEST_DATASET_PATH = Path('.') / 'test'

basicConfig(level=INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = getLogger(__name__)
MODEL = AutoModelForImageTextToText.from_pretrained(
    Path('.') / "gemma3-bank-card-recognition-merged",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
)
PROCESSOR = AutoProcessor.from_pretrained(Path('.') / "gemma3-bank-card-recognition-merged")


def run_inference(sample, MODEL, PROCESSOR):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract the data from this bank card image"},
                {"type": "image", "image": Image.open(sample)}
            ]
        },
    ]
    
    inputs = PROCESSOR.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = PROCESSOR(
        text=[inputs],
        images=[Image.open(sample)],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(MODEL.device)
    
    stop_token_ids = [PROCESSOR.tokenizer.eos_token_id, PROCESSOR.tokenizer.convert_tokens_to_ids("<end_of_turn>")]
    generated_ids = MODEL.generate(**inputs, max_new_tokens=256, top_p=1.0, do_sample=True, temperature=0.8, eos_token_id=stop_token_ids, disable_compile=True)

    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = PROCESSOR.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]


if __name__ == '__main__':
    #parser = ArgumentParser(description='Run the inference on the bank card image.')
    #parser.add_argument('sample', type=str, help='Path to the bank card image')
    #args = parser.parse_args()

    #sample = Path(args.sample)
    first = True
    timings = []
    for image in TEST_DATASET_PATH.iterdir():
        if first:
            result = run_inference(image, MODEL, PROCESSOR)
            LOGGER.info(result)
            first = False
        else:
            start = time.time()
            result = run_inference(image, MODEL, PROCESSOR)
            end = time.time()
            elapsed = end - start
            timings.append(elapsed)
            LOGGER.info(f"{image.name} (time: {elapsed:.2f}s): {result}")

    if timings:
        LOGGER.info(f'Summary time of processing {len(timings)}: {sum(timings)} seconds.')
