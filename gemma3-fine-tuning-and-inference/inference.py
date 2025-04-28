import torch
import time
import json
import re

from argparse import ArgumentParser
from logging import basicConfig, INFO, getLogger
from PIL import Image
from pathlib import Path
from transformers import AutoModelForImageTextToText, AutoProcessor


# Logging setup
basicConfig(level=INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = getLogger(__name__)

# Load model and processor
MODEL = AutoModelForImageTextToText.from_pretrained(
    Path('.') / "",  # Merged model path
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
)
PROCESSOR = AutoProcessor.from_pretrained(Path('.') / "")  # Merged model path

def try_fix_json(text):
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)
    if text.count("{") > text.count("}"):
        text += "}"
    return text

def run_inference(image_path):
    messages = [
        {"role": "system", "content": ""},  # Put same prompt here as for fine tuning
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract the data from this image"},
                {"type": "image", "image": Image.open(image_path)}
            ]
        },
    ]

    inputs = PROCESSOR.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = PROCESSOR(
        text=[inputs],
        images=[Image.open(image_path)],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(MODEL.device)

    stop_token_ids = [
        PROCESSOR.tokenizer.eos_token_id,
        PROCESSOR.tokenizer.convert_tokens_to_ids("<end_of_turn>")
    ]

    generated_ids = MODEL.generate(
        **inputs,
        max_new_tokens=512,
        top_p=0.9,
        do_sample=True,
        temperature=0.15,
        eos_token_id=stop_token_ids,
        disable_compile=True,
    )

    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = PROCESSOR.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    try:
        LOGGER.info(output_text)
        return json.loads(output_text)
    except json.JSONDecodeError:
        fixed_text = try_fix_json(output_text)
        try:
            return json.loads(fixed_text)
        except json.JSONDecodeError:
            LOGGER.warning(f"Failed to parse model output:\n{output_text}")
            return {}

def evaluate_prediction(pred, true):
    correct = 0
    total = 0

    for key, true_val in true.items():
        if key not in pred:
            LOGGER.debug(f"Missing key in prediction: {key}")
            continue

        pred_val = pred[key]

        if isinstance(true_val, list) and isinstance(pred_val, list):
            if pred_val == true_val:
                correct += 1
        elif pred_val == true_val:
            correct += 1
        total += 1

    return correct, total

def evaluate_on_directory(directory: Path):
    images = sorted(directory.glob("*.jpg"))
    if not images:
        LOGGER.error("No images found in the directory.")
        return

    total_correct = 0
    total_fields = 0
    total_inference_time = 0.0

    for i, image_path in enumerate(images):
        json_path = image_path.with_suffix(".json")
        if not json_path.exists():
            LOGGER.warning(f"Missing JSON for image: {image_path.name}")
            continue

        if i >= 1:
            start_time = time.time()
            pred = run_inference(image_path)
            elapsed = time.time() - start_time
            total_inference_time += elapsed
        else:
            pred = run_inference(image_path)

        with open(json_path, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)

        correct, total = evaluate_prediction(pred, ground_truth)
        total_correct += correct
        total_fields += total

        LOGGER.info(f"[{i+1}/{len(images)}] {image_path.name} - Accuracy: {correct}/{total} ({correct/total if total > 0 else 0:.2%})")

    LOGGER.info(f"Total accuracy: {total_correct}/{total_fields} ({total_correct/total_fields:.2%})")
    if len(images) > 1:
        LOGGER.info(f"Total inference time (excluding first image): {total_inference_time:.2f} seconds")
        LOGGER.info(f"Avg inference time per image: {total_inference_time / (len(images) - 1):.2f} seconds")

def evaluate_single_image(image_path: Path):
    json_path = image_path.with_suffix(".json")
    if not json_path.exists():
        LOGGER.error(f"JSON file not found for image: {image_path}")
        return

    start_time = time.time()
    pred = run_inference(image_path)
    elapsed = time.time() - start_time

    with open(json_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    correct, total = evaluate_prediction(pred, ground_truth)

    LOGGER.info(f"{image_path.name} - Accuracy: {correct}/{total} ({correct/total if total > 0 else 0:.2%})")
    LOGGER.info(f"Inference time: {elapsed:.2f} seconds")

if __name__ == '__main__':
    parser = ArgumentParser(description='Run inference on validation data or a single image.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--val_dir', type=str, help='Path to the validation directory')
    group.add_argument('--image', type=str, help='Path to a single image')

    args = parser.parse_args()

    if args.image:
        evaluate_single_image(Path(args.image))
    else:
        evaluate_on_directory(Path(args.val_dir))
