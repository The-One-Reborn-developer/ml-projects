import json
import os

from logging import basicConfig, INFO, getLogger
from pathlib import Path


basicConfig(level=INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = getLogger(__name__)

DATASET_PATH = Path('.') / ''
IMAGE_URLS_FILE = DATASET_PATH / 'urls.txt'
OUTPUT_FILE = DATASET_PATH / '.jsonl'


def format_data(sample):
    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": sample["system_prompt"]}]},
            {"role": "user", "content": [{"type": "image", "images": sample["image"]}]},
            {"role": "assistant", "content": [{"type": "text", "text": sample["json_data"]}]},
        ],
    }


def create_jsonl(image_urls_file, data_dir, output_file):
    try:
        LOGGER.info('Reading urls file...')
        with open(image_urls_file, 'r') as f:
            image_urls = [line.strip() for line in f]
        
        LOGGER.info('Writing to output file...')
        with open(output_file, 'w') as outfile:
            for i, _ in enumerate(image_urls, start=1):
                img_json_file = os.path.join(data_dir, f'{i:d}.json')

                if os.path.exists(img_json_file):
                    with open(img_json_file, 'r') as json_file:
                        img_json = json.load(json_file)
                        formatted_json = json.dumps(img_json, indent=4, ensure_ascii=False)
                else:
                    LOGGER.warning(f"File {img_json_file} not found, skipping.")
                    continue

                image_path = os.path.join(data_dir, f'{i}.jpg')

                sample = {
                    "system_prompt": '',  # Put a prompt here
                    "image": image_path,
                    "json_data": formatted_json,
                }

                formatted_record = format_data(sample)
                outfile.write(json.dumps(formatted_record, ensure_ascii=False) + '\n')
    except Exception:
        LOGGER.exception('Failed to create jsonl file.')


if __name__ == '__main__':
    LOGGER.info('Started creation of jsonl file...')
    create_jsonl(IMAGE_URLS_FILE, DATASET_PATH, OUTPUT_FILE)
