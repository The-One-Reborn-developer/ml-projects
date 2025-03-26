import os
import json
import ollama
import base64

from logging import basicConfig, getLogger, INFO
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()

basicConfig(level=INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = getLogger(__name__)

JSONL_PATH = Path('.') / 'gemma3-image-recognition' / 'bank_cards.jsonl'
ARCHITECTURE = 'gemma3:12b'


def base64_encode(image_path: str) -> str:
    with open(image_path, mode='rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def load_jsonl_dataset(jsonl_file):
    data = []

    with open(jsonl_file, mode='r', encoding='utf-8') as file:
        for index, line in enumerate(file):
            try:
                content = json.loads(line)
                image_path = content['messages'][1]['images']
                LOGGER.info(f'Processing {index + 1}: {image_path}')

                if not os.path.exists(image_path):
                    LOGGER.warning(f'{image_path} does not exist')
                    continue
                
                encoded_image = base64_encode(image_path)
                LOGGER.info(f"Encoded image (truncated): {encoded_image[:50]}...")
                content['messages'][1]['images'] = [encoded_image]

                data.append(content['messages'])
            except Exception:
                LOGGER.exception(f'Error processing image {image_path}.')
                continue

    return data[:1]


if __name__ == '__main__':
    dataset = load_jsonl_dataset(JSONL_PATH)

    if len(dataset) > 0:
        LOGGER.info(f'Loaded dataset with {len(dataset)} entries.')
        response = ollama.chat(model=ARCHITECTURE, messages=dataset[0])
        LOGGER.info(f'Gemma 3:12b response:\n{response}')
    else:
        LOGGER.error('Dataset is empty or failed to load.')
