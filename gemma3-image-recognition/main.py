import os
import json

from io import BytesIO
from logging import basicConfig, getLogger, INFO
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import Dataset


load_dotenv()
login(os.getenv('HUGGINGFACE_TOKEN'))

basicConfig(level=INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = getLogger(__name__)

JSONL_PATH = Path('.') / 'gemma3-image-recognition' / 'bank_cards.jsonl'


def load_jsonl_dataset(jsonl_file):
    data = []

    with open(jsonl_file, mode='r', encoding='utf-8') as file:
        for index, line in enumerate(file):
            try:
                content = json.loads(line)
                image_path = content['messages'][1]['content'][0]['image']
                LOGGER.info(f'Processing {index + 1}: {image_path}')

                if not os.path.exists(image_path):
                    LOGGER.warning(f'{image_path} does not exist')
                    continue
                
                image = Image.open(image_path).convert('RGB')
                '''
                with Image.open(image_path) as image:
                    image = image.convert('RGB')
                    image_bytes = BytesIO()
                    image.save(image_bytes, format='JPEG')
                    image_bytes = image_bytes.getvalue()
                '''
                content['messages'][1]['content'][0]['image'] = image

                data.append(content)
            except Exception:
                LOGGER.exception(f'Error processing image {image_path}.')
                continue

    return Dataset.from_list(data[:1]) if data else None


if __name__ == '__main__':
    dataset = load_jsonl_dataset(JSONL_PATH)

    if len(dataset) > 0:
        LOGGER.info(f'Loaded dataset with {len(dataset)} entries.')
        LOGGER.info(f'First entry: {dataset[0]}')
    else:
        LOGGER.error('Dataset is empty or failed to load.')
