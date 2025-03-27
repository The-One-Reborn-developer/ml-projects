import os
import json
import base64

from ollama import chat
from logging import basicConfig, getLogger, INFO
from pathlib import Path
from dotenv import load_dotenv
from prompts import SYSTEM_PROMPT


load_dotenv()

basicConfig(level=INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = getLogger(__name__)

JSONL_PATH = Path('.') / 'gemma3-image-recognition' / 'bank_cards.jsonl'
ARCHITECTURE = 'gemma3:4b'


def base64_encode(image_path: str) -> str:
    with open(image_path, mode='rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def load_jsonl_dataset(jsonl_file):
    data = []

    with open(jsonl_file, mode='r', encoding='utf-8') as file:
        for index, line in enumerate(file):
            try:
                content = json.loads(line)
                image_path = content[1]['images']
                LOGGER.info(f'Processing {index + 1}: {image_path}')

                if not os.path.exists(image_path):
                    LOGGER.warning(f'{image_path} does not exist')
                    continue
                
                encoded_image = base64_encode(image_path)
                content[1]['images'] = [encoded_image]

                data.append(content)
            except Exception:
                LOGGER.exception(f'Error processing image {image_path}.')
                continue

    return data


if __name__ == '__main__':
    dataset = load_jsonl_dataset(JSONL_PATH)

    if len(dataset) > 0:
        LOGGER.info(f'Loaded dataset with {len(dataset)} entries.')
        for entry in dataset:
            formatted_json = f"```json\n{json.dumps(entry[2]['content'], indent=4, ensure_ascii=False)}\n```"
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "images": entry[1]['images']},
                {"role": "assistant", "content": formatted_json}
            ]

            response = chat(model=ARCHITECTURE, messages=messages)
            LOGGER.info(response)
    else:
        LOGGER.error('Dataset is empty or failed to load.')
