import os
import json
import argparse

from prompts import SYSTEM_PROMPT, USER_PROMPT


def create_jsonl(image_urls_file, data_dir, output_file):
    with open(image_urls_file, 'r') as f:
        image_urls = [line.strip() for line in f]

    with open(output_file, 'w') as outfile:
        for i, img_url in enumerate(image_urls, start=1):
            img_json_file = os.path.join(data_dir, f'{i:d}.json')

            if os.path.exists(img_json_file):
                with open(img_json_file, 'r') as json_file:
                    img_json = json.load(json_file)
            else:
                print(f"Файл {img_json_file} не найден, пропускаем.")
                continue

            input_output_record = {
                "input_image": os.path.join(data_dir, f'{i}.jpg'),
                "input_text": USER_PROMPT,
                "output_text": json.dumps(img_json, ensure_ascii=False)
            }

            prompts_record = {
                "contents": [
                    {
                        "role": "system",
                        "parts": [
                            {
                                "text": SYSTEM_PROMPT
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "parts": [
                            {
                                "fileData": {
                                    "mimeType": "image/jpeg",
                                    "fileUri": os.path.join(data_dir, f'{i}.jpg')
                                }
                            },
                            {
                                "text": USER_PROMPT
                            }
                        ]
                    },
                    {
                        "role": "model",
                        "parts": [
                            {
                                "text": json.dumps(img_json, ensure_ascii=False)
                            }
                        ]
                    }
                ]
            }

            outfile.write(json.dumps(input_output_record, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate JSONL file from image URLs and corresponding JSON data.")

    parser.add_argument('image_urls_file', type=str, help='Path to the file containing image URLs')
    parser.add_argument('data_dir', type=str, help='Directory containing img-xxx.json files')
    parser.add_argument('output_file', type=str, help='Path to the output JSONL file')

    args = parser.parse_args()

    create_jsonl(args.image_urls_file, args.data_dir, args.output_file)
