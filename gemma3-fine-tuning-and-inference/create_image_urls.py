import os
import re

from pathlib import Path


DIRECTORY = Path('.') / 'romanian_bank_cards'
OUTPUT_FILE = DIRECTORY / 'urls.txt'


def natural_sort_key(file_path):
    file_name = os.path.basename(file_path)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', file_name)]


def list_jpg_files():
    try:
        jpg_files = []
        for root, _, files in os.walk(DIRECTORY):
            for file in files:
                if file.lower().endswith(".jpg"):
                    jpg_files.append(os.path.abspath(os.path.join(root, file)))
        
        jpg_files.sort(key=natural_sort_key)
        
        with open(OUTPUT_FILE, 'w') as f:
            for file_path in jpg_files:
                f.write(file_path + "\n")
        
        print(f"File paths saved to {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    list_jpg_files()
