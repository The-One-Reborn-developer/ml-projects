import os
import sys
import re

def natural_sort_key(file_path):
    file_name = os.path.basename(file_path)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', file_name)]

def list_jpg_files(directory, output_file):
    try:
        jpg_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(".jpg"):
                    jpg_files.append(os.path.abspath(os.path.join(root, file)))
        
        jpg_files.sort(key=natural_sort_key)
        
        with open(output_file, 'w') as f:
            for file_path in jpg_files:
                f.write(file_path + "\n")
        
        print(f"File paths saved to {output_file}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <directory_path> <output_file>")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    output_file_path = sys.argv[2]
    
    list_jpg_files(directory_path, output_file_path)
