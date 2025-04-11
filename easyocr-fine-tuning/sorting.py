import os

input_file = './dataset/validation_crops/labels.txt'
output_file = './dataset/validation_crops/sorted_labels.txt'

def extract_img_index(line):
    filename = line.split()[0]  # e.g., './dataset/validation_crops/img-001_crop_4.png'
    base = os.path.basename(filename)  # 'img-001_crop_4.png'
    parts = base.split('_')  # ['img-001', 'crop', '4.png']
    img_part = parts[0]  # 'img-001'
    crop_part = int(parts[2].split('.')[0])  # 4
    return img_part, crop_part

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

sorted_lines = sorted(lines, key=lambda line: extract_img_index(line))

with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines(sorted_lines)

print(f"Sorted labels saved to {output_file}")
