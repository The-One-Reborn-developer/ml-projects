import os
import cv2
from easyocr import Reader

input_dir = './dataset/validation'
output_dir = './dataset/validation_crops'
labels_path = './dataset/validation_crops/labels.txt'
os.makedirs(output_dir, exist_ok=True)

reader = Reader(['en', 'ro'], gpu=False)

with open(labels_path, 'w') as labels_file:
    for image_filename in os.listdir(input_dir):
        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(input_dir, image_filename)
            image_cv = cv2.imread(image_path)

            results = reader.readtext(image_cv)

            for idx, (bbox, text, conf) in enumerate(results):
                x_coords = [int(point[0]) for point in bbox]
                y_coords = [int(point[1]) for point in bbox]

                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                cropped = image_cv[y_min:y_max, x_min:x_max]
                crop_filename = f'{os.path.splitext(image_filename)[0]}_crop_{idx}.png'
                crop_path = os.path.join(output_dir, crop_filename)

                cv2.imwrite(crop_path, cropped)
                labels_file.write(f'{crop_path} {text}\n')
