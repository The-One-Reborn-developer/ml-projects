import os
import cv2

from pprint import pprint
from easyocr import Reader


image_path = os.path.join('./easyocr-test/test.jpg')
image = cv2.imread(image_path)

reader = Reader(['en', 'ro'], gpu=False)
result = reader.readtext(image)
pprint(result, indent=2)
