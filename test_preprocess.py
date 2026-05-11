import torch
from inference_utils import preprocess_image
img = preprocess_image("coco/val2017/000000000139.jpg", 640)
print("Output tensor shape:", img.shape)
