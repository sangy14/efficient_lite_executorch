import torch
import os
import json
import csv
from PIL import Image
from torchvision import transforms
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
def preprocess_image(image_path, input_size=320):
    """
    Preprocess image for EfficientDet inference (compatible with XNNPACK).
    Args:
        image_path: Path to input image
        input_size: Target square input size
    Returns:
        torch.Tensor: Preprocessed image tensor (3, H, W) as float32
    """
    input_image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = input_image.size
    # Calculate scale to fit within input_size while maintaining aspect ratio
    scale = input_size / max(orig_w, orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    print(f"Original size: ({orig_w}, {orig_h}), Resized size: ({new_w}, {new_h})")
    # Resize and pad to exact square (input_size x input_size)
    # EfficientDet-Lite models typically expect top-left padding (letterboxing)
    resized_image = input_image.resize((new_w, new_h), Image.BILINEAR)
    padded_image = Image.new("RGB", (input_size, input_size), (114, 114, 114)) 
    padded_image.paste(resized_image, (0, 0))
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        # EfficientDet-Lite models typically expect [0, 1] range without ImageNet normalization
        # transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        # ),
    ])
    img_tensor = preprocess(padded_image)
    # Ensure the tensor is float32 and on CPU (required for XNNPACK)
    img_tensor = img_tensor.to(dtype=torch.float32, device='cpu')
    print(f"Preprocessed image tensor shape: {tuple(img_tensor.shape)}, dtype:{img_tensor.dtype}, device: {img_tensor.device}")
    return img_tensor
def load_images(image_paths, input_size=320):
    """
    Load and preprocess images from the given paths.
    Args:
        image_paths: List of image file paths
        input_size: Target image size
    Returns:
        tuple: (tensors, valid_filenames) where tensors is a list of preprocessed images
    """
    tensors = []
    valid_filenames = []
    print(f"Processing {len(image_paths)} images...")
    for idx, filename in enumerate(image_paths):
        if os.path.exists(filename):
            try:
                img_tensor = preprocess_image(filename, input_size)
                tensors.append(img_tensor)
                valid_filenames.append(filename)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        else:
            print(f"File {filename} not found.")
    return tensors, valid_filenames

def load_coco_dataset(images_dir, annotations_file=None):
    """
    Load COCO dataset images.
    Args:
        images_dir: Directory containing COCO images
        annotations_file: Path to COCO annotations JSON (optional)
    Returns:
        list: List of image paths
    """
    if not os.path.exists(images_dir):
        print(f"Warning: Images directory not found: {images_dir}")
        return []
    # Get all JPEG files
    image_paths = []
    for filename in sorted(os.listdir(images_dir)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(images_dir, filename))
    return image_paths

# Mapping from contiguous indices (0-79) to COCO IDs (1-90)
# Derived from the official COCO 2017 validation set
COCO_ID_MAP = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
]

def get_coco_id(contiguous_id):
    """Map contiguous 0-79 index to COCO 1-90 ID."""
    if 0 <= contiguous_id < len(COCO_ID_MAP):
        return COCO_ID_MAP[contiguous_id]
    return contiguous_id + 1

def get_coco_category_names():
    """
    Get COCO category names and IDs.
    Returns:
        dict: Mapping from category_id to category_name
    """
    coco_id_to_name = {
        1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
        6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
        11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
        16: "bird", 17: "cat", 18: "dog", 18: "horse", 19: "sheep", 20: "cow",
        21: "elephant", 22: "bear", 23: "zebra", 24: "giraffe", 27: "backpack",
        28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee",
        35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat",
        40: "baseball glove", 41: "skateboard", 42: "surfboard", 43: "tennis racket",
        44: "bottle", 46: "wine glass", 47: "cup", 48: "fork", 49: "knife",
        50: "spoon", 51: "bowl", 52: "banana", 53: "apple", 54: "sandwich",
        55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza",
        60: "donut", 61: "cake", 62: "chair", 63: "couch", 64: "potted plant",
        65: "bed", 67: "dining table", 70: "toilet", 72: "tv", 73: "laptop",
        74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave",
        79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator", 84: "book",
        85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier",
        90: "toothbrush"
    }
    return coco_id_to_name

def validate_coco_predictions(predictions_json, annotations_json=None):
    """
    Validate COCO predictions format.
    Args:
        predictions_json: Path to predictions JSON file
        annotations_json: Path to ground truth annotations JSON (optional)
    Returns:
        dict: Validation results
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {}
    }
    try:
        with open(predictions_json, 'r') as f:
            predictions = json.load(f)
    except Exception as e:
        results["valid"] = False
        results["errors"].append(f"Failed to load predictions: {e}")
        return results
    # Check basic structure
    if not isinstance(predictions, list):
        results["valid"] = False
        results["errors"].append("Predictions must be a list")
        return results
    if len(predictions) == 0:
        results["warnings"].append("No predictions found")
    # Validate individual predictions
    image_ids = set()
    category_ids = set()
    for i, pred in enumerate(predictions):
        if not isinstance(pred, dict):
            results["errors"].append(f"Prediction {i} is not a dict")
            continue
        required_fields = {"image_id", "category_id", "bbox", "score"}
        missing_fields = required_fields - set(pred.keys())
        if missing_fields:
            results["errors"].append(f"Prediction {i} missing fields: {missing_fields}")
            image_ids.add(pred.get("image_id"))
            category_ids.add(pred.get("category_id"))
    results["stats"]["num_predictions"] = len(predictions)
    results["stats"]["num_images"] = len(image_ids)
    results["stats"]["num_categories"] = len(category_ids)
    return results
    
def calculate_coco_metrics(predictions, annotations_json, img_ids=None):
    """
    Calculate COCO evaluation metrics.
    Args:
        predictions: Path to predictions JSON OR a list of detection dictionaries
        annotations_json: Path to ground truth annotations JSON
        img_ids: Optional list of image IDs to evaluate on
    Returns:
        dict: COCO metrics (mAP, mAR, etc.)
    """
    try:
        coco_gt = COCO(annotations_json)
        # loadRes can take a filename OR a list of dictionaries directly
        coco_dt = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        
        if img_ids:
            coco_eval.params.imgIds = img_ids
            
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        # Extract metrics
        metrics = {
            "mAP@0.5:0.95": float(coco_eval.stats[0]),
            "mAP@0.5": float(coco_eval.stats[1]),
            "mAP@0.75": float(coco_eval.stats[2]),
            "mAP_small": float(coco_eval.stats[3]),
            "mAP_medium": float(coco_eval.stats[4]),
            "mAP_large": float(coco_eval.stats[5]),
            "mAR@1": float(coco_eval.stats[6]),
            "mAR@10": float(coco_eval.stats[7]),
            "mAR@100": float(coco_eval.stats[8]),
            "mAR_small": float(coco_eval.stats[9]),
            "mAR_medium": float(coco_eval.stats[10]),
            "mAR_large": float(coco_eval.stats[11]),
        }
        return metrics
    except Exception as e:
        print(f"Error calculating COCO metrics: {e}")
        return {}
        
def save_results_json(results, output_path):
    """
    Save results to JSON file.
    Args:
        results: Dictionary to save
        output_path: Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {output_path}")
        
def save_results_csv(results, output_path, fieldnames=None):
    """
    Save results to CSV file.
    Args:
        results: List of dictionaries to save
        output_path: Output file path
        fieldnames: CSV column names (auto-detected if None)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not results:
        print(f"No results to save to {output_path}")
        return
    if fieldnames is None:
        fieldnames = list(results[0].keys())
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} rows to {output_path}")

def get_file_size_mb(filename):
    """Get file size in megabytes."""
    if os.path.exists(filename):
        size_bytes = os.path.getsize(filename)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    return 0.0