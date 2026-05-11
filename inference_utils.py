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
    resized_image = input_image.resize((new_w, new_h), Image.BILINEAR)
    padded_image = Image.new("RGB", (input_size, input_size), (114, 114, 114)) #Typically 114 for detection models
    padded_image.paste(resized_image, ((input_size - new_w) // 2, (input_size - new_h) // 2))
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        # EfficientDet typically uses ImageNet normalization
        transforms.Normalize(
            mean=[0.485, 0.456,     0.406],
            std=[0.229, 0.224, 0.225]
        ),
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

def get_coco_category_names():
    """
    Get COCO category names and IDs.
    Returns:
        dict: Mapping from category_id to category_name
    """
    coco_categories = {
        1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
        6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
        11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
        16: "cat", 17: "dog", 18: "horse", 19: "sheep", 20: "cow",
        21: "elephant", 22: "bear", 23: "zebra", 24: "giraffe", 25: "backpack",
        26: "umbrella", 27: "handbag", 28: "tie", 29: "suitcase", 30: "frisbee",
        31: "skis", 32: "snowboard", 33: "sports ball", 34: "kite", 35: "baseball bat",
        36: "baseball glove", 37: "skateboard", 38: "surfboard", 39: "tennis racket",
        40: "bottle", 41: "wine glass", 42: "cup", 43: "fork", 44: "knife",
        45: "spoon", 46: "bowl", 47: "banana", 48: "apple", 49: "sandwich",
        50: "orange", 51: "broccoli", 52: "carrot", 53: "hot dog", 54: "pizza",
        55: "donut", 56: "cake", 57: "chair", 58: "couch", 59: "potted plant",
        60: "bed", 61: "dining table", 62: "toilet", 63: "tv", 64: "laptop",
        65: "mouse", 66: "remote", 67: "keyboard", 68: "microwave", 69: "oven",
        70: "toaster", 71: "sink", 72: "refrigerator", 73: "book", 74: "clock",
        75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier", 79:
        "toothbrush"
    }
    return coco_categories

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
    
def calculate_coco_metrics(predictions_json, annotations_json):
    """
    Calculate COCO evaluation metrics.
    Args:
        predictions_json: Path to predictions JSON
        annotations_json: Path to ground truth annotations JSON
    Returns:
        dict: COCO metrics (mAP, mAR, etc.)
    """
    try:
        coco_gt = COCO(annotations_json)
        coco_dt = coco_gt.loadRes(predictions_json)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
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