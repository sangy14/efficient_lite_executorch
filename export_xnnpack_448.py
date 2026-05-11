"""
Export EfficientDet-Lite4 to ExecutorTorch with XNNPACK backend.
Includes export-friendly post-processing (top-K, anchor decoding, fixed-iteration
greedy NMS) that faithfully reproduces effdet's DetBenchPredict pipeline but uses
ONLY fixed-size tensor operations — no data-dependent Python control flow.
Output: [batch, max_det_per_image, 6] -> (x1, y1, x2, y2, score, class_id)
class_id is 1-based (0 = background), matching effdet convention.
IMPORTANT: Run with virtual environment activated:
source venv/bin/activate
python export_efficientdet_lite4_xnnp.py
"""
import glob
import torch
import torch.nn as nn
import os
import random
import contextlib
import numpy as np
import warnings
from torch.export import export, Dim
import torchvision
import torchvision.ops.boxes as box_ops
# Suppress warnings from optree/JAX about LeafSpec deprecation
warnings.filterwarnings("ignore", category=FutureWarning)
# Monkey patch F.pad to avoid -inf padding which causes XNNPACK serialization
# errors
_original_pad = torch.nn.functional.pad
def _patched_pad(input, pad, mode='constant', value=None):
    if mode == 'constant' and value is not None and (value == float('-inf') or value == -float('inf')):
        value = -10000.0
    return _original_pad(input, pad, mode=mode, value=value)
torch.nn.functional.pad = _patched_pad
from executorch.exir import to_edge, to_edge_transform_and_lower
from executorch.backends.xnnpack.recipes.xnnpack_recipe_provider import (
XNNPACKQuantizer,
get_symmetric_quantization_config,
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
# Safely bypass EXIR passes if they crash, rather than failing the whole export
try:
    from torch.fx.passes.infra.pass_base import PassResult
# Patch EXIR's FBN pass (This is the one that crashes during to_edge)
    try:
        import executorch.exir.passes.fuse_batch_norm_pass as exir_fbn
        _old_exir_fbn_call = exir_fbn.FuseBatchNormPass.call
        def _safe_exir_fbn_call(self, graph_module):
            try:
                return _old_exir_fbn_call(self, graph_module)
            except Exception as e:
                print(f" [Warning] Bypassing EXIR FuseBatchNormPass due to error:{e}")
                return PassResult(graph_module, False)
        exir_fbn.FuseBatchNormPass.call = _safe_exir_fbn_call
    except Exception:
        pass
    # Patch XNNPACK's FBN pass (Just in case)
    try:
        import executorch.backends.xnnpack.passes.fuse_batch_norm_pass as fbn
        _old_fbn_call = fbn.FuseBatchNormPass.call
        def _safe_fbn_call(self, graph_module):
            try:
                return _old_fbn_call(self, graph_module)
            except Exception as e:
                print(f" [Warning] Bypassing XNNPACK FuseBatchNormPass due to error:{e}")
                return PassResult(graph_module, False)
        fbn.FuseBatchNormPass.call = _safe_fbn_call
    except Exception:
        pass
except Exception:
    pass
# No longer needed as we modified effdet/efficientdet.py to use static sizes.
try:
    from effdet import create_model
    from effdet.anchors import Anchors
    EFFDET_AVAILABLE = True
    # Patch timm padding immediately after import to replace -inf with -10000.0
    try:
        import timm.models.layers.padding as padding_old
        _orig_pad_same_old = padding_old.pad_same
        def _patched_pad_same_old(x, k, s, d=(1, 1), value=0):
            if value == float('-inf') or value == -float('inf'):
                value = -10000.0
            return _orig_pad_same_old(x, k, s, d, value=value)
        padding_old.pad_same = _patched_pad_same_old
    except ImportError:
        pass
    try:
        import timm.layers.padding as padding_new
        _orig_pad_same_new = padding_new.pad_same
        def _patched_pad_same_new(x, k, s, d=(1, 1), value=0):
            if value == float('-inf') or value == -float('inf'):
                value = -10000.0
            return _orig_pad_same_new(x, k, s, d, value=value)
        padding_new.pad_same = _patched_pad_same_new
    except ImportError:
        pass

except ImportError:
    EFFDET_AVAILABLE = False
    print("Warning: effdet not available. Install with: pip install effdet")
from inference_utils import preprocess_image, load_coco_dataset, get_file_size_mb
BASE_OUTPUT_DIR = "efficientdet_lite4_448_models"
MODEL_NAME = "tf_efficientdet_lite4"
BATCH_SIZE = 1
INPUT_SIZE = 448
MAX_CALIBRATION_IMAGES = 20
@contextlib.contextmanager
def patch_batch_norm_for_fp32():
    """
    Context manager to mathematically decompose batch norm into primitive ops.
    This safely bypasses EXIR's FuseBatchNormPass crash during FP32 export,
    without globally affecting base model exports or INT8 quantization.
    """
    _orig_f_batch_norm = torch.nn.functional.batch_norm
    def _patched_f_batch_norm(input, running_mean, running_var, weight=None,
    bias=None, training=False, momentum=0.1, eps=1e-5):
        if training:
            return _orig_f_batch_norm(input, running_mean, running_var, weight,
            bias, training, momentum, eps)
        dim = input.dim()
        shape = [1,-1] + [1] * (dim - 2)
        out = input
        if running_mean is not None and running_var is not None:
            out = (out - running_mean.view(shape)) * torch.rsqrt(running_var.view(shape) + eps)  
        if weight is not None:
            out = out * weight.view(shape)
        if bias is not None:
            out = out + bias.view(shape)
        return out
    torch.nn.functional.batch_norm = _patched_f_batch_norm
    try:
        yield
    finally:
        torch.nn.functional.batch_norm = _orig_f_batch_norm

import json
import math
from typing import List

def ensure_dir_for_file(filename):
    dirpath = os.path.dirname(filename)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
# =============================================================================
# EfficientDetExportWrapper
#
# ALL indexing with argmax results uses torch.gather() to avoid
# data-dependent .item() calls that torch.export cannot trace.
# =============================================================================
class EfficientDetExportWrapper(nn.Module):
    """
    Complete EfficientDet with export-friendly post-processing.
    Output:
    detections: [batch, max_det_per_image, 6]   
    Each row: (x1, y1, x2, y2, score, class_id)
    class_id is 1-based (background = 0).
    Rows beyond actual detections are zero-filled.
    """
    def __init__(self, model, config, max_detection_points=5000,
                max_det_per_image=100, iou_threshold=0.5):
        super().__init__()
        self.model = model
        self.num_classes = config.num_classes
        self.num_levels = config.num_levels
        self.max_detection_points = max_detection_points
        self.max_det_per_image = max_det_per_image
        self.iou_threshold = iou_threshold
        anchor_gen = Anchors.from_config(config)
        self.register_buffer("anchor_boxes", anchor_gen.boxes)
        print(f" Registered {anchor_gen.boxes.shape[0]} anchor boxes as buffer")
    
    def _post_process(self, cls_outputs, box_outputs):
        """Top-K selection across all FPN levels. Matches effdet/bench.py."""
        batch_size = cls_outputs[0].shape[0]
        num_classes = self.num_classes
        k = self.max_detection_points
        cls_outputs_all = torch.cat([
            cls_outputs[level].permute(0, 2, 3, 1).reshape(batch_size,
            (cls_outputs[level].shape[2] * cls_outputs[level].shape[3] *
            cls_outputs[level].shape[1]) // num_classes, num_classes)
            for level in range(self.num_levels)
        ], dim=1)
        box_outputs_all = torch.cat([
            box_outputs[level].permute(0, 2, 3, 1).reshape(batch_size,
            (box_outputs[level].shape[2] * box_outputs[level].shape[3] *
            box_outputs[level].shape[1]) // 4, 4)
            for level in range(self.num_levels)
        ], dim=1)
        _, cls_topk_indices_all = torch.topk(
            cls_outputs_all.reshape(batch_size, cls_outputs_all.shape[1] *
            num_classes), dim=1, k=k
        )
        # Cast to float to avoid Long tensor observers during PT2E quantization
        cls_topk_indices_all_f = cls_topk_indices_all.float()
        num_classes_f = float(num_classes)
        indices_all_f = torch.floor(cls_topk_indices_all_f / num_classes_f)
        classes_all_f = torch.remainder(cls_topk_indices_all_f, num_classes_f)
        indices_all = indices_all_f.long()
        classes_all = classes_all_f.long()
        box_outputs_topk = torch.gather(
            box_outputs_all, 1, indices_all.unsqueeze(2).expand(batch_size, k, 4)
        )
        cls_outputs_topk = torch.gather(
            cls_outputs_all, 1, indices_all.unsqueeze(2).expand(batch_size, k,
            num_classes)
        )
        cls_outputs_topk = torch.gather(
            cls_outputs_topk, 2, classes_all.unsqueeze(2)
        )
        return cls_outputs_topk, box_outputs_topk, indices_all_f, classes_all_f

    def _decode_box_outputs(self, rel_codes, anchors):
        """Decode boxes. Matches effdet/anchors.py
        decode_box_outputs(output_xyxy=True)."""
        anc_y1, anc_x1, anc_y2, anc_x2 = anchors.unbind(dim=1)
        ycenter_a = (anc_y1 + anc_y2) / 2.0
        xcenter_a = (anc_x1 + anc_x2) / 2.0
        ha = anc_y2 - anc_y1
        wa = anc_x2 - anc_x1
        ty, tx, th, tw = rel_codes.unbind(dim=1)
        w = torch.exp(tw) * wa
        h = torch.exp(th) * ha
        ycenter = ty * ha + ycenter_a
        xcenter = tx * wa + xcenter_a
        ymin = ycenter - h / 2.0
        xmin = xcenter - w / 2.0
        ymax = ycenter + h / 2.0
        xmax = xcenter + w / 2.0
        return torch.stack([xmin, ymin, xmax, ymax], dim=1)

    def _fixed_nms(self, boxes, scores, class_ids_f):
        """
        Fixed-iteration greedy NMS — fully export-compatible.
        KEY FIX: torch.argmax() returns a 0-dim tensor with symbolic value.
        Using tensor[symbolic_idx] triggers .item() which is data-dependent.
        ALL indexing uses torch.gather() — pure tensor ops, no .item().
        """
        max_det = self.max_det_per_image
        n = boxes.shape[0]
        max_coord = 10000.0 # Avoid boxes.max() which reduces over all dims and can trigger 0-D bounds
        offset_boxes = boxes + class_ids_f.unsqueeze(1) * max_coord
        _, sorted_idx = torch.topk(scores, k=self.max_detection_points, dim=0)
        sorted_idx_f = sorted_idx.float()
        sorted_boxes = offset_boxes[sorted_idx]
        sorted_scores = scores[sorted_idx]
        sb_x1, sb_y1, sb_x2, sb_y2 = sorted_boxes.unbind(dim=1)
        areas = (sb_x2 - sb_x1) * (sb_y2 - sb_y1)
        # Use float32 to avoid boolean tensor quantization crashes (linspace_cpu not implemented for Bool)
        suppressed = torch.zeros(n, dtype=torch.float32, device=boxes.device)
        all_selected_idx_f = []
        all_selected_valid = []
        for _i in range(max_det):
            available_scores = sorted_scores * (1.0 - suppressed)
            # Use topk(k=1) instead of argmax to prevent 0-D scalar tensors during
            # export
            _, best_1 = torch.topk(available_scores, k=1, dim=0)
            # ===== ALL INDEXING VIA torch.gather (no .item()) =====
            # best_score -> shape [1]
            best_score = torch.gather(available_scores, 0, best_1)
            all_selected_valid.append((best_score > 0.0).float())
            # best_orig_idx_f -> shape [1]
            best_orig_idx_f = torch.gather(sorted_idx_f, 0, best_1)
            all_selected_idx_f.append(best_orig_idx_f)
            # best_box -> shape [1, 4]
            best_box = torch.gather(
                sorted_boxes, 0, best_1.unsqueeze(1).expand(1, 4)
            )
            # Unbind along dim=1 -> four tensors of shape [1]
            bb_x1, bb_y1, bb_x2, bb_y2 = best_box.unbind(dim=1)
            ix1 = torch.max(bb_x1, sb_x1)
            iy1 = torch.max(bb_y1, sb_y1)
            ix2 = torch.min(bb_x2, sb_x2)
            iy2 = torch.min(bb_y2, sb_y2)
            inter = torch.clamp(iy2 - iy1, min=0.0, max=100000.0) * torch.clamp(ix2- ix1, min=0.0, max=100000.0)
            union = areas + (bb_x2 - bb_x1) * (bb_y2 - bb_y1) - inter
            iou = inter / (union + 1e-6)
            is_overlap = (iou > self.iou_threshold).float()
            suppressed = torch.clamp(suppressed + is_overlap, min=0.0, max=1.0)
        selected_indices_f = torch.cat(all_selected_idx_f)
        selected_valid = torch.cat(all_selected_valid)
        return selected_indices_f, selected_valid

    def forward(self, x):
        """Full forward: backbone -> top-K -> decode -> NMS -> [1, max_det, 6]"""
        class_out, box_out = self.model(x)
        cls_topk, box_topk, indices_f, classes_f = self._post_process(class_out,box_out)
        idx = indices_f[0].long()
        anchor_boxes = torch.gather(self.anchor_boxes, 0, idx.unsqueeze(1).expand(-1, 4))
        decoded_boxes = self._decode_box_outputs(box_topk[0].float(), anchor_boxes)
        scores = cls_topk[0].sigmoid().squeeze(1).float()
        selected_indices_f, selected_valid = self._fixed_nms(decoded_boxes, scores, classes_f[0])
        selected_indices = selected_indices_f.long()
        # Gather final detections (also using torch.gather)
        sel_2d = selected_indices.unsqueeze(1).expand(self.max_det_per_image, 4)
        det_boxes = torch.gather(decoded_boxes, 0, sel_2d)
        det_scores = torch.gather(scores, 0, selected_indices)
        det_classes_raw_f = torch.gather(classes_f[0], 0, selected_indices)
        det_classes = det_classes_raw_f + 1.0
        valid_float = selected_valid.float()
        det_boxes = det_boxes * valid_float.unsqueeze(1)
        det_scores = det_scores * valid_float
        det_classes = det_classes * valid_float
        detections = torch.cat([
            det_boxes,
            det_scores.unsqueeze(1),
            det_classes.unsqueeze(1),
        ], dim=1)
        return detections.unsqueeze(0)
        # =============================================================================
# Export / Quantization / Calibration Helpers
# =============================================================================
def export_and_save_model(model, example_input, output_name, description,batch_size, use_xnnpack=True):
    print("\n" + "=" * 70)
    print(f"EXPORTING {description}")
    print("=" * 70)
    # Disable gradients to avoid autograd artifacts that prevent operator fusion in
    # EXIR
    try:
        model.eval()
        model.requires_grad_(False)
    except Exception as e:
        print(f" [Info] Skipping eval()/requires_grad_() for already exported model.")
    def dealias_nodes(program):
        """No de-aliasing needed for 448x448 resolution."""
        return program

    print("Step 1/3: Exporting to PyTorch FX graph...")
    try:
        with torch.no_grad():
            if isinstance(model, torch.export.ExportedProgram):
                exported_program = model
                print(" Model is already an ExportedProgram, skipping torch.export")
            else:
                if batch_size > 1:
                    batch = Dim("batch", max=1024)
                    dynamic_shapes = ({0: batch},)
                    exported_program = export(
                        model, example_input,
                        dynamic_shapes=dynamic_shapes
                    )
                else:
                    exported_program = export(model, example_input)
                    print(" torch.export succeeded")
        
        # if use_xnnpack:
        #     print(" Applying Node De-aliasing...")
        #     exported_program = dealias_nodes(exported_program)

    except Exception as e:
        print(f"Error during export: {e}")
        raise
    print("Step 2/3: Lowering to ExecutorTorch Edge IR...")
    try:
        if use_xnnpack:
            edge_program = to_edge_transform_and_lower(
                exported_program,
                partitioner=[XnnpackPartitioner()]
            )
            print(" XNNPACK partitioning succeeded")
            print(" Applying Node De-aliasing to Edge IR...")
            edge_program = dealias_nodes(edge_program)
        else:
            edge_program = to_edge_transform_and_lower(exported_program)
    except Exception as e:
        print(f"Error during Edge IR lowering/partitioning: {e}")
        raise
    print("Step 3/3: Converting to ExecutorTorch format...")
    executorch_program = edge_program.to_executorch()
    output_filename = os.path.join(BASE_OUTPUT_DIR, output_name)
    ensure_dir_for_file(output_filename)
    print(f" Saving to {output_filename}...")
    with open(output_filename, "wb") as f:
        executorch_program.write_to_file(f)
    file_size = get_file_size_mb(output_filename)
    print(f" Done! File size: {file_size:.2f} MB")
    return executorch_program, output_filename
def quantize_model(model, example_input, quantizer_config, calibration_data=None):
    model.eval()
    model.requires_grad_(False)
    print("Exporting for quantization...")
    with torch.no_grad():
        gm = export(model, example_input).module()
        print("Applying quantization...")
        quantizer = XNNPACKQuantizer()
        # Restrict quantization specifically to the EfficientDet submodules
        # This prevents the post-processing (NMS, top-k) from being quantized,
        # which avoids Long tensor crashes and preserves bounding box accuracy.
        for name, _ in model.model.named_children():
            quantizer.set_module_name(f"model.{name}", quantizer_config)
        print("Preparing quantization observers...")
        prepared_model = prepare_pt2e(gm, quantizer)
        print("Calibrating...")
        if calibration_data:
            print(f" Running calibration on {len(calibration_data)} batches...")
            with torch.no_grad():
                for batch_idx, batch_input in enumerate(calibration_data):
                    if batch_idx % 5 == 0:
                        print(f" Calibration batch {batch_idx +1}/{len(calibration_data)}")
                        prepared_model(*batch_input)
                else:
                    prepared_model(*example_input)
        print("Converting to quantized model...")
        quantized_model = convert_pt2e(prepared_model)
        return quantized_model
        
def load_efficientdet_model(model_name="tf_efficientdet_lite4", pretrained=True):
    if not EFFDET_AVAILABLE:
        raise RuntimeError("effdet is required. Install with: pip install effdet")
    print(f"Loading {model_name} with image_size=({INPUT_SIZE}, {INPUT_SIZE})...")
    # Pass image_size to ensure BiFPN internal shapes match our input
    raw_model = create_model(model_name, pretrained=pretrained, image_size=(INPUT_SIZE, INPUT_SIZE))
    raw_model.eval()
    # Pre-process Conv2d and BatchNorm2d to avoid EXIR pass crashes with None
    # This fixes the 'FuseBatchNormPass' crash by ensuring bias/weight tensors
    # exist.
    for m in raw_model.modules():
        if isinstance(m, nn.Conv2d) and m.bias is None:
            m.register_parameter(
                'bias', nn.Parameter(torch.zeros(m.out_channels, dtype=m.weight.dtype, device=m.weight.device))
            )
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            device = next((p.device for p in m.parameters() if p is not None),
            next((b.device for b in m.buffers() if b is not None), torch.device('cpu')))
            if getattr(m, 'weight', None) is None:
                m.register_parameter('weight',
                nn.Parameter(torch.ones(m.num_features, dtype=torch.float32, device=device)))
            if getattr(m, 'bias', None) is None:
                m.register_parameter('bias',
                nn.Parameter(torch.zeros(m.num_features, dtype=torch.float32, device=device)))
            if getattr(m, 'running_mean', None) is None:
                m.register_buffer('running_mean', torch.zeros(m.num_features,
                dtype=torch.float32, device=device))
            if getattr(m, 'running_var', None) is None:
                m.register_buffer('running_var', torch.ones(m.num_features,
                dtype=torch.float32, device=device))
    config = raw_model.config
    print(f" num_classes={config.num_classes}, num_levels={config.num_levels}")
    print(f" image_size={config.image_size}")
    print(f" max_detection_points={config.max_detection_points}")
    print(f" max_det_per_image={config.max_det_per_image}")
    print("Wrapping in EfficientDetExportWrapper...")
    wrapped_model = EfficientDetExportWrapper(
        model=raw_model,
        config=config,
        max_detection_points=config.max_detection_points,
        max_det_per_image=config.max_det_per_image,
        iou_threshold=0.5,
    )
    wrapped_model.eval()
    return wrapped_model

def prepare_calibration_data(batch_size, input_size):
    coco_images_dir = "coco/val2017"
    if not os.path.exists(coco_images_dir):
        print(f"Warning: {coco_images_dir} not found.")
        calibration_dir = "coco_calibration_subset"
        if os.path.exists(calibration_dir):
            print(f"Using images from {calibration_dir}...")
            image_files = sorted(
                glob.glob(os.path.join(calibration_dir, "*.jpg"))
            )
        else:
            print("Creating synthetic calibration data...")
            os.makedirs(coco_images_dir, exist_ok=True)
            image_files = []
            for i in range(MAX_CALIBRATION_IMAGES):
                img_path = os.path.join(
                    coco_images_dir, f"synthetic_{i:04d}.jpg"
                )
                random_img = np.random.randint(
                    0, 255, (input_size, input_size, 3), dtype=np.uint8
                )
                from PIL import Image
                Image.fromarray(random_img).save(img_path)
                image_files.append(img_path)
    else:
        image_files = sorted(
            glob.glob(os.path.join(coco_images_dir, "*.jpg"))
        )
    image_files = image_files[:MAX_CALIBRATION_IMAGES]
    if not image_files:
        print("Warning: No calibration images found!")
        return None
    print(f"Preparing calibration data from {len(image_files)} images...")
    dataset = []
    current_batch = []
    for img_idx, img_path in enumerate(image_files):
        try:
            if img_idx % 5 == 0:
                print(f" Loading image {img_idx + 1}/{len(image_files)}")
                tensor = preprocess_image(img_path, input_size)
                current_batch.append(tensor)
                if len(current_batch) == batch_size:
                    batch_tensor = torch.stack(current_batch)
                    dataset.append((batch_tensor,))
                    current_batch = []
        except Exception as error:
            print(f"Warning: failed to preprocess {img_path}: {error}")
        if current_batch:
            while len(current_batch) < batch_size:
                current_batch.append(current_batch[0])
            batch_tensor = torch.stack(current_batch)
            dataset.append((batch_tensor,))
    if not dataset:
        print("Warning: No calibration batches created!")
        return None
    print(f"Prepared {len(dataset)} calibration batches (batch_size={batch_size})")
    return dataset

# =============================================================================
# Main Export
# =============================================================================
def export_all_models():
    print("\n" + "=" * 70)
    print("EFFICIENTDET-LITE4 MODEL EXPORT (WITH POST-PROCESSING)")
    print("=" * 70)
    print("Output: [batch, 100, 6] -> (x1, y1, x2, y2, score, class_id)")
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print("\nPreparing calibration data...")
    calibration_data = prepare_calibration_data(BATCH_SIZE, INPUT_SIZE)
    if not calibration_data:
        print("Error: No calibration data available.")
        return {}
    example_input = calibration_data[0]
    print(f"Example input shape: {example_input[0].shape}")
    results = {}
    try:
        model = load_efficientdet_model(MODEL_NAME, pretrained=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        return {}
    print("\nRunning test forward pass...")
    with torch.no_grad():
        test_out = model(example_input[0])
    print(f"Output shape: {test_out.shape}")
    valid_mask = test_out[0, :, 4] > 0.01
    num_valid = valid_mask.sum().item()
    print(f" -> {int(num_valid)} detections with score > 0.01")
    # print("\n[1/2] Skipping FP32 model export...")
    # try:
    #     with patch_batch_norm_for_fp32():
    #         _, fp32_file = export_and_save_model(
    #             model, example_input,
    #             "efficientdet_lite4_fp32_with_nms.pte",
    #             "FP32 MODEL WITH POST-PROCESSING",
    #             batch_size=BATCH_SIZE,
    #         )
    #     fp32_size = get_file_size_mb(fp32_file)
    #     results["fp32_with_nms"] = {
    #         "filename": fp32_file,
    #         "size_mb": fp32_size,
    #         "quantization": "FP32",
    #     }
    # except Exception as e:
    #     print(f"Error exporting FP32 model: {e}")
    # # # ---- INT8 Static Per-Channel (uncomment when ready) ----
    print("\n[2/2] Exporting INT8 static per-channel model...")
    try:
        model_q = load_efficientdet_model(MODEL_NAME, pretrained=True)
        quantized_model = quantize_model(
            model_q, example_input,
            get_symmetric_quantization_config(
                is_per_channel=True, is_dynamic=False
            ),
            calibration_data=calibration_data,
        )
        with patch_batch_norm_for_fp32():
            _, q_file = export_and_save_model(
                quantized_model, example_input,
                "efficientdet_lite4_int8_static_perchannel_with_nms.pte",
                "INT8 STATIC PER-CHANNEL WITH POST-PROCESSING",
                batch_size=BATCH_SIZE,
            )
        q_size = get_file_size_mb(q_file)
        results["int8_static_perchannel"] = {
            "filename": q_file,
            "size_mb": q_size,
            "quantization": "INT8 Static Per-Channel",
        }
    except Exception as e:
        print(f"Error exporting INT8 per-channel model: {e}")

    print("\n[3/3] Exporting INT8 static per-tensor model...")
    try:
        model_qt = load_efficientdet_model(MODEL_NAME, pretrained=True)
        quantized_model_t = quantize_model(
            model_qt, example_input,
            get_symmetric_quantization_config(
                is_per_channel=False, is_dynamic=False
            ),
            calibration_data=calibration_data,
        )
        with patch_batch_norm_for_fp32():
            _, qt_file = export_and_save_model(
                quantized_model_t, example_input,
                "efficientdet_lite4_int8_static_pertensor_with_nms.pte",
                "INT8 STATIC PER-TENSOR WITH POST-PROCESSING",
                batch_size=BATCH_SIZE,
            )
        qt_size = get_file_size_mb(qt_file)
        results["int8_static_pertensor"] = {
            "filename": qt_file,
            "size_mb": qt_size,
            "quantization": "INT8 Static Per-Tensor",
        }
    except Exception as e:
        print(f"Error exporting INT8 per-tensor model: {e}")
    print("\n" + "=" * 70)
    print("EXPORT SUMMARY")
    print("=" * 70)
    for model_type, info in results.items():
        print(f" {model_type}: {info['quantization']} | "
              f"{info['size_mb']:.2f} MB | {info['filename']}")
    print("=" * 70)
    return results
if __name__ == "__main__":
    results = export_all_models()
    os.makedirs("efficientdet_results", exist_ok=True)
    with open("efficientdet_results/export_summary.json", "w") as f:
        json_results = {
            k: {
                "filename": v["filename"],
                "size_mb": round(v["size_mb"], 2),
                "quantization": v["quantization"],
            }
            for k, v in results.items()
        }
        json.dump(json_results, f, indent=2)
        print(f"\nExport summary saved to efficientdet_results/export_summary.json")