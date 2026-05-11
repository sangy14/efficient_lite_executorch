"""
Export EfficientDet-Lite0 to ExecutorTorch with XNNPACK backend.
"""
import glob
import torch
import torch.nn as nn
import os
import random
import contextlib
import numpy as np
import warnings
import json
import math
from typing import List
from torch.export import export, Dim
import torchvision
import torchvision.ops.boxes as box_ops

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Monkey patch F.pad
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

try:
    from effdet import create_model
    from effdet.anchors import Anchors
    EFFDET_AVAILABLE = True
except ImportError:
    EFFDET_AVAILABLE = False

from inference_utils import preprocess_image, load_coco_dataset, get_file_size_mb

BASE_OUTPUT_DIR = "efficientdet_lite0_xnnpack_models"
MODEL_NAME = "tf_efficientdet_lite0"
BATCH_SIZE = 1
INPUT_SIZE = 320
MAX_CALIBRATION_IMAGES = 20

@contextlib.contextmanager
def patch_batch_norm_for_fp32():
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

def ensure_dir_for_file(filename):
    dirpath = os.path.dirname(filename)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

class EfficientDetExportWrapper(nn.Module):
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
        batch_size = cls_outputs[0].shape[0]
        num_classes = self.num_classes
        k = self.max_detection_points
        cls_outputs_all = torch.cat([
            cls_outputs[level].permute(0, 2, 3, 1).reshape(batch_size, -1, num_classes)
            for level in range(self.num_levels)
        ], dim=1)
        box_outputs_all = torch.cat([
            box_outputs[level].permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
            for level in range(self.num_levels)
        ], dim=1)
        _, cls_topk_indices_all = torch.topk(
            cls_outputs_all.reshape(batch_size, -1), dim=1, k=k
        )
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
            cls_outputs_all, 1, indices_all.unsqueeze(2).expand(batch_size, k, num_classes)
        )
        cls_outputs_topk = torch.gather(cls_outputs_topk, 2, classes_all.unsqueeze(2))
        return cls_outputs_topk, box_outputs_topk, indices_all_f, classes_all_f

    def _decode_box_outputs(self, rel_codes, anchors):
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
        max_det = self.max_det_per_image
        n = boxes.shape[0]
        max_coord = 10000.0
        offset_boxes = boxes + class_ids_f.unsqueeze(1) * max_coord
        _, sorted_idx = torch.topk(scores, k=self.max_detection_points, dim=0)
        sorted_idx_f = sorted_idx.float()
        sorted_boxes = offset_boxes[sorted_idx]
        sorted_scores = scores[sorted_idx]
        sb_x1, sb_y1, sb_x2, sb_y2 = sorted_boxes.unbind(dim=1)
        areas = (sb_x2 - sb_x1) * (sb_y2 - sb_y1)
        suppressed = torch.zeros(n, dtype=torch.float32, device=boxes.device)
        all_selected_idx_f = []
        all_selected_valid = []
        for _i in range(max_det):
            available_scores = sorted_scores * (1.0 - suppressed)
            _, best_1 = torch.topk(available_scores, k=1, dim=0)
            best_score = torch.gather(available_scores, 0, best_1)
            all_selected_valid.append((best_score > 0.0).float())
            best_orig_idx_f = torch.gather(sorted_idx_f, 0, best_1)
            all_selected_idx_f.append(best_orig_idx_f)
            best_box = torch.gather(sorted_boxes, 0, best_1.unsqueeze(1).expand(1, 4))
            bb_x1, bb_y1, bb_x2, bb_y2 = best_box.unbind(dim=1)
            ix1 = torch.max(bb_x1, sb_x1)
            iy1 = torch.max(bb_y1, sb_y1)
            ix2 = torch.min(bb_x2, sb_x2)
            iy2 = torch.min(bb_y2, sb_y2)
            inter = torch.clamp(iy2 - iy1, min=0.0) * torch.clamp(ix2- ix1, min=0.0)
            union = areas + (bb_x2 - bb_x1) * (bb_y2 - bb_y1) - inter
            iou = inter / (union + 1e-6)
            is_overlap = (iou > self.iou_threshold).float()
            suppressed = torch.clamp(suppressed + is_overlap, min=0.0, max=1.0)
        selected_indices_f = torch.cat(all_selected_idx_f)
        selected_valid = torch.cat(all_selected_valid)
        return selected_indices_f, selected_valid

    def forward(self, x):
        class_out, box_out = self.model(x)
        cls_topk, box_topk, indices_f, classes_f = self._post_process(class_out, box_out)
        idx = indices_f[0].long()
        anchor_boxes = torch.gather(self.anchor_boxes, 0, idx.unsqueeze(1).expand(-1, 4))
        decoded_boxes = self._decode_box_outputs(box_topk[0].float(), anchor_boxes)
        scores = cls_topk[0].sigmoid().squeeze(1).float()
        selected_indices_f, selected_valid = self._fixed_nms(decoded_boxes, scores, classes_f[0])
        selected_indices = selected_indices_f.long()
        sel_2d = selected_indices.unsqueeze(1).expand(self.max_det_per_image, 4)
        det_boxes = torch.gather(decoded_boxes, 0, sel_2d)
        det_scores = torch.gather(scores, 0, selected_indices)
        det_classes_raw_f = torch.gather(classes_f[0], 0, selected_indices)
        det_classes = det_classes_raw_f + 1.0
        valid_float = selected_valid.float()
        det_boxes = det_boxes * valid_float.unsqueeze(1)
        det_scores = det_scores * valid_float
        det_classes = det_classes * valid_float
        detections = torch.cat([det_boxes, det_scores.unsqueeze(1), det_classes.unsqueeze(1)], dim=1)
        return detections.unsqueeze(0)

def export_and_save_model(model, example_input, output_name, description, batch_size, use_xnnpack=True):
    print(f"\nEXPORTING {description}")
    try:
        model.eval()
        model.requires_grad_(False)
    except: pass

    def dealias_nodes(program):
        # Lite0 likely doesn't need the spatial shift fix Lite4 needed
        return program

    try:
        with torch.no_grad():
            if isinstance(model, torch.export.ExportedProgram):
                exported_program = model
            else:
                exported_program = export(model, example_input)
        
        if use_xnnpack:
            edge_program = to_edge_transform_and_lower(exported_program, partitioner=[XnnpackPartitioner()])
            edge_program = dealias_nodes(edge_program)
        else:
            edge_program = to_edge_transform_and_lower(exported_program)
            
        executorch_program = edge_program.to_executorch()
        output_filename = os.path.join(BASE_OUTPUT_DIR, output_name)
        ensure_dir_for_file(output_filename)
        with open(output_filename, "wb") as f:
            executorch_program.write_to_file(f)
        print(f" Done! Saved to {output_filename}")
        return executorch_program, output_filename
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def quantize_model(model, example_input, quantizer_config, calibration_data=None):
    model.eval()
    model.requires_grad_(False)
    gm = export(model, example_input).module()
    quantizer = XNNPACKQuantizer()
    for name, _ in model.model.named_children():
        quantizer.set_module_name(f"model.{name}", quantizer_config)
    prepared_model = prepare_pt2e(gm, quantizer)
    if calibration_data:
        with torch.no_grad():
            for batch in calibration_data:
                prepared_model(*batch)
    return convert_pt2e(prepared_model)

def load_lite0_model(pretrained=True):
    raw_model = create_model(MODEL_NAME, pretrained=pretrained, image_size=(INPUT_SIZE, INPUT_SIZE))
    raw_model.eval()
    for m in raw_model.modules():
        if isinstance(m, nn.Conv2d) and m.bias is None:
            m.register_parameter('bias', nn.Parameter(torch.zeros(m.out_channels)))
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if getattr(m, 'weight', None) is None: m.register_parameter('weight', nn.Parameter(torch.ones(m.num_features)))
            if getattr(m, 'bias', None) is None: m.register_parameter('bias', nn.Parameter(torch.zeros(m.num_features)))
            if getattr(m, 'running_mean', None) is None: m.register_buffer('running_mean', torch.zeros(m.num_features))
            if getattr(m, 'running_var', None) is None: m.register_buffer('running_var', torch.ones(m.num_features))
    wrapped = EfficientDetExportWrapper(raw_model, raw_model.config)
    wrapped.eval()
    return wrapped

def export_all():
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    example_input = (torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE),)
    results = {}
    
    # FP32
    model = load_lite0_model()
    with patch_batch_norm_for_fp32():
        _, pte = export_and_save_model(model, example_input, "lite0_fp32.pte", "FP32", 1)
        if pte: results["fp32"] = pte

    # INT8
    try:
        model_q = load_lite0_model()
        q_model = quantize_model(model_q, example_input, get_symmetric_quantization_config(is_per_channel=True))
        with patch_batch_norm_for_fp32():
            _, pte = export_and_save_model(q_model, example_input, "lite0_int8_perchannel.pte", "INT8", 1)
            if pte: results["int8"] = pte
    except Exception as e:
        print(f"Quantization failed: {e}")

    print("\nSummary:", results)

if __name__ == "__main__":
    export_all()