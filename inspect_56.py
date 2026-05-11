import torch
from export_xnnpack import patch_batch_norm_for_fp32
from torch.export import export
from effdet import create_model
from export_xnnpack import EfficientDetExportWrapper

model = create_model("tf_efficientdet_lite4", pretrained=False, num_classes=90)
model.eval()
wrapper = EfficientDetExportWrapper(model, model.config)
wrapper.eval()

with patch_batch_norm_for_fp32():
    example_input = (torch.randn(1, 3, 640, 640),)
    ep = export(wrapper, example_input)
    
    for node in ep.graph.nodes:
        if 'tensor_meta' in node.meta:
            shape = node.meta['tensor_meta'].shape
            if 56 in shape:
                print(f"Node: {node.name}, Target: {node.target}, Shape: {shape}")
