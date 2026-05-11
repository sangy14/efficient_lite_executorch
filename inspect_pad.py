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
        if node.target == torch.ops.aten.pad.default:
            pad = node.args[1]
            if isinstance(pad, (list, tuple)) and len(pad) >= 4:
                # pad format is [pad_w_left, pad_w_right, pad_h_top, pad_h_bottom]
                if pad[0] != pad[1] or pad[2] != pad[3] or pad[0] != pad[2]:
                    print(f"Asymmetric pad found: {node.name}, {pad}")
