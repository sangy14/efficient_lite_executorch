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
        if node.target == torch.ops.aten.convolution.default or node.target == torch.ops.aten.max_pool2d_with_indices.default:
            # For conv: args are (input, weight, bias, stride, padding, dilation, transposed, output_padding, groups)
            if node.target == torch.ops.aten.convolution.default:
                stride = node.args[3]
                padding = node.args[4]
                kernel = node.args[1].meta['tensor_meta'].shape[2:] if hasattr(node.args[1], 'meta') else None
            else:
                kernel = node.args[1]
                stride = node.args[2]
                padding = node.args[3]
                
            if isinstance(stride, (list, tuple)) and len(stride) >= 2 and stride[0] != stride[1]:
                print(f"Asymmetric stride found: {node.name}, {stride}")
            if isinstance(padding, (list, tuple)) and len(padding) >= 2 and padding[0] != padding[1]:
                print(f"Asymmetric padding found: {node.name}, {padding}")
            if isinstance(kernel, (list, tuple)) and len(kernel) >= 2 and kernel[0] != kernel[1]:
                print(f"Asymmetric kernel found: {node.name}, {kernel}")
