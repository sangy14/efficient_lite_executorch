import torch
import numpy as np
from executorch.runtime import Runtime
import os

def test_pte_inference(pte_path):
    print(f"Testing inference with {pte_path}")
    if not os.path.exists(pte_path):
        print(f"Error: {pte_path} does not exist")
        return

    try:
        from executorch.extension.pybindings.portable_lib import _load_for_executorch
    except ImportError:
        print("Error: executorch extension not found")
        return

    # Load the program
    program = _load_for_executorch(pte_path)
    
    # Create dummy input (1, 3, 640, 640)
    input_data = torch.randn(1, 3, 640, 640)
    
    # Execute
    print("Executing forward pass...")
    try:
        outputs = program.forward([input_data])
        if isinstance(outputs, (list, tuple)):
            print(f"Success! Output shapes: {[o.shape for o in outputs]}")
        else:
            print(f"Success! Output shape: {outputs.shape}")
    except Exception as e:
        print(f"Failed to execute: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    pte_path = "efficientdet_lite4_exported_models/efficientdet_lite4_int8_static_perchannel_with_nms.pte"
    test_pte_inference(pte_path)
