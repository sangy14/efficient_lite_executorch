import torch
import os
import sys
from pathlib import Path

try:
    from executorch.extension.pybindings.portable_lib import _load_for_executorch
except ImportError:
    print("ExecuTorch not available.")
    sys.exit(1)

def inspect_pte(model_path):
    print(f"\nInspecting: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: {model_path} does not exist.")
        return

    try:
        program = _load_for_executorch(str(model_path))
        method_name = "forward"
        meta = program.method_meta(method_name)
        
        print(f"Method: {method_name}")
        print(f"Number of inputs: {meta.num_inputs()}")
        for i in range(meta.num_inputs()):
            input_meta = meta.input_tensor_meta(i)
            print(f"  Input[{i}]: shape={input_meta.sizes()}, dtype={input_meta.dtype()}")
            
        print(f"Number of outputs: {meta.num_outputs()}")
        for i in range(meta.num_outputs()):
            output_meta = meta.output_tensor_meta(i)
            print(f"  Output[{i}]: shape={output_meta.sizes()}, dtype={output_meta.dtype()}")
            
    except Exception as e:
        print(f"Error inspecting model: {e}")

if __name__ == "__main__":
    model_dir = Path("efficientdet_lite4_448_models")
    if model_dir.exists():
        for pte_file in sorted(model_dir.glob("*.pte")):
            inspect_pte(pte_file)
    else:
        print(f"Directory {model_dir} not found.")
