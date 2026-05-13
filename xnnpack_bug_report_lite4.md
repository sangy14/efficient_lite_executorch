# Bug Report: XNNPACK INT8 Delegate fails on EfficientNet/EfficientDet-Lite4 with Shape Mismatch and StaticConstantPad Errors

## 🐛 Describe the bug
When attempting to export and execute the `EfficientDet-Lite4` model (using the `EfficientNet-Lite4` backbone) with the **XNNPACK** backend in **INT8** (per-channel symmetric quantization), the inference fails during execution.

The failure occurs due to a mismatch at the delegate boundary. Even after addressing serialization issues with non-finite values, the XNNPACK delegate produces incorrectly sized output tensors, leading to a runtime crash.

**Note:** The model exports and executes flawlessly when using the FP32 XNNPACK backend. The issue is strictly isolated to the INT8 quantized path.

---

## 💻 Environment
* **OS:** macOS (Apple Silicon / ARM64)
* **Python version:** 3.11.14
* **ExecuTorch Version:** (Current stable/nightly)
* **Model:** `tf_efficientdet_lite4` (from `effdet` library, with 640x640 input)
* **Quantization:** `get_symmetric_quantization_config(is_per_channel=True)` via `XNNPACKQuantizer`

---

## 📋 Error Logs

### Runtime Shape Mismatch (During inference)
When running the model exported via `export_xnnpack.py`, the execution crashes with a static tensor resize error at the 20x20 spatial scale (Stage 7 output for 640x640 inputs):
```text
[tensor_impl.cpp:110] Attempted to resize a static tensor. Expected shape (1, [CHANNELS], 20, 20), but received (1, [CHANNELS], [INCORRECT_DIM], 20).
[XNNExecutor.cpp:239] Failed to resize output tensor for XNNExecutor
[method.cpp:1426] CALL_DELEGATE execute failed at instruction 0: 0x10
```

---

## 🔍 Implementation Details (from `export_xnnpack.py`)

### 1. Handling `-inf` Padding Values
EfficientDet utilizes `-inf` values for padding/masking in several stages. We found that XNNPACK (and Flatbuffer serialization) does not support `-inf` constant values in the INT8 path. To resolve this, we implemented monkey-patching of `torch.nn.functional.pad` and `timm` padding layers to replace `-inf` with a large finite constant (`-10000.0`):

```python
# export_xnnpack.py
_original_pad = torch.nn.functional.pad
def _patched_pad(input, pad, mode='constant', value=None):
    if mode == 'constant' and value is not None and (value == float('-inf') or value == -float('inf')):
        value = -10000.0
    return _original_pad(input, pad, mode=mode, value=value)
torch.nn.functional.pad = _patched_pad
```

### 2. Boundary Condition
In our export pipeline, we use the standard `XnnpackPartitioner`. Because XNNPACK often rejects specific padding configurations (like "Same" padding with certain strides), these padding nodes are left on the CPU while the subsequent convolutions are delegated to XNNPACK. This creates a delegate-to-CPU boundary where the XNNPACK INT8 convolution kernel appears to be miscalculating the output dimensions or strides, resulting in the shape mismatch error shown above.

---

## 🔍 Root Cause Analysis & Suspects
1. **Depthwise Kernel Bug:** The runtime shape mismatch suggests that the INT8 XNNPACK Depthwise Convolution kernel is miscalculating output strides or buffer sizes when operating on quantized tensors at this specific scale.
2. **Serialization Constraint:** The requirement to patch `-inf` indicates that the XNNPACK delegate or its Flatbuffer schema may not be correctly handling non-finite constants for quantized tensors.

---

## 🛠️ Steps to Reproduce
1. Load `tf_efficientdet_lite4`.
2. Apply `XNNPACKQuantizer` with per-channel symmetric weights.
3. Patch `F.pad` to replace `-inf` with `-10000.0`.
4. Lower to Edge IR using `XnnpackPartitioner`.
5. Run inference. The model will crash with `Attempted to resize a static tensor`.

---

## 🎯 Expected behavior
The INT8 quantized model should execute successfully, producing the correct tensor shape outputs as the FP32 model.
