# YOLO-NAS for Frigate (OpenVINO/Intel N100)

Export YOLO-NAS models optimized for Frigate NVR with OpenVINO on Intel N100.

## Setup

Use the included dev container with VS Code or GitHub Codespaces:

1. Install [VS Code](https://code.visualstudio.com/) and the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
2. Open this folder in VS Code
3. When prompted, click "Reopen in Container" (or run `Dev Containers: Reopen in Container` from the command palette)
4. Wait for the container to build and setup to complete

The container automatically installs Python 3.9, all dependencies, and applies necessary patches.

## Export Model

```bash
python run.py
```

Outputs `yolo_nas_int8.onnx` with INT8 quantization for optimal N100 performance.

## Frigate Config

```yaml
detectors:
  ov:
    type: openvino
    device: GPU

model:
  model_type: yolonas
  path: /config/model_cache/yolo_nas_int8.onnx
  width: 320
  height: 320
  input_tensor: nchw
  input_pixel_format: bgr
  labelmap_path: /config/model_cache/labels.txt
```

## Model Variants

Edit `run.py` to use different sizes: `YOLO_NAS_S` (fast), `YOLO_NAS_M` (balanced), `YOLO_NAS_L` (accurate).
