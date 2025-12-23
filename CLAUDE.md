# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Export YOLO-NAS models to ONNX format for use with Frigate NVR object detection.

## Setup

```bash
# Install dependencies (Python 3.9 required)
pip install super-gradients==3.7.1

# System deps for OpenCV
apt install -y libgl1 libglib2.0-0
```

**Note:** super-gradients has a broken URL issue. Apply fix:
```bash
# Replace deprecated sghub.deci.ai with working S3 URL
sed -i 's/sghub.deci.ai/sg-hub-nv.s3.amazonaws.com/' \
  $(python -c "import super_gradients; print(super_gradients.__path__[0])")/training/pretrained_models.py
sed -i 's/sghub.deci.ai/sg-hub-nv.s3.amazonaws.com/' \
  $(python -c "import super_gradients; print(super_gradients.__path__[0])")/training/utils/checkpoint_utils.py
```

## Export Model

```bash
python run.py
```

Outputs `yolo_nas_s.onnx` with:
- Input: 320x320
- Max predictions: 20
- Confidence threshold: 0.4
- Format: FLAT_FORMAT (Frigate compatible)

## Model Variants

Available in `Models`: `YOLO_NAS_S`, `YOLO_NAS_M`, `YOLO_NAS_L`

## Labels

`labels.txt` contains 80 COCO classes used by the model.
