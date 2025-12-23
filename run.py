from super_gradients.common.object_names import Models
from super_gradients.conversion import DetectionOutputFormatMode, ExportQuantizationMode
from super_gradients.training import models

model = models.get(Models.YOLO_NAS_M, pretrained_weights="coco")

model.export(
    "yolo_nas_int8.onnx",
    output_predictions_format=DetectionOutputFormatMode.FLAT_FORMAT,
    max_predictions_per_image=5,
    num_pre_nms_predictions=300,
    confidence_threshold=0.5,
    input_image_shape=(320, 320),
    quantization_mode=ExportQuantizationMode.INT8,
)
