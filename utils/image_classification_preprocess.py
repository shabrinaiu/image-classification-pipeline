from torchvision import transforms
from PIL import Image
import numpy as np
import torch


def crop_and_preprocess_detections(
    image: np.ndarray,  # [H, W, C], dtype=uint8 or float32
    detections: list,  # List[Detection]
    classifier_transform,  # torchvision transform pipeline
    min_crop_size: int = 32,  # minimum size for crop
):
    """
    Args:
        image: np.ndarray [H, W, C] (RGB or BGR)
        detections: list of Detection objects (each has .box as [x_min, y_min, x_max, y_max])
        classifier_transform: torchvision.transforms.Compose
        min_crop_size: int, skip crops smaller than this
    Returns:
        batch: torch.Tensor [N, 3, H, W]
        valid_detections: list of Detection (aligned to batch)
    """
    preprocessed_crops = []
    valid_detections = []
    H, W, C = image.shape

    for det in detections:
        x_min, y_min, x_max, y_max = map(int, det.box)
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_max, W)
        y_max = min(y_max, H)
        crop_w, crop_h = x_max - x_min, y_max - y_min
        if crop_w < min_crop_size or crop_h < min_crop_size:
            continue

        # Crop image: [H, W, C]
        crop = image[y_min:y_max, x_min:x_max, :]
        # Convert to PIL image
        crop_pil = Image.fromarray(crop)
        # Apply transform (resize, normalize, etc.)
        processed = classifier_transform(crop_pil)
        preprocessed_crops.append(processed)
        valid_detections.append(det)

    if preprocessed_crops:
        batch = torch.stack(preprocessed_crops, dim=0)
    else:
        batch = torch.empty(0)
    return batch, valid_detections
