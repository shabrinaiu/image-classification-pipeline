from dataclasses import dataclass
from typing import List
import torch


@dataclass
class Detection:
    # [x_min, y_min, x_max, y_max]
    box: List[torch.Tensor]
    confidence: float
    class_name: str


@dataclass
class Classification:
    confidence: float
    class_name: str
