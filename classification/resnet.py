from classification import BaseClassifier
import torch
import torchvision.transforms as transforms
from torchvision import models


class ResNetClassifier(BaseClassifier):
    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(self.class_names))  # 8 car types
        self.model.load_state_dict(
            torch.load("/content/classifier_best_weights.pth")
        )  # Upload your weights
        self.model.eval()

    def predict(self, batch):
        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.nn.functional.softmax(logits, dim=1)
            predicted = torch.argmax(probs, dim=1)  # [batch_size]
            confidences = probs[range(len(predicted)), predicted]  # [batch_size]

        # Return a list of Classification objects, one per image in the batch
        return [
            self.__output_prediction(pred, conf)
            for pred, conf in zip(predicted.tolist(), confidences.tolist())
        ]
