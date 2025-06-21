import torch
from torchvision import transforms

from classification.resnet import ResNetClassifier
from detection.yolo import YOLODetector
from utils.image_classification_preprocess import crop_and_preprocess_detections
from utils.video_preprocess import batch_generator

VIDEO_PATH = './data/input.mp4'
MODEL_INPUT_SIZE = (224, 224)  # Example model input size (height, width)
BATCH_SIZE = 8                 # Number of frames per batch

batch = batch_generator(VIDEO_PATH, BATCH_SIZE)

detection_model = YOLODetector()

detection_results = detection_model.detect(batch)

classifier_model = ResNetClassifier()

classifier_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # ResNet50 expects 224x224 input
        transforms.ToTensor(),  # Converts PIL image to [0,1] tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet means
            std=[0.229, 0.224, 0.225],  # ImageNet stds
        ),
    ]
)

for image, detection_list in zip(batch, detection_results):
    classification_batch, valid_dets = crop_and_preprocess_detections(
        image, detection_list, classifier_transform
    )

    if classification_batch.shape[0] > 0:
        with torch.no_grad():
            prediction = classifier_model.predict(
                classification_batch
            ) 
            print(prediction)
