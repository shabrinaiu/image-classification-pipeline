import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import CocoDetection
from torchvision import transforms

class CustomCocoClassification(CocoDetection):
    def __getitem__(self, index):
        img, targets = super().__getitem__(index)
        if len(targets) == 0:
            label = 0  # background
        else:
            label = targets[0]['category_id']  # pick first object, or modify as needed
        return img, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = CustomCocoClassification(
    root='data/MobilCoco',  # Directory with images
    annFile='data/MobilCoco/annotations.json',
    transform=transform
)


def train():
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 12)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 3. Training Loop
    for epoch in range(10):  # Set your number of epochs
        model.train()
        total_loss = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataset):.4f}")

    # 4. Save Model
    torch.save(model.state_dict(), 'classifier_best_weights.pth')

train()