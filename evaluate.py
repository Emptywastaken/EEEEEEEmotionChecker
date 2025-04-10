import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

from model import EmotionRecognitionCNN
from dataset import FER2013Dataset

def evaluate_model(model_path, data_path, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionRecognitionCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataset = FER2013Dataset(base_path=data_path, batch_size=batch_size)
    _, test_loader = dataset.get_loaders()
    class_names = dataset.get_classes()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())

    cm = confusion_matrix(true_labels, predicted_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=class_names))

# RUN EVALUATION
if __name__ == "__main__":
    evaluate_model(
        model_path="emotion_model3_58.pth",
        data_path="archive"
    )
