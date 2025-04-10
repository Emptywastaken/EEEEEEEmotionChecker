import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import EmotionRecognitionCNN
from dataset import FER2013Dataset

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionRecognitionCNN().to(device)

# --- Dataset ---
dataset_path = "archive"
dataset = FER2013Dataset(base_path=dataset_path, batch_size=64)
train_loader, _ = dataset.get_loaders()  # We only use training loader here

# --- Training Setup ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
EPOCHS = 10

# --- Training Loop ---
for epoch in range(EPOCHS):
    model.train() # training mode
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    # wrap the training data in a progress bar (good view with real-time updates)
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        # clear old gradients before computing new ones in backpropagation
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Accuracy tracking
        _, predicted = torch.max(outputs, 1) #highest-scoring class
        correct_predictions += (predicted == labels).sum().item() # count correct predictions
        total_predictions += labels.size(0)

        # display current average loss and accuracy
        progress_bar.set_postfix(
            loss=running_loss / len(train_loader),
            accuracy=correct_predictions / total_predictions
        )

    # Compute the final average loss and accuracy
    avg_loss = running_loss / len(train_loader)
    avg_accuracy = correct_predictions / total_predictions
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

torch.save(model.state_dict(), "emotion_model2_XX.pth")
print("Model saved as 'emotion_model2.pth'")