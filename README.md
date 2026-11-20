#EX-04 DL – DEVELOPING A NEURAL NETWORK CLASSIFICATION MODEL USING TRANSFER LEARNING

**Name:** CJ Rohit  
**Register Number:** 212224243005  
**Department:** Artificial Intelligence and Machine Learning

## AIM

To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## PROBLEM STATEMENT AND DATASET

**Problem Statement:**

To develop a deep learning-based image classification model using the pre-trained VGG19 architecture. The model is trained to classify images into two categories (Negative and Positive) using transfer learning.

**Dataset:**

The dataset contains two categories of chip images:
- Training Dataset
- Testing Dataset

The dataset is loaded using the `ImageFolder` function from PyTorch. Images are resized to 224 × 224 pixels and converted into tensors before training.

## NEURAL NETWORK MODEL

**VGG19 Architecture:**

VGG19 is a pre-trained convolutional neural network (CNN) containing 19 weight layers. It consists of convolutional layers, pooling layers, and fully connected layers.

**Model Architecture:**

Input Image (224 × 224 × 3)  
↓  
Convolutional Layers  
↓  
ReLU Activation  
↓  
Max Pooling Layers  
↓  
VGG19 Feature Extraction  
↓  
Flatten Layer  
↓  
Fully Connected Layers  
↓  
Modified Output Layer (1 Neuron)  
↓  
Binary Classification (Positive / Negative)

## DESIGN STEPS

### STEP 1:
Import required libraries and define image transformations.

### STEP 2:
Load training and testing datasets using ImageFolder.

### STEP 3:
Visualize sample images from the dataset.

### STEP 4:
Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.

### STEP 5:
Define loss function (BCEWithLogitsLoss) and optimizer (Adam). Train the model and plot the loss curve.

### STEP 6:
Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions.

## PROGRAM

**Name:** CJ Rohit  
**Register Number:** 212224243005

```python
import torch as t
import torch.nn as nn
import torch.optim as optim
import torchvision

from torchvision import datasets, models
from torchvision.models import VGG19_Weights
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

# Step 1: Image Transformations

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Step 2: Load Dataset

!unzip -qq ./chip_data.zip -d data

dataset_path = './data/dataset'

train_dataset = datasets.ImageFolder(
    root=f"{dataset_path}/train",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root=f"{dataset_path}/test",
    transform=transform
)

# Step 3: Visualize Sample Images

def show_sample_images(dataset, num_images=5):

    fig, axes = plt.subplots(1, num_images, figsize=(12, 4))

    for i in range(num_images):

        image, label = dataset[i]

        image = image.permute(1, 2, 0)

        image = image * np.array([0.229, 0.224, 0.225])
        image = image + np.array([0.485, 0.456, 0.406])
        image = np.clip(image, 0, 1)

        axes[i].imshow(image)
        axes[i].set_title(dataset.classes[label])
        axes[i].axis("off")

    plt.show()


print("Number of training samples:", len(train_dataset))

first_image, label = train_dataset[0]
print("Image shape:", first_image.shape)

print("Number of testing samples:", len(test_dataset))

first_image1, label = test_dataset[0]
print("Image shape:", first_image1.shape)

show_sample_images(train_dataset)

# Data Loaders

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# Step 4: Load Pre-trained VGG19 Model

model = models.vgg19(
    weights=VGG19_Weights.DEFAULT
)

# Modify Final Layer for Binary Classification

model.classifier[-1] = nn.Linear(
    model.classifier[-1].in_features, 1
)

# Freeze Feature Extraction Layers

for param in model.features.parameters():
    param.requires_grad = False

device = t.device("cuda" if t.cuda.is_available() else "cpu")

model = model.to(device)

# Step 5: Loss Function and Optimizer

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)

# Train the Model

def train_model(model, train_loader, test_loader, num_epochs=100):

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels.unsqueeze(1).float()
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_losses.append(
            running_loss / len(train_loader)
        )

        # Validation

        model.eval()
        val_loss = 0.0

        with t.no_grad():

            for images, labels in test_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels.unsqueeze(1).float()
                )

                val_loss += loss.item()

        val_losses.append(
            val_loss / len(test_loader)
        )

        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {train_losses[-1]:.4f}, "
            f"Validation Loss: {val_losses[-1]:.4f}"
        )

    # Plot Loss Curve

    plt.figure(figsize=(8, 6))

    plt.plot(
        range(1, num_epochs+1),
        train_losses,
        label="Train Loss",
        marker="o"
    )

    plt.plot(
        range(1, num_epochs+1),
        val_losses,
        label="Validation Loss",
        marker="s"
    )

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")

    plt.legend()
    plt.show()


# Start Training

train_model(model, train_loader, test_loader)


# Step 6: Model Evaluation

def test_model(model, test_loader):

    model.eval()

    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with t.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            probs = t.sigmoid(outputs)

            predicted = (probs > 0.5).int().cpu().numpy().flatten()

            total += labels.size(0)

            correct += (predicted == labels.numpy()).sum()

            all_preds.extend(predicted)
            all_labels.extend(labels.numpy())

    accuracy = correct / total

    print(f"Test Accuracy: {accuracy:.4f}")

    class_names = test_dataset.classes

    # Confusion Matrix

    cm = confusion_matrix(
        all_labels,
        all_preds
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.show()

    # Classification Report

    print("Classification Report:")

    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=class_names
        )
    )


test_model(model, test_loader)


# New Sample Data Prediction

def predict_image(model, image_index, dataset):

    model.eval()

    image, label = dataset[image_index]

    with t.no_grad():

        image_tensor = image.unsqueeze(0).to(device)

        output = model(image_tensor)

        probability = t.sigmoid(output)

        predicted = (probability > 0.5).int().item()

    class_names = dataset.classes

    image_to_display = image.permute(1, 2, 0).numpy()

    image_to_display = image_to_display * np.array(
        [0.229, 0.224, 0.225]
    )

    image_to_display = image_to_display + np.array(
        [0.485, 0.456, 0.406]
    )

    image_to_display = np.clip(
        image_to_display, 0, 1
    )

    plt.figure(figsize=(4, 4))

    plt.imshow(image_to_display)

    plt.title(
        f"Actual: {class_names[label]}\n"
        f"Predicted: {class_names[predicted]}"
    )

    plt.axis("off")
    plt.show()


# Display Predictions

predict_image(model, image_index=55, dataset=test_dataset)

predict_image(model, image_index=25, dataset=test_dataset)
```

## OUTPUT


## Training Loss, Validation Loss Vs Iteration Plot

<img width="762" height="516" alt="436130344-1d87ce9a-920d-4e4d-8b40-53e5eda1ede3" src="https://github.com/user-attachments/assets/2bb5f027-203e-4713-ac42-9eda66f418ff" />


## Confusion Matrix

<img width="677" height="607" alt="436130292-124fc61d-5797-4f05-b563-7a8ee582dfc5" src="https://github.com/user-attachments/assets/a95885dd-cca5-4441-b8b3-7e1ef3187ba8" />


## Classification Report

<img width="551" height="214" alt="444039121-c928c653-9f1e-4834-85e7-ec5bd823ff06" src="https://github.com/user-attachments/assets/24ff8f74-fb4a-4d5e-a1d8-36b284df293d" />


### New Sample Data Prediction

<img width="442" height="463" alt="444039309-cb02ea61-8425-434b-b584-5fa354e01bef" src="https://github.com/user-attachments/assets/d813599a-db18-4187-b516-6afe7fbeaaaa" />



<img width="454" height="483" alt="444039370-1164f9df-21b1-4758-9c68-6b008a987fe2" src="https://github.com/user-attachments/assets/3eea66f8-71b2-460e-9be6-163bf1ccec1e" />


## RESULT


Thus, the VGG19 model was successfully implemented using transfer learning for binary image classification. The model was trained and evaluated using the given dataset. Its performance was measured using test accuracy, confusion matrix, and classification report. The model also successfully predicted the classes of sample test images.
