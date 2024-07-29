import pandas as pd
import pydicom
import os
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, Subset
import torch
import torch.nn as nn
import PIL.Image as Image
import numpy as np
from sklearn.utils import resample
from torch.optim.lr_scheduler import ReduceLROnPlateau

extract_path = 'E:/IISER/INbreast Release 1.0/AllDICOMs'

dicoms = extract_path

# load the BIRADS CSV file
birads_file_path = 'E:/IISER/INbreast Release 1.0/BIRADS.csv'
birads_data = pd.read_csv(birads_file_path)

class_counts = {0: 65, 1: 224, 2: 23, 3: 43, 4: 47, 5: 8}
classes = ['1', '2', '3', '4', '5', '6']

# INbreast dataset class
class INbreast(Dataset):
    def __init__(self, dicoms, labels, transform=None):
        self.dicoms = dicoms
        self.labels = labels
        if os.path.exists(dicoms):
            self.images = [f for f in os.listdir(dicoms) if f.endswith('.dcm')]
            print(f"Found {len(self.images)} DICOM files in {dicoms}")
        self.label_mapping = {int(img.split('_')[0]): idx for idx, img in enumerate(self.images)}
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.dicoms, self.images[idx])
        dicom_image = pydicom.dcmread(img_name)
        image = dicom_image.pixel_array.astype(np.float32)
        image = Image.fromarray(image).convert('L')
        image = image.convert('RGB')

        patient_id = int(self.images[idx].split('_')[0])
        label = self.labels[self.labels['PatientID'] == patient_id]['BIRADS'].values[0] - 1

        if self.transform:
            if label == 1:                                      # no augmentation for class 2
                image = self.transform(image)
            else:
                image = train_transform(image)

        return image, label

class resnet_(nn.Module):
    def __init__(self, pretrained=True):
        super(resnet_, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Linear(resnet.fc.in_features, 6)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224, scale=(0.8, 2.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.4, hue=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

initial_dataset = INbreast(dicoms=dicoms, labels=birads_data, transform=val_transform)

# separate images by class
class_images = {i: [] for i in range(6)}
for idx in range(len(initial_dataset)):
    _, label = initial_dataset[idx]
    class_images[label].append(idx)

# resample
balanced_indices = []
max_samples = max(class_counts.values())

for cls, images in class_images.items():
    if len(images) < max_samples:
        resampled_indices = resample(images, replace=True, n_samples=max_samples, random_state=42)
    else:
        resampled_indices = images              # use original indices if no resampling is needed
    balanced_indices.extend(resampled_indices)

# balanced dataset:
balanced_dataset = Subset(initial_dataset, balanced_indices)
dataset = balanced_dataset

indices = list(range(len(dataset)))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

total_correct = 0
total_samples = 0
num_epochs = 100

train_size = int(0.8 * len(balanced_dataset))
val_size = len(balanced_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(balanced_dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

model = resnet_(pretrained=True).to(device)

weights_folder = './resnet_weights'
os.makedirs(weights_folder, exist_ok=True)                  # create folder for saving the weights
weights_path = os.path.join(weights_folder, 'resnet_weights.pth')

optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-6)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
class_counts_list = [class_counts.get(i, 0) for i in range(6)]
class_weights = torch.tensor([max(class_counts) / c if c > 0 else 0.0 for c in class_counts], dtype=torch.float).to(device)
loss_function = nn.CrossEntropyLoss()

def calculate_metrics(true_positives, false_positives, false_negatives, true_negatives):
    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    specificity = true_negatives / (true_negatives + false_positives + 1e-10)
    return precision, recall, specificity

# training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    num_batches_per_epoch = len(train_loader)

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 17 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{num_batches_per_epoch}], Loss: {loss.item():.4f}')

    epoch_loss = running_loss / num_batches_per_epoch
    scheduler.step(epoch_loss)

    # validation loop
    model.eval()
    val_loss = 0.0
    n_correct = 0
    n_samples = 0
    n_class_correct = [0] * 6
    n_class_samples = [0] * 6

    true_positives = [0] * 6
    false_positives = [0] * 6
    false_negatives = [0] * 6
    true_negatives = [0] * 6

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)
            val_loss += loss.item()

            _, predictions = torch.max(outputs, 1)
            n_correct += (predictions == labels).sum().item()
            n_samples += labels.size(0)

            for i in range(len(labels)):
                label = labels[i].item()
                pred = predictions[i].item()
                if label == pred:
                    n_class_correct[label] += 1
                    true_positives[label] += 1
                else:
                    false_positives[pred] += 1
                    false_negatives[label] += 1
                    for j in range(6):
                        if j != label and j != pred:
                            true_negatives[j] += 1

                n_class_samples[label] += 1

    total_correct += n_correct
    total_samples += n_samples

    val_loss /= len(val_loader)
    val_accuracy = 100 * n_correct / n_samples
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

for i in range(6):
    if n_class_samples[i] > 0:
        precision, recall, specificity = calculate_metrics(true_positives[i], false_positives[i], false_negatives[i], true_negatives[i])
        class_accuracy = n_class_correct[i] / n_class_samples[i]
        print(f'Class {classes[i]} - Accuracy: {class_accuracy:.2f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Specificity: {specificity:.4f}')
    else:
        print(f'No samples for class {classes[i]}')

overall_accuracy = (total_correct / total_samples) * 100
print(f'Overall Accuracy: {overall_accuracy:.4f}')

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
