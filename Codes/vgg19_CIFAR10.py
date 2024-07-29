import torch
import os
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VGG19(nn.Module):
  def __init__(self):
      super(VGG19, self).__init__()

      #convolutional layers
      #block 1:  2 conv (2)
      self.block1 = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(32),
        nn.Conv2d(32, 32, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(32)
      )

      #block 2:  2 conv (4)
      self.block2 = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64)
      )

      #block 3:  2 conv (8)
      self.block3 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2, 2)
      )

      #block 4:  2 conv (12)
      self.block4 = nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(256),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(256),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(256),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(256),
        nn.MaxPool2d(2, 2)
      )

      #block 5:  2 conv (16)
      self.block5 = nn.Sequential(
        nn.Conv2d(256, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(512),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(512),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(512),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(512),
        nn.MaxPool2d(2, 2)
      )

      # FC layer: 3 FC  (19)
      self.FC = nn.Sequential(
        nn.Linear(512 * 4 * 4, 1024),          # output size is 8192, so 512 * 4 * 4
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(1024, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(1024, 10)                        # CIFAR10 has 10 classes..
      )

  def forward(self, x):
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.block5(x)

    x = x.view(x.size(0), -1)                     # letting pytorch flatten
    x = self.FC(x)
    return x

model = VGG19().to(device)
weights_folder = './vgg19_weights'
os.makedirs(weights_folder, exist_ok=True)                  # create folder for saving the weights
weights_path = os.path.join(weights_folder, 'vgg19_weights.pth')
torch.save(model.state_dict(), weights_path)
#print(model)

#training
epochs = 100
batch_size = 16               # try changing it and see what happens
learning_rate = 0.0001

tform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
                            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#loading the CIFAR10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=tform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=tform)

#data loaders:
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = ('Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

loss_function = nn.CrossEntropyLoss()                                      # -1/N sumation i=1 to N sumation j=1 to M y_ij * log y_predicted_ij
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)      #also try AdaGrad instead of SGD..

total_steps = len(train_loader)                #getting the number of batches

for epoch in range(epochs):
  for i, (images, labels) in enumerate(train_loader):
    images = images.to(device)
    labels = labels.to(device)

    #forward
    outputs = model(images)
    loss = loss_function(outputs, labels)

    optimizer.zero_grad()    #zero the gradients
    loss.backward()          #backwaard
    optimizer.step()         #updating params

    if (i + 1) % 2000 == 0:
      print(f'Epoch: {epoch + 1}/{epochs}, step: {i + 1}/{total_steps}, loss: {loss.item():.4f}')

print("Finished Training..")

# model evaluation
with torch.no_grad():
  n_correct = 0
  n_samples = 0
  n_class_correct = [0] * 10
  n_class_samples = [0] * 10
  tp = [0] * 10
  fp = [0] * 10
  fn = [0] * 10
  tn = [0] * 10

  for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)
    _, predictions = torch.max(outputs, 1)

    n_samples += labels.size(0)
    n_correct += (predictions == labels).sum().item()

    for i in range(labels.size(0)):
      label = labels[i].item()
      pred = predictions[i].item()
      n_class_samples[label] += 1
      if label == pred:
        n_class_correct[label] += 1
        tp[label] += 1
      else:
        fp[pred] += 1
        fn[label] += 1

  # compute tn for each class
  for i in range(10):
    tn[i] = n_samples - tp[i] - fp[i] - fn[i]

  accuracy = 100 * n_correct / n_samples
  print(f'Accuracy: {accuracy:.2f}%')

  for i in range(10):
    accuracy = 100 * (n_class_correct[i] / n_class_samples[i]) if n_class_samples[i] > 0 else 0
    precision = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0
    specificity = tn[i] / (tn[i] + fp[i]) if (tn[i] + fp[i]) > 0 else 0
    sensitivity = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0

    print(f'Class: {classes[i]}')
    print(f'  Accuracy: {accuracy:.2f}%')
    print(f'  Precision: {precision:.2f}')
    print(f'  Specificity: {specificity:.2f}')
    print(f'  Sensitivity: {sensitivity:.2f}')

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
