import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms, torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class resnet_(nn.Module):
  def __init__(self, pretrained=True):
    super(resnet_, self).__init__()
    resnet = models.resnet50(pretrained=pretrained)
    self.features = nn.Sequential(*list(resnet.children())[:-1])
    self.classifier = nn.Linear(resnet.fc.in_features, 10)

  def forward(self, x):
    x = self.features(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x

model = resnet_().to(device)

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
  n_correct = 0                         # total corrects
  n_samples = 0                         # total samples
  n_class_correct = [0] * 10            # correct per class
  n_class_samples = [0] * 10            # samples per class
  tp = [0] * 10
  fp = [0] * 10
  fn = [0] * 10
  tn = [0] * 10

  for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)

    _, predictions = torch.max(outputs, 1)

    n_samples += labels.size(0)                            #total samples in current batch
    n_correct += (predictions == labels).sum().item()      #total correct predictions in current batch

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


  for i in range(10):
    tn[i] = n_samples - tp[i] - fp[i] - fn[i]                    	# compute tn for each class

  accuracy = 100 * n_correct / n_samples
  print(f'Accuracy: {accuracy:.2f}%')

  for i in range(10):
    accuracy = 100 * (n_class_correct[i] / n_class_samples[i]) if n_class_samples[i] > 0 else 0
    precision = tp[i] / (tp[i] + fp[i] + 1e-10)
    specificity = tn[i] / (tn[i] + fp[i] + 1e-10)
    sensitivity = tp[i] / (tp[i] + fn[i] + 1e-10)

    print(f'Class: {classes[i]}')
    print(f'  Accuracy: {accuracy:.2f}%')
    print(f'  Precision: {precision:.2f}')
    print(f'  Specificity: {specificity:.2f}')
    print(f'  Sensitivity: {sensitivity:.2f}')

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
