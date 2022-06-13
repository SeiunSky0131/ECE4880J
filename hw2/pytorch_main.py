from sklearn.preprocessing import OneHotEncoder
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4
num_workers = 2

############# Dataset #############

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=num_workers)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=num_workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

############# Model #############
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

############# Loss function and optimizer #############
""" Set the learning rate """
lr = 0.01 # 1e-5, 1e-4, 1e-3, 1e-2, 1e-1,1

loss_type = 'MSE' # or 'l1'
if loss_type == 'crossentropy':
    criterion = nn.CrossEntropyLoss()
""" Change to a L1 Loss here """ 
if loss_type == 'l1':
    criterion = nn.L1Loss()
if loss_type == 'MSE':
    criterion = nn.MSELoss()

optimizer_type = 'SGD' # or 'Adam
if optimizer_type == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
""" Change to a Adam optimizer here """ 
if optimizer_type == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=lr)

############# Training #############

print('Start training')

""" Set the number of epochs"""
epoch_num = 4 # 1, 2, 4, 8, 16

for epoch in range(epoch_num):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        """ forward here""" 
        outputs = net(inputs)
        """ calculate loss here""" 
        # If we use crossentropy as loss, we do not need to keep two inputs in the same shape
        if loss_type == "crossentropy":
            loss = criterion(outputs, labels)
        else:
            # need to one-hot-encode the labels
            ohe_labels = np.zeros(np.shape(outputs))
            for j, label in enumerate(labels):
               ohe_labels[j][label] = 1
            ohe_labels = torch.tensor(ohe_labels.astype(float), dtype = torch.float32)
            # then we are able to calculate the loss using l1 or MSE
            loss = criterion(outputs, ohe_labels)
        """ backward loss here""" 
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

############# Testing #############

# prepare to count predictions for each class
correct = 0
total = 0
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                """ Fill in here""" 
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
