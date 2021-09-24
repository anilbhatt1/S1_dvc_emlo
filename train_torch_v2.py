import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, models, transforms
import numpy as np
import os
import time
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import gc
torch.cuda.empty_cache()

print('Imports successful')

data_dir = os.path.join('d', os.sep, 'AI_D_Drive','emlo','s1_dvc', 'example-versioning','data')

data_str = ["train", "validation"]
data_transform = transforms.Compose([transforms.Resize([32, 32]), transforms.ToTensor()])
img_datasets = {
    x: datasets.ImageFolder(root=data_dir+'/{}'.format(x), transform=data_transform)
    for x in data_str
}
dataloader = {
    x: torch.utils.data.DataLoader(dataset=img_datasets[x], batch_size=16, shuffle=True)
    for x in data_str
}

X_example, y_example = next(iter(dataloader["train"]))

print("X_example's number:{}".format(len(X_example)))
print("y_example's number:{}".format(len(y_example)))
index_classes = img_datasets["train"].class_to_idx
print(index_classes)
example_classes = img_datasets["train"].classes
print(example_classes)


img = torchvision.utils.make_grid(X_example)
img = img.numpy().transpose([1, 2, 0])
print([example_classes[i] for i in y_example])
plt.imshow(img)
plt.show()

print('CUDA Available : ',torch.cuda.is_available())

class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()

        # CONVOLUTION BLOCK 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )  # in = 32x32x3 , out = 30x30x32, RF = 3

        # CONVOLUTION BLOCK 2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )  # in = 30x30x32 , out = 28x28x32, RF = 5

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) #    in = 28x28x32 , out = 14x14x32, RF = 6

        # CONVOLUTION BLOCK 3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )  # in = 14x14x32 , out = 12x12x32, RF = 10

        # CONVOLUTION BLOCK 4
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )  # in = 12x12x32 , out = 10x10x32, RF = 14

        # CONVOLUTION BLOCK 5
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )  # in = 10x10x32 , out = 8x8x32, RF = 18

        # CONVOLUTION BLOCK 6
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )  # in = 8x8x32 , out = 6x6x32, RF = 22

        # OUTPUT BLOCK
        self.Gap1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        )  # in = 6x6x16 , out = 1x1x16, RF = 32
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=(1, 1), padding=0, bias=False)
        )  # in = 1x1x16 , out = 1x1x1
        self.linear = nn.Linear(in_features=16, out_features=2, bias=False)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.fc1(self.Gap1(x))
        x = x.view(-1, 2)
        return F.log_softmax(x, dim=-1)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Device to be used :', device)
model = CNNModel().to(device)


# # class for Calculating and storing training losses and training accuracies of model for each batch per epoch ##
class Train_loss:

    def train_loss_calc(self, model, device, train_loader, optimizer, epoch, factor):

        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.epoch = epoch
        self.factor = factor

        model.train()
        pbar = tqdm(train_loader)  # Wrapping train_loader in tqdm to show progress bar for each epoch while training

        correct = 0
        total = 0
        train_losses = 0
        train_acc = 0

        for batch_idx, data in enumerate(pbar, 0):

            images, labels = data

            images, labels = images.to(device), labels.to(device)  # Moving images and correspondig labels to GPU
            optimizer.zero_grad()  # Zeroing out gradients at start of each batch so that backpropagation won't take accumulated value
            labels_pred = model(images)  # Calling CNN model to predict the images
            loss = F.nll_loss(labels_pred,
                              labels)  # Calculating Negative Likelihood Loss by comparing prediction vs ground truth

            # Applying L1 regularization to the training loss calculated
            L1_criterion = nn.L1Loss(size_average=None, reduce=None, reduction='mean')
            reg_loss = 0
            for param in model.parameters():
                zero_tensor = torch.rand_like(param) * 0  # Creating a zero tensor with same size as param
                reg_loss += L1_criterion(param, zero_tensor)
            loss += factor * reg_loss

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Calculating accuracies
            labels_pred_max = labels_pred.argmax(dim=1,
                                                 keepdim=True)  # Getting the index of max log probablity predicted by model
            correct += labels_pred_max.eq(
                labels.view_as(labels_pred_max)).sum().item()  # Getting count of correctly predicted
            total += len(images)  # Getting count of processed images
            train_acc_batch = (correct / total) * 100
            pbar.set_description(
                desc=f'Train Loss = {loss.item()} Batch Id = {batch_idx} Train Accuracy = {train_acc_batch:0.2f}')

        train_acc = train_acc_batch  # To capture only final batch accuracy of an epoch
        train_losses = loss  # To capture only final batch loss of an epoch

        return train_losses, train_acc

class Test_loss:

    def test_loss_calc(self, model, device, test_loader):
        self.model = model
        self.device = device
        self.test_loader = test_loader

        model.eval()

        correct = 0
        total = 0
        test_loss = 0
        test_acc = 0
        correct_0, correct_1, total_0, total_1 = 0, 0, 0, 0

        with torch.no_grad():  # For test data, we won't do backprop, hence no need to capture gradients
            for images, labels in test_loader:

                images, labels = images.to(device), labels.to(device)
                labels_pred = model(images)
                test_loss += F.nll_loss(labels_pred, labels, reduction='sum').item()
                labels_pred_max = labels_pred.argmax(dim=1, keepdim=True)
                correct += labels_pred_max.eq(labels.view_as(labels_pred_max)).sum().item()
                total += labels.size(0)

                labels_lst = labels.tolist()
                labels_pred_max_lst = labels_pred_max.tolist()

                for i in range(len(labels_lst)):
                    if labels_lst[i] == 0:
                        if labels_lst[i] == labels_pred_max_lst[i][0]:
                            correct_0 += 1
                        total_0 += 1
                    elif labels_lst[i] == 1:
                        if labels_lst[i] == labels_pred_max_lst[i][0]:
                            correct_1 += 1
                        total_1 += 1

            test_loss /= total  # Calculating overall test loss for the epoch
            test_acc = (correct / total) * 100
            test_0_acc = (correct_0 / total_0) * 100
            test_1_acc = (correct_1 / total_1) * 100

            print(
                f'Test set: Avg loss: {test_loss: .4f}, Test Acc: {test_acc:.2f}, Correct : {correct}, Total : {total}\n')
            print(f'Test set: Class 0 acc: {test_0_acc: .4f}, Correct : {correct_0}, Total : {total_0}\n')
            print(f'Test set: Class 1 acc: {test_1_acc: .4f}, Correct : {correct_1}, Total : {total_1}\n')

        return test_loss, test_acc, test_0_acc, test_1_acc

train_loss = Train_loss()
test_loss  = Test_loss()
EPOCH = 40
L1_factor=0.0005
optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.8, weight_decay = 0)

train_loss_ov, train_acc_ov, test_loss_ov, test_acc_ov, test_acc_ov0, test_acc_ov1 = [], [], [], [], [], []
for epoch in range(EPOCH ):
    print(f'Current epoch: {epoch + 1}/{EPOCH}')
    train_losses, train_acc = train_loss.train_loss_calc(model,device,train_loader=dataloader['train'],optimizer=optimizer,epoch=EPOCH,factor=L1_factor)
    test_losses, test_acc, test_0_acc, test_1_acc   = test_loss.test_loss_calc(model,device,test_loader=dataloader['validation'])
    train_loss_ov.append(train_losses)
    train_acc_ov.append(train_acc)
    test_loss_ov.append(test_losses)
    test_acc_ov.append(test_acc)
    test_acc_ov0.append(test_0_acc)
    test_acc_ov1.append(test_1_acc)

header = ['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'test_acc_class_0', 'test_acc_class_1']
with open('./metrics.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for i in range(len(train_loss_ov)):
        data = []
        data.append(i)
        data.append(round(train_loss_ov[i].item(), 4))
        data.append(round(train_acc_ov[i], 4))
        data.append(round(test_loss_ov[i], 4))
        data.append(round(test_acc_ov[i], 4))
        data.append(round(test_acc_ov0[i], 4))
        data.append(round(test_acc_ov1[i], 4))
        writer.writerow(data)
f.close()

torch.save(model.state_dict(), './model.pth')