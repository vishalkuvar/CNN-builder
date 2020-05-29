# program to build CNNs

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from sklearn import metrics

class Network(nn.Module):

    def __init__(
        self,
        n_channels, #int
        image_size, #int
        conv_layers = [16, 32, 64], #list of int
        kernel = [5, 1], #list of int [kernel_size, kernel_stride]
        padding = 0, #int
        dropout_prob = 0.0, #float <=1
        fc_layers, [32, 16], #list of int
        classes, #list of int
        activation_layer = torch.nn.ReLU(), #torch.nn object
        learning_rate = 0.001, #float
        epochs = 10 #int
        ):

        super(Network, self).__init__()

        self.n_channels = n_channels
        self.image_size = image_size
        self.conv_layers = conv_layers
        self.kernel = kernel
        self.padding = padding
        self.dropout_prob = dropout_prob
        self.fc_layers = fc_layers
        self.classes = classes
        self.activation_layer = activation_layer
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(f'Running on {self.device}')


    def forward(self):

        layer = 1
        prev_channels = self.n_channels
        prev_dim = self.image_size

        self.net = nn.Sequential()

        if len(self.conv_layers) > 0:
            for conv in self.conv_layers:
                self.net.add_module(f'{layer}_convolutional', nn.Conv2d(prev_channels, conv, self.kernel[0], self.kernel[1], padding=self.padding))
                self.net.add_module(f'{layer}_activation', self.activation_layer)

                prev_dim = ((prev_dim - self.kernel[0] + 2*self.padding) // self.kernel[1]) + 1

                self.net.add_module(f'{layer}_pooling', nn.MaxPool2d(2, 2))
                self.net.add_module(f'{layer}_dropout', nn.Dropout(self.dropout_prob))

                prev_dim = ((prev_dim - 2) // 2) + 1
                prev_channels = conv
                layer += 1

        input_dim = prev_channels * prev_dim * prev_dim
        self.net.add_module('flatten', nn.Flatten())

        if len(self.fc_layers) > 0:

            for fc in self.fc_layers:
                self.net.add_module(f'{layer}_linear', nn.Linear(input_dim, fc))
                self.net.add_module(f'{layer}_activation', self.activation_layer)

                input_dim = fc
                layer += 1

        self.net.add_module('output', nn.Linear(input_dim, len(self.classes)))
        self.net.add_module(f'output_activation', torch.nn.Softmax())

        self.net.to(self.device)


    def train(self, trainloader): #trainloader --> dataloader object

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        train_loss = 0.0

        start = time.time()
        for epoch in range(self.epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            return_loss = 0.0
            total = 0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                total += 1

            train_loss = running_loss / total
            print('Epoch: %d, Loss: %.3f' %(epoch + 1, train_loss))

        print('Finished Training')
        end = time.time()

        return (end - start), train_loss


    def use(self, testloader): #testloader --> dataloader object

        criterion = nn.CrossEntropyLoss()
        correct = 0
        total = 0
        test_labels = []
        outputs_list = []
        with torch.no_grad():
            running_loss = 0.0
            for i, data in enumerate(testloader, 0):
                images, labels = data[0].to(self.device), data[1].to(self.device)
                test_labels.append(labels[0].item())
                outputs = self.net(images)
                outputs_list.append(outputs[0].cpu().numpy())
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = correct / total
            test_loss = running_loss / total
            test_labels = np.array(test_labels)
            confusion_matrix = metrics.confusion_matrix(test_labels, np.argmax(outputs_list, axis=1))
            f1_micro_score = metrics.f1_score(test_labels, np.argmax(outputs_list, axis=1), average='micro')
            f1_macro_score = metrics.f1_score(test_labels, np.argmax(outputs_list, axis=1), average='macro')
            f1_weighted_score = metrics.f1_score(test_labels, np.argmax(outputs_list, axis=1), average='weighted')

        return accuracy, test_loss, confusion_matrix, f1_micro_score, f1_macro_score, f1_weighted_score
