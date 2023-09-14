import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, dropout_probability):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim_in*dim_in*3, dim_hidden)
        self.bn1 = nn.BatchNorm1d(dim_hidden)
        self.relu1 = nn.ReLU(dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_hidden)
        self.bn2 = nn.BatchNorm1d(dim_hidden)
        self.relu2 = nn.ReLU(dim_hidden)
        self.fc3 = nn.Linear(dim_hidden, dim_out)
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, x):
        x = x.view(-1, 28*28*3)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1)

class SmallMLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, dropout_probability):
        super(SmallMLP, self).__init__()
        self.fc1 = nn.Linear(dim_in*dim_in*3, dim_hidden)
        self.bn1 = nn.BatchNorm1d(dim_hidden)
        self.relu1 = nn.ReLU(dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_out)
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, x):
        x = x.view(-1, 28*28*3)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

class LeNet5(nn.Module):
    def __init__(self, num_classes, dropout_probability=0.0):
        super(LeNet5, self).__init__()
        
        self.num_classes = num_classes
        in_channels = 3

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.Tanh(),
            nn.Dropout(dropout_probability),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Dropout(dropout_probability),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probas = F.log_softmax(logits, dim=1)
        return probas