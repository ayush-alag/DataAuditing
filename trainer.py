import csv
import os

import numpy as np
import torch
import torch.optim as optim
from torchvision import models
import wandb
import math

from arch import MLP, LocationModel

from torchinfo import summary

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class Trainer():
    def __init__(self, dataset, name, dim, criterion, max_epoch, mode='base', ckpt_name='', mixup=False, dropout_probability=0, expt="", plateau=-1):
        wandb.init(project=expt)
        wandb.config = {
            "dimensions": dim,
            "epochs": max_epoch,
            "dataset": dataset,
            "dropout": dropout_probability,
        }
        self.name = name
        self.mode = mode
        assert self.name in ['MNIST', 'COVIDx', 'Location']
        self.dataset = dataset
        self.train_loader = dataset.train_dataloader()
        self.test_loader = dataset.test_dataloader()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        if self.name == 'MNIST':
            self.model = MLP.MLP(28, dim, 10, dropout_probability).to(self.device)
        elif self.name == "Location":
            if self.mode == 'defense':
                self.model = LocationModel.DefenseMLP(dataset.input_shape, dataset.output_dims).to(self.device)
            else:
                self.model = LocationModel.LocationMLP(dataset.input_shape, dataset.output_dims, dropout_probability).to(self.device)
        else:
            self.model = models.resnet18(pretrained=False, num_classes=2).to(self.device)
        self.criterion = criterion
        self.max_epoch = max_epoch
        
        if self.name == 'MNIST':
            self.optimizer = optim.SGD(self.model.parameters(
        ), lr=self.get_lr(0), momentum=0.9, weight_decay=1e-4)
        elif self.name =='COVIDx':
            self.optimizer = optim.Adam(self.model.parameters(
        ), lr=self.get_lr(0), weight_decay=1e-7)
        elif self.name == 'Location':
            self.optimizer = optim.SGD(self.model.parameters(
        ), lr=self.get_lr(0), momentum=0.9, weight_decay=1e-4)

        self.plateau = plateau
        if plateau != -1:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=plateau)

        self.res = []
        self.ckpt_name = ckpt_name

        self.mixup = mixup
        self.alpha = 1
        self.expt = expt

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                output = self.model(images)
                class_loss = self.criterion(output, labels).item()

                if output.data.shape[1] == 1:
                    pred = np.where(np.asarray(output.data.detach().cpu()) >= 0.5, 1, 0)
                    correct += np.sum(pred == labels.detach().cpu().numpy().astype(int))
                else:
                    pred = output.data.max(1)[1]
                    correct += pred.eq(labels.view(-1)).sum().item()

                test_loss += class_loss

        test_loss /= len(self.test_loader)
        correct /= len(self.test_loader.dataset)

        return test_loss, correct

    def train(self, epoch):
        self.model.train()
        correct = 0
        train_loss = 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Training pass
            self.optimizer.zero_grad()

            if self.mixup:
                inputs, targets_a, targets_b, lam = mixup_data(images, labels,
                                                               self.alpha)
                inputs, targets_a, targets_b = inputs.to(self.device), targets_a.to(
                    self.device), targets_b.to(self.device)
                output = self.model(inputs)
                loss = mixup_criterion(
                    self.criterion, output, targets_a, targets_b, lam)
            else:
                output = self.model(images)
                loss = self.criterion(output, labels)
                
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

            if self.mixup == False:
                # print(output.data, labels)
                if output.data.shape[1] == 1:
                    pred = np.where(np.asarray(output.data.detach().cpu()) >= 0.5, 1, 0)
                    correct += np.sum(pred == labels.detach().cpu().numpy().astype(int))
                else:
                    pred = output.data.max(1)[1]
                    correct += pred.eq(labels.view(-1)).sum().item()

        # call at the epoch level
        if self.plateau != -1:
            self.scheduler.step(loss)
            
        wandb.log({"training loss": train_loss})
        wandb.watch(self.model)

        train_loss /= len(self.train_loader)
        correct /= len(self.train_loader.dataset)
        test_loss, test_acc = self.test()

        print('Train set: Average loss: {:.4f}, Average acc: {:.4f}\t Test set: Average loss: {:.4f}, Average acc: {:.4f}'.format(
            train_loss, correct, test_loss, test_acc))
        self.res.append([train_loss, correct, test_loss, test_acc])

    def get_lr(self, epoch):
        if self.name == 'MNIST':
            if (epoch+1) > 10:
                return 1e-2
            elif (epoch+1) >= 100:
                return 5e-3
            return 5e-2
        elif self.name == 'COVIDx':
            if (epoch+1) > 10:
                return 5e-6
            return 2e-5
        elif self.name == 'Location':
            return 0.1 * math.pow(0.1, epoch // 150)

    def save_ckpt(self, epoch):
        if epoch % 10 == 0 or epoch == self.max_epoch:
            print("saving model..")
            print(f'saves_new/{self.expt}/{self.name}/{self.mode}/{self.ckpt_name}training_epoch{epoch}.pkl')

            if not os.path.exists(f'saves_new/{self.expt}/{self.name}/{self.mode}'):
                os.makedirs(f'saves_new/{self.expt}/{self.name}/{self.mode}')

            torch.save(self.model.state_dict(
            ), f'saves_new/{self.expt}/{self.name}/{self.mode}/{self.ckpt_name}training_epoch{epoch}.pkl')

    def save_log(self):
        logname = f'saves_new/{self.expt}/{self.name}/{self.mode}/{self.ckpt_name}log.txt'

        with open(logname, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter='\t')
            writer.writerow(
                ["Train loss", "Train acc", "Test loss", "Test acc"])
            for row in self.res:
                writer.writerow(row)

    def run(self):
        print(self.expt)
        for epoch in range(1, self.max_epoch+1):
            print(f'Starting epoch {epoch}')
            self.train(epoch)
            self.save_ckpt(epoch)

        self.save_log()
