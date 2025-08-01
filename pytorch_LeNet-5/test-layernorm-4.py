import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

DEVICE = 'cpu'

import numpy as np
from datetime import datetime

RANDOM_SEED = 42
LEARNING_RATE = 0.0005
BATCH_SIZE = 32
N_EPOCHS = 3

IMG_SIZE = 32
N_CLASSES = 10

def get_accuracy(model, data_loader, device, act_override=None):

    correct_pred = 0
    n = 0

    with torch.no_grad():
        for X, y_true in data_loader:
            X = X.to(device)
            y_true = y_true.to(device)

            y_prob = model(X, act_override=act_override)
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()

    return correct_pred.float() / n


def train(train_loader, model, criterion, optimizer, device):
    
    model.train()
    running_loss = 0

    for X, y_true in train_loader:

        optimizer.zero_grad()

        X = X.to(device)
        y_true = y_true.to(device)

        y_hat = model(X)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)

        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    
    return model, optimizer, epoch_loss

def validate(valid_loader, model, criterion, device):

    model.eval()
    running_loss = 0

    for X, y_true in valid_loader:

        X = X.to(device)
        y_true = y_true.to(device)

        y_hat = model(X)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)

    return model, epoch_loss

def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1):
    # ... 기존 그대로
    # metrics를 저장하기 위한 객체 설정
    best_loss = 1e10
    train_losses = []
    valid_losses = []
    for epoch in range(0, epochs):
        # training
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):
            
            model.train()
            train_acc = get_accuracy(model, valid_loader, device=device)
            
            model.eval()
            valid_acc = get_accuracy(model, valid_loader, device=device)
            
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')
            
            # ✅ 추가: quad_relu polynomial accuracies
            print("quad_relu polynomial accuracies:")
            for key, (poly_func, poly_str) in quad_relu_polynomials.items():
                acc = get_accuracy(model, valid_loader, device=device, act_override=poly_func)
                print(f"  {key}: {100 * acc:.2f}% | {poly_str}")

    return model, optimizer, (train_losses, valid_losses)



#MNIST DATASET

# transforms 정의하기

transforms = transforms.Compose([transforms.Resize((32, 32)),
                                 transforms.ToTensor()])

# data set 다운받고 생성하기
train_dataset = datasets.MNIST(root='mnist_data',
                               train=True,
                               transform=transforms,
                               download=True)

valid_dataset = datasets.MNIST(root='mnist_data',
                               train=False,
                               transform=transforms)

# data loader 정의하기
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False)


# Custom Activation Function


quad_relu_polynomials = {
    'quad_v1': (lambda x: x,
                "x"),
    'quad_v2': (lambda x: x ** 2,
                "x ** 2"),
    'quad_v3': (lambda x: 0.125 * x**2 + 0.5 * x + 0.25,
                "0.25 + 0.5 * x + 0.125 * x**2"),            
    'quad_v4': (lambda x: 0.125*x**2 + 0.5 * x,
                "0.5 * x + 0.125*x**2"),
    'quad_v5': (lambda x: 0.234606 + 0.5 * x + 0.204875 * x ** 2 - 0.0063896 * x ** 4,
                "0.234606 + 0.5 * x + 0.204875 * x ** 2 - 0.0063896 * x ** 4"),
    'quad_v6': (lambda x: 1.5522e-9 * x**18 - 1.7764e-7 * x**16 + 8.5114e-6 * x**14 - 2.2146e-4 * x**12 + 3.3960e-3 * x**10 - 3.1183e-2 * x**8 + 1.6707e-1 * x**6 - 4.9304e-1 * x**4 + 8.5369e-1 * x**2 + 0.5 * x + 3.8838e-2,
                "1.5522e-9 * x**18 - 1.7764e-7 * x**16 + ... + 0.5*x + 3.8838e-2")
}

# 선택
quad_relu, quad_relu_str = quad_relu_polynomials['quad_v2']



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # Convolution layers
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # LayerNorm for conv layers
        # normalized_shape = output channel dimension
        self.ln1 = nn.LayerNorm([6, 28, 28])   # conv1 output size before pooling (assuming input 32x32)
        self.ln2 = nn.LayerNorm([16, 10, 10])  # conv2 output size before pooling

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # LayerNorm for fc layers
        self.ln3 = nn.LayerNorm(120)
        self.ln4 = nn.LayerNorm(84)

    def forward(self, x, act_override=None):
        if act_override is not None:
            act = act_override
        elif self.training:
            act = F.relu
        else:
            act = quad_relu

        # Conv1 + LayerNorm + Activation + Pool
        x = self.conv1(x)
        #x = self.ln1(x)
        x = act(x)
        x = F.avg_pool2d(x, (2, 2))

        # Conv2 + LayerNorm + Activation + Pool
        x = self.conv2(x)
        #x = self.ln2(x)
        x = act(x)
        x = F.avg_pool2d(x, 2)

        # Flatten
        x = x.view(-1, self.num_flat_features(x))

        # FC1 + LayerNorm + Activation
        x = self.fc1(x)
        #x = self.ln3(x)
        x = act(x)

        # FC2 + LayerNorm + Activation
        x = self.fc2(x)
        #x = self.ln4(x)
        x = act(x)

        # FC3 (logits)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features




net = Net()
print(net)



torch.manual_seed(RANDOM_SEED)

model = net.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()


model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader,
                                    valid_loader, N_EPOCHS, DEVICE)

