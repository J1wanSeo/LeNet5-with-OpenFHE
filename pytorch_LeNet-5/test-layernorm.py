import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

from utils_approx import ReLU_maker

DEVICE = 'cpu'

import numpy as np
from datetime import datetime

import os

RANDOM_SEED = 42
LEARNING_RATE = 0.0005
BATCH_SIZE = 32
N_EPOCHS = 10

IMG_SIZE = 32

def select_activation():
    print("Select Activation function:")
    print("0: linear (x)")
    print("1: square (x^2)")
    print("2: CryptoNet (0.25 + 0.5 * x + 0.125 * x^2)")
    # print("3: quad_v4 (0.5 * x + 0.125 * x^2)")
    print("3: quad (0.234606 + 0.5 * x + 0.204875 * x^2 - 0.0063896 * x^4)")
    print("4: student (custom polynomial)")
    print("5: ReLU-maker (utils_approx.ReLU_maker)")
    choice = input("Enter number (0~5): ")
    try:
        choice_int = int(choice)
        if choice_int not in range(6):
            raise ValueError
    except:
        print("Invalid input, defaulting to CryptoNet")
        choice_int = 2

    key_list = list(quad_relu_polynomials.keys())
    selected_key = key_list[choice_int]
    print(f"Selected activation: {selected_key} - {quad_relu_polynomials[selected_key][1]}")
    return quad_relu_polynomials[selected_key]


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
        if epoch in [4, 9]:        
            print("quad_relu polynomial accuracies:")
            model_save(epoch)
            for key, (poly_func, poly_str) in quad_relu_polynomials.items():
                acc = get_accuracy(model, valid_loader, device=device, act_override=poly_func)
                print(f"  {key}: {100 * acc:.2f}% | {poly_str}")

    return model, optimizer, (train_losses, valid_losses)

# Weight/Bias Extraction

def model_save(epoch):
    # 저장 폴더 생성
    folder_name = f"lenet_weights_epoch({epoch+1})"
    os.makedirs(folder_name, exist_ok=True)

    # conv1 weight, bias
    conv1_w = model.conv1.weight.detach().cpu().numpy()
    conv1_b = model.conv1.bias.detach().cpu().numpy()
    np.save(f"lenet_weights_epoch({epoch+1})"+"/conv1_weight.npy", conv1_w)
    np.save(f"lenet_weights_epoch({epoch+1})"+"/conv1_bias.npy", conv1_b)

    # conv2 weight, bias
    conv2_w = model.conv2.weight.detach().cpu().numpy()
    conv2_b = model.conv2.bias.detach().cpu().numpy()
    np.save(f"lenet_weights_epoch({epoch+1})"+"/conv2_weight.npy", conv2_w)
    np.save(f"lenet_weights_epoch({epoch+1})"+"/conv2_bias.npy", conv2_b)

    # conv3 weight, bias
    # conv3_w = model.conv3.weight.detach().cpu().numpy()
    # conv3_b = model.conv3.bias.detach().cpu().numpy()
    # np.save(f"lenet_weights_epoch({epoch+1})"+"/conv3_weight.npy", conv3_w)
    # np.save(f"lenet_weights_epoch({epoch+1})"+"/conv3_bias.npy", conv3_b)
    
    # fc1 weight, bias
    fc1_w = model.fc1.weight.detach().cpu().numpy()
    fc1_b = model.fc1.bias.detach().cpu().numpy()
    np.save(f"lenet_weights_epoch({epoch+1})"+"/fc1_weight.npy", fc1_w)
    np.save(f"lenet_weights_epoch({epoch+1})"+"/fc1_bias.npy", fc1_b)

    # fc2 weight, bias
    fc2_w = model.fc2.weight.detach().cpu().numpy()
    fc2_b = model.fc2.bias.detach().cpu().numpy()
    np.save(f"lenet_weights_epoch({epoch+1})"+"/fc2_weight.npy", fc2_w)
    np.save(f"lenet_weights_epoch({epoch+1})"+"/fc2_bias.npy", fc2_b)

    # fc3 weight, bias
    fc3_w = model.fc3.weight.detach().cpu().numpy()
    fc3_b = model.fc3.bias.detach().cpu().numpy()
    np.save(f"lenet_weights_epoch({epoch+1})"+"/fc3_weight.npy", fc3_w)
    np.save(f"lenet_weights_epoch({epoch+1})"+"/fc3_bias.npy", fc3_b)


    # === BatchNorm 저장 ===
    def save_bn(layer, prefix):
        gamma = layer.weight.detach().cpu().numpy()
        beta = layer.bias.detach().cpu().numpy()
        mean = layer.running_mean.detach().cpu().numpy()
        var = layer.running_var.detach().cpu().numpy()

        np.save(f"{folder_name}/{prefix}_bn_gamma.npy", gamma)
        np.save(f"{folder_name}/{prefix}_bn_beta.npy", beta)
        np.save(f"{folder_name}/{prefix}_bn_mean.npy", mean)
        np.save(f"{folder_name}/{prefix}_bn_var.npy", var)

    save_bn(model.bn1, "conv1")
    save_bn(model.bn2, "conv2")
    save_bn(model.bn3, "fc1")
    save_bn(model.bn4, "fc2")  # fc1 뒤에도 BN 붙어있다면

    print("All weights and biases saved to ./lenet_weights_epoch()")


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
    # 'quad_v4': (lambda x: 0.125*x**2 + 0.5 * x,
    #             "0.5 * x + 0.125*x**2"),
    'quad_v5': (lambda x: 0.234606 + 0.5 * x + 0.204875 * x ** 2 - 0.0063896 * x ** 4,
                "0.234606 + 0.5 * x + 0.204875 * x ** 2 - 0.0063896 * x ** 4"),
    # 'quad_v6': (lambda x: 1.5522e-9 * x**18 - 1.7764e-7 * x**16 + 8.5114e-6 * x**14 - 2.2146e-4 * x**12 + 3.3960e-3 * x**10 - 3.1183e-2 * x**8 + 1.6707e-1 * x**6 - 4.9304e-1 * x**4 + 8.5369e-1 * x**2 + 0.5 * x + 3.8838e-2,
    #             "1.5522e-9 * x**18 - 1.7764e-7 * x**16 + ... + 0.5*x + 3.8838e-2"),
    'ReLU-maker' : (lambda x: ReLU_maker({'type':'proposed','alpha':13,'B':50})(x), "ReLU Maker with alpha==13")

    }

# 선택
quad_relu, quad_relu_str = select_activation()



class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        # Convolution layers
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # self.conv3 = nn.Conv2d(16, 120, 5)

        # LayerNorm for conv layers
        # normalized_shape = output channel dimension
        self.bn1 = nn.BatchNorm2d(6)   # conv1 output size before pooling (assuming input 32x32)
        self.bn2 = nn.BatchNorm2d(16)  # conv2 output size before pooling
        # self.bn3 = nn.BatchNorm2d(120)

        # Fully connected layers
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # LayerNorm for fc layers
        self.bn3 = nn.BatchNorm1d(120)
        self.bn4 = nn.BatchNorm1d(84)

        self.register_buffer('mean', torch.tensor(0.1307))
        self.register_buffer('std', torch.tensor(0.3081))

    def forward(self, x, act_override=None):
        if act_override is not None:
            act = act_override
        elif self.training:
            act = F.relu
        else:
            act = quad_relu
            
        # input normalization
        # x = x / 255.0

        # x = (x - self.mean) / self.std
        # Conv1 + LayerNorm + Activation + Pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = act(x)
        x = F.avg_pool2d(x, (2, 2))

        # Conv2 + LayerNorm + Activation + Pool
        x = self.conv2(x)
        x = self.bn2(x)
        x = act(x)
        x = F.avg_pool2d(x, 2)
        
        # x = self.conv3(x)   # (batch,120,1,1)
        # x = self.bn3(x)
        # x = act(x)

        # Flatten
        x = x.view(-1, 400)

        # FC1 + LayerNorm + Activation
        x = self.fc1(x)
        x = self.bn3(x)
        x = act(x)

        # FC3 (logits)
        x = self.fc2(x)
        x = self.bn4(x)
        x = act(x)

        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features




net = LeNet5()
print(net)



torch.manual_seed(RANDOM_SEED)

model = net.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()


model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader,
                                    valid_loader, N_EPOCHS, DEVICE)

