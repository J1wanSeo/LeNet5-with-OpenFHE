import torch
import numpy as np
from torchvision import datasets, transforms


def load_weights_from_npy(model, folder_path):
    model.conv1.weight.data = torch.from_numpy(np.load(f"{folder_path}/conv1_weight.npy"))
    model.conv1.bias.data = torch.from_numpy(np.load(f"{folder_path}/conv1_bias.npy"))
    model.conv2.weight.data = torch.from_numpy(np.load(f"{folder_path}/conv2_weight.npy"))
    model.conv2.bias.data = torch.from_numpy(np.load(f"{folder_path}/conv2_bias.npy"))
    model.fc1.weight.data = torch.from_numpy(np.load(f"{folder_path}/fc1_weight.npy"))
    model.fc1.bias.data = torch.from_numpy(np.load(f"{folder_path}/fc1_bias.npy"))
    model.fc2.weight.data = torch.from_numpy(np.load(f"{folder_path}/fc2_weight.npy"))
    model.fc2.bias.data = torch.from_numpy(np.load(f"{folder_path}/fc2_bias.npy"))
    model.fc3.weight.data = torch.from_numpy(np.load(f"{folder_path}/fc3_weight.npy"))
    model.fc3.bias.data = torch.from_numpy(np.load(f"{folder_path}/fc3_bias.npy"))

    def load_bn(layer, prefix):
        layer.weight.data = torch.from_numpy(np.load(f"{folder_path}/{prefix}_bn_gamma.npy"))
        layer.bias.data = torch.from_numpy(np.load(f"{folder_path}/{prefix}_bn_beta.npy"))
        layer.running_mean.data = torch.from_numpy(np.load(f"{folder_path}/{prefix}_bn_mean.npy"))
        layer.running_var.data = torch.from_numpy(np.load(f"{folder_path}/{prefix}_bn_var.npy"))

    load_bn(model.bn1, "conv1")
    load_bn(model.bn2, "conv2")
    load_bn(model.bn3, "fc1")
    load_bn(model.bn4, "fc2")
    print(f"Weights and BN parameters loaded from {folder_path}")


def get_mnist_dataset():
    transforms_val = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    valid_dataset = datasets.MNIST(root='./mnist_data', train=False, transform=transforms_val, download=True)
    return valid_dataset
