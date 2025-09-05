import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from .relu import quad_relu_polynomials


def save_channel_outputs(tensor, prefix):
    out_path = "./results/"
    os.makedirs(out_path, exist_ok=True)
    if tensor.dim() == 4:
        for ch in range(tensor.size(1)):
            np.savetxt(
                os.path.join(out_path, f"{prefix}_channel{ch}.txt"),
                tensor[0, ch].flatten().cpu().numpy(),
                delimiter=","
            )
    elif tensor.dim() == 2:
        for ch in range(tensor.size(1)):
            np.savetxt(
                os.path.join(out_path, f"{prefix}_feature{ch}.txt"),
                tensor[0, ch].cpu().numpy().reshape(1),
                delimiter=","
            )


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.bn3 = nn.BatchNorm1d(120)
        self.bn4 = nn.BatchNorm1d(84)
        self.register_buffer('mean', torch.tensor(0.1307))
        self.register_buffer('std', torch.tensor(0.3081))

    def forward(self, x, act_override=None):
        if act_override is not None:
            act = act_override
        elif act_override is None:
            act = F.relu
        elif self.training:
            act = F.relu
        else:
            act = quad_relu_polynomials['CryptoNet'][0]

        conv1_out = self.conv1(x)
        bn1_out = self.bn1(conv1_out)
        relu1_out = act(bn1_out)
        pool1_out = F.avg_pool2d(relu1_out, 2)

        conv2_out = self.conv2(pool1_out)
        bn2_out = self.bn2(conv2_out)
        relu2_out = act(bn2_out)
        pool2_out = F.avg_pool2d(relu2_out, 2)

        x = pool2_out.view(-1, 400)

        fc1_out = self.fc1(x)
        bn3_out = self.bn3(fc1_out)
        relu3_out = act(bn3_out)

        fc2_out = self.fc2(relu3_out)
        bn4_out = self.bn4(fc2_out)
        relu4_out = act(bn4_out)

        fc3_out = self.fc3(relu4_out)

        # save_channel_outputs(bn1_out, "py_conv1_output")
        # save_channel_outputs(relu1_out, "py_relu1_output")
        # save_channel_outputs(pool1_out, "py_pool1_output")
        # save_channel_outputs(bn2_out, "py_conv2_output")
        # save_channel_outputs(relu2_out, "py_relu2_output")
        # save_channel_outputs(pool2_out, "py_pool2_output")
        # save_channel_outputs(bn3_out, "py_fc1_output")
        # save_channel_outputs(relu3_out, "py_fc1_relu_output")
        # save_channel_outputs(bn4_out, "py_fc2_output")
        # save_channel_outputs(relu4_out, "py_fc2_relu_output")
        # save_channel_outputs(fc3_out, "py_fc3_output")

        return fc3_out
