import torch
import numpy as np
import os

def load_txt_tensor(path, shape):
    with open(path, 'r') as f:
        content = f.read().replace('\n', '').strip()
        tokens = [x.strip() for x in content.split(',') if x.strip() != '']
        data = [float(x) for x in tokens]
    return torch.tensor(data, dtype=torch.float32).view(*shape)


# input_image = torch.rand(1, 1, 32, 32, dtype=torch.float32).view(1, 1, 32, 32)
# np.savetxt("./lenet_weights_epoch(10)/input_image.txt", input_image.view(1, -1).numpy(), delimiter=",")

base_path = "./lenet_weights_epoch(10)"
input_tensor = load_txt_tensor("./lenet_weights_epoch(10)/input_image.txt", (1, 1, 32, 32))

# Conv1 params
weight1 = load_txt_tensor(f"{base_path}/conv1_weight.txt", (6, 1, 5, 5))
bias1   = load_txt_tensor(f"{base_path}/conv1_bias.txt", (6,))
gamma1  = load_txt_tensor(f"{base_path}/conv1_bn_gamma.txt", (6,))
beta1   = load_txt_tensor(f"{base_path}/conv1_bn_beta.txt", (6,))
mean1   = load_txt_tensor(f"{base_path}/conv1_bn_mean.txt", (6,))
var1    = load_txt_tensor(f"{base_path}/conv1_bn_var.txt", (6,))
# conv2
weight2 = load_txt_tensor(f"{base_path}/conv2_weight.txt", (16, 6, 5, 5))
bias2   = load_txt_tensor(f"{base_path}/conv2_bias.txt", (16,))
gamma2  = load_txt_tensor(f"{base_path}/conv2_bn_gamma.txt", (16,))
beta2   = load_txt_tensor(f"{base_path}/conv2_bn_beta.txt", (16,))
mean2   = load_txt_tensor(f"{base_path}/conv2_bn_mean.txt", (16,))
var2    = load_txt_tensor(f"{base_path}/conv2_bn_var.txt", (16,))


conv1 = torch.nn.Conv2d(1, 6, 5, stride=1, bias=True)
conv1.weight.data = weight1
conv1.bias.data = bias1

conv2 = torch.nn.Conv2d(6, 16, 5, stride=1, bias=True)
conv2.weight.data = weight2
conv2.bias.data = bias2

bn1 = torch.nn.BatchNorm2d(6)
bn1.weight.data = gamma1
bn1.bias.data = beta1
bn1.running_mean = mean1
bn1.running_var = var1
bn1.eval()

bn2 = torch.nn.BatchNorm2d(16)
bn2.weight.data = gamma2
bn2.bias.data = beta2
bn2.running_mean = mean2
bn2.running_var = var2
bn2.eval()


def approx_relu4(x):
    return 0.234606 + 0.5 * x + 0.204875 * x**2 - 0.0063896 * x**4


with torch.no_grad():
    x = conv1(input_tensor)
    x_bn = bn1(x)      
    x_relu = approx_relu4(x_bn)          


    x_pool = torch.nn.functional.avg_pool2d(x_relu, kernel_size=2, stride=2)

    x_conv2 = conv2(x_pool)
    x_bn2 = bn2(x_conv2)
    x_relu2 = approx_relu4(x_bn2)

    x_pool2 = torch.nn.functional.avg_pool2d(x_relu2, kernel_size=2, stride=2)


out_path = "./results/"
os.makedirs(out_path, exist_ok=True)


for ch in range(6):
    np.savetxt(
        os.path.join(out_path, f"py_conv1_output_{ch}.txt"),
        x_bn[0, ch].view(-1).numpy(),
        delimiter=","
    )


for ch in range(6):
    np.savetxt(
        os.path.join(out_path, f"py_relu1_output_{ch}.txt"),
        x_relu[0, ch].view(-1).numpy(),
        delimiter=","
    )

for ch in range(6):
    np.savetxt(
        os.path.join(out_path, f"py_pool1_output_{ch}.txt"),
        x_pool[0, ch].view(-1).numpy(),
        delimiter=","
    )

for ch in range(16):
    np.savetxt(
        os.path.join(out_path, f"py_conv2_output_{ch}.txt"),
        x_bn2[0, ch].view(-1).numpy(),
        delimiter=","
    )

for ch in range(16):
    np.savetxt(
        os.path.join(out_path, f"py_relu2_output_{ch}.txt"),
        x_relu2[0, ch].view(-1).numpy(),    
        delimiter=","
    )

for ch in range(16):
    np.savetxt(
        os.path.join(out_path, f"py_pool2_output_{ch}.txt"),
        x_pool2[0, ch].view(-1).numpy(),
        delimiter=","
    )


