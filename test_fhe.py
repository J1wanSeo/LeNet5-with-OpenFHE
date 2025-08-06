import torch
import numpy as np
import os
import torch.nn.functional as F

def load_txt_tensor(path, shape):
    with open(path, 'r') as f:
        content = f.read().replace('\n', '').strip()
        tokens = [x.strip() for x in content.split(',') if x.strip() != '']
        data = [float(x) for x in tokens]
    return torch.tensor(data, dtype=torch.float32).view(*shape)


# input_image = torch.rand(1, 1, 32, 32, dtype=torch.float32).view(1, 1, 32, 32)
# np.savetxt("./lenet_weights_epoch(10)/input_image.txt", input_image.view(1, -1).numpy(), delimiter=",")

base_path = "./pytorch_LeNet5/parameters_standard"
input_tensor = load_txt_tensor("./input_image.txt", (1, 1, 32, 32))

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

# fc layers params
weight_fc1 = load_txt_tensor(f"{base_path}/fc1_weight.txt", (120, 400))  # (out_features, in_features)
bias_fc1 = load_txt_tensor(f"{base_path}/fc1_bias.txt", (120,))
gamma_fc1 = load_txt_tensor(f"{base_path}/fc1_bn_gamma.txt", (120,))
beta_fc1 = load_txt_tensor(f"{base_path}/fc1_bn_beta.txt", (120,))
mean_fc1 = load_txt_tensor(f"{base_path}/fc1_bn_mean.txt", (120,))
var_fc1 = load_txt_tensor(f"{base_path}/fc1_bn_var.txt", (120,))

weight_fc2 = load_txt_tensor(f"{base_path}/fc2_weight.txt", (84, 120))
bias_fc2 = load_txt_tensor(f"{base_path}/fc2_bias.txt", (84,))
gamma_fc2 = load_txt_tensor(f"{base_path}/fc2_bn_gamma.txt", (84,))
beta_fc2 = load_txt_tensor(f"{base_path}/fc2_bn_beta.txt", (84,))
mean_fc2 = load_txt_tensor(f"{base_path}/fc2_bn_mean.txt", (84,))
var_fc2 = load_txt_tensor(f"{base_path}/fc2_bn_var.txt", (84,))

weight_fc3 = load_txt_tensor(f"{base_path}/fc3_weight.txt", (10, 84))
bias_fc3 = load_txt_tensor(f"{base_path}/fc3_bias.txt", (10,))


print(mean1.shape)
print(var1.shape)

# 모듈 생성 및 파라미터 할당
conv1 = torch.nn.Conv2d(1, 6, 5)
conv1.weight.data = weight1
conv1.bias.data = bias1

conv2 = torch.nn.Conv2d(6, 16, 5)
conv2.weight.data = weight2
conv2.bias.data = bias2

fc1 = torch.nn.Linear(400, 120)
fc1.weight.data = weight_fc1
fc1.bias.data = bias_fc1

fc2 = torch.nn.Linear(120, 84)
fc2.weight.data = weight_fc2
fc2.bias.data = bias_fc2

fc3 = torch.nn.Linear(84, 10)
fc3.weight.data = weight_fc3
fc3.bias.data = bias_fc3

bn1 = torch.nn.BatchNorm2d(6)
bn1.weight.data = gamma1
bn1.bias.data = beta1
bn1.running_mean.data.copy_(mean1)
bn1.running_var.data.copy_(var1)
bn1.eval()

bn2 = torch.nn.BatchNorm2d(16)
bn2.weight.data = gamma2
bn2.bias.data = beta2
bn2.running_mean.data.copy_(mean2)
bn2.running_var.data.copy_(var2)
bn2.eval()

bn3 = torch.nn.BatchNorm1d(120)
bn3.weight.data = gamma_fc1
bn3.bias.data = beta_fc1
bn3.running_mean.data.copy_(mean_fc1)
bn3.running_var.data.copy_(var_fc1)
bn3.eval()

bn4 = torch.nn.BatchNorm1d(84)
bn4.weight.data = gamma_fc2
bn4.bias.data = beta_fc2
bn4.running_mean.data.copy_(mean_fc2)
bn4.running_var.data.copy_(var_fc2)
bn4.eval()


def approx_relu4(x):
    return x
with torch.no_grad():
    conv1_out = conv1(input_tensor)
    bn1_out = bn1(conv1_out)
    relu1_out = approx_relu4(bn1_out)
    pool1_out = F.avg_pool2d(relu1_out, 2)

    conv2_out = conv2(pool1_out)
    bn2_out = bn2(conv2_out)
    relu2_out = approx_relu4(bn2_out)
    pool2_out = F.avg_pool2d(relu2_out, 2)

    flatten_out = pool2_out.view(pool2_out.size(0), -1)

    fc1_out = fc1(flatten_out)
    bn3_out = bn3(fc1_out)
    relu3_out = approx_relu4(bn3_out)

    fc2_out = fc2(relu3_out)
    bn4_out = bn4(fc2_out)
    relu4_out = approx_relu4(bn4_out)

    fc3_out = fc3(relu4_out)



out_path = "./results/"
os.makedirs(out_path, exist_ok=True)

def save_channel_outputs(tensor, prefix):
    if tensor.dim() == 4:
        # 4D tensor (N,C,H,W) -> 각 채널별 1D 배열로 저장
        for ch in range(tensor.size(1)):
            np.savetxt(
                os.path.join(out_path, f"{prefix}_channel{ch}.txt"),
                tensor[0, ch].flatten().cpu().numpy(),
                delimiter=","
            )
    elif tensor.dim() == 2:
        # 2D tensor (N, Features) -> 1D 벡터별 저장
        for ch in range(tensor.size(1)):
            np.savetxt(
                os.path.join(out_path, f"{prefix}_feature{ch}.txt"),
                tensor[0, ch].cpu().numpy().reshape(1),
                delimiter=","
            )

# 출력 저장
save_channel_outputs(bn1_out, "py_conv1_output")
save_channel_outputs(relu1_out, "py_relu1_output")
save_channel_outputs(pool1_out, "py_pool1_output")

save_channel_outputs(bn2_out, "py_conv2_output")
save_channel_outputs(relu2_out, "py_relu2_output")
save_channel_outputs(pool2_out, "py_pool2_output")

save_channel_outputs(bn3_out, "py_fc1_output")
save_channel_outputs(relu3_out, "py_fc1_relu_output")

save_channel_outputs(bn4_out, "py_fc2_output")
save_channel_outputs(relu4_out, "py_fc2_relu_output")

save_channel_outputs(fc3_out, "py_fc3_output")