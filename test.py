import torch
import numpy as np

def load_txt_tensor(path, shape):
    with open(path, 'r') as f:
        content = f.read().replace('\n', '').strip()
        tokens = [x.strip() for x in content.split(',') if x.strip() != '']
        data = [float(x) for x in tokens]
    return torch.tensor(data, dtype=torch.float32).view(*shape)


# 1. 32x32 input Image
input_image = torch.arange(1, 1025, dtype=torch.float32).view(1, 1, 32, 32)

# 2. flatten the image input
np.savetxt("input_image.txt", input_image.view(1,-1).numpy(), delimiter=",")

# 3. load image data
input_tensor = load_txt_tensor("input_image.txt", (1, 1, 32, 32))           
weight_tensor = load_txt_tensor("./lenet_weights_epoch(10)/conv1_weight.txt", (6, 1, 5, 5))           
bias_tensor = load_txt_tensor("./lenet_weights_epoch(10)/conv1_bias.txt", (6,))                       
# bn load
gamma_tensor = load_txt_tensor("./lenet_weights_epoch(10)/conv1_bn_gamma.txt", (6,))
beta_tensor = load_txt_tensor("./lenet_weights_epoch(10)/conv1_bn_beta.txt", (6,))
mean_tensor = load_txt_tensor("./lenet_weights_epoch(10)/conv1_bn_mean.txt", (6,))
var_tensor = load_txt_tensor("./lenet_weights_epoch(10)/conv1_bn_var.txt", (6,))

# 4. Layers
conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, bias=True)
bn1 = torch.nn.BatchNorm2d(num_features=6)

# 5. Set weights/bias
conv1.weight.data = weight_tensor
conv1.bias.data = bias_tensor

# 6. Set BN parameters
bn1.weight.data = gamma_tensor
bn1.bias.data = beta_tensor
bn1.running_mean = mean_tensor
bn1.running_var = var_tensor
bn1.eval()  # inference 모드에서 running stats 사용

# 7. Forward
with torch.no_grad():
    conv_output = conv1(input_tensor)     # (1, 6, 28, 28)
    bn_output = bn1(conv_output)          # BN 적용된 결과

# 8. Save output per channel
for ch in range(6):
    np.savetxt(
        f"conv1_output_channel_{ch}_by_pytorch.txt",
        bn_output[0, ch].view(-1).numpy(),
        delimiter=","
    )
