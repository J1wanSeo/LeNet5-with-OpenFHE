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
weight_tensor = load_txt_tensor("conv1_weight.txt", (6, 1, 5, 5))           
bias_tensor = load_txt_tensor("conv1_bias.txt", (6,))                       

# 4. Conv1 
conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
conv1.weight.data = weight_tensor
conv1.bias.data = bias_tensor

# 5. Forward 
with torch.no_grad():
    output = conv1(input_tensor)  # → (1, 6, 28, 28)

# 6. save result
for ch in range(6):
    np.savetxt(
        f"conv1_output_channel_{ch}_by_pytorch.txt",
        output[0, ch].view(-1).numpy(),
        delimiter=","
    )

