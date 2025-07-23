import torch
import numpy as np
import os

def load_txt_tensor(path, shape):
    with open(path, 'r') as f:
        content = f.read().replace('\n', '').strip()
        tokens = [x.strip() for x in content.split(',') if x.strip() != '']
        data = [float(x) for x in tokens]
    return torch.tensor(data, dtype=torch.float32).view(*shape)

# ==== 1. FC1 파라미터 로딩 ====
base_path = "./lenet_weights_epoch(10)"
fc_in_dim = 120
fc_out_dim = 84

input_tensor = torch.arange(1, fc_in_dim + 1, dtype=torch.float32).view(1, -1)
np.savetxt("./results/fc1_input.txt", input_tensor.numpy().flatten(), delimiter=",")

weight = load_txt_tensor(f"{base_path}/fc1_weight.txt", (fc_out_dim, fc_in_dim))
bias   = load_txt_tensor(f"{base_path}/fc1_bias.txt", (fc_out_dim,))

# ==== 2. FC 레이어 정의 및 파라미터 설정 ====
fc1 = torch.nn.Linear(fc_in_dim, fc_out_dim)
fc1.weight.data = weight
fc1.bias.data = bias

# ==== 3. 추론 ====
with torch.no_grad():
    output = fc1(input_tensor)

# ==== 4. 저장 ====
out_path = "./results/"
os.makedirs(out_path, exist_ok=True)
np.savetxt(os.path.join(out_path, "pytorch_fc1_output.txt"), output.numpy().flatten(), fmt="%.8f", delimiter=",")

print("[PyTorch] torch.nn.Linear 기반 FC1 결과 저장 완료.")
