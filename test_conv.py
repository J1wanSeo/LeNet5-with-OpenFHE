import torch
import torch.nn.functional as F

# 8x8 입력 생성 (0.01부터 0.64까지)
input_data = torch.linspace(0.01, 0.25, steps=25).reshape(1, 1, 5, 5)
# 5x5 필터 생성 (모두 0.1)
weight = torch.full((1, 1, 3, 3), 0.1)

# bias 0
bias = torch.zeros(1)

# conv2d 수행
output = F.conv2d(input_data, weight, bias=bias, stride=1, padding=0)

# 텍스트 파일로 저장
output_np = output.squeeze().numpy()

with open("conv2d_output.txt", "w") as f:
    for row in output_np:
        f.write(",".join(f"{v:.8f}" for v in row) + "\n")

print("[INFO] Conv2D output saved to conv2d_output.txt")
