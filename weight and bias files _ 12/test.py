import torch
import numpy as np

def load_txt_tensor(path, shape):
    with open(path, 'r') as f:
        content = f.read().replace('\n', '').strip()
        tokens = [x.strip() for x in content.split(',') if x.strip() != '']
        data = [float(x) for x in tokens]
    return torch.tensor(data, dtype=torch.float32).view(*shape)


# 1. 고정된 32x32 입력 이미지 (1~1024)
input_image = torch.arange(1, 1025, dtype=torch.float32).view(1, 1, 32, 32)

# 2. flatten 해서 txt로 저장 (OpenFHE용)
np.savetxt("input_image.txt", input_image.view(1,-1).numpy(), delimiter=",")
# 1. 파일에서 데이터 불러오기
input_tensor = load_txt_tensor("input_image.txt", (1, 1, 32, 32))           # 입력
weight_tensor = load_txt_tensor("conv1_weight.txt", (6, 1, 5, 5))           # Conv1 weight
bias_tensor = load_txt_tensor("conv1_bias.txt", (6,))                       # Conv1 bias

# 2. Conv1 정의 및 가중치 설정
conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
conv1.weight.data = weight_tensor
conv1.bias.data = bias_tensor

# 3. Forward 실행
with torch.no_grad():
    output = conv1(input_tensor)  # → (1, 6, 28, 28)

# 4. 결과 저장 (첫 채널만 저장 예시)
np.savetxt("conv1_output_channel0_from_txt.txt", output[0, 0].view(-1).numpy(), delimiter=",")

# 5. 결과 일부 확인
print("Output shape:", output.shape)
print("Output (channel 0, top-left 5x5):")
print(output[0, 0, :5, :5])

