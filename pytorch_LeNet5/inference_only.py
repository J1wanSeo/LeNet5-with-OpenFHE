import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from torchvision import datasets, transforms

from .utils_approx import ReLU_maker




def print_mnist_ascii(img_tensor, width=32, height=32):

    img = img_tensor.squeeze().cpu().numpy()

    # 밝기 기준 문자표 (밝기 클수록 더 진한 문자)
    chars = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@']

    # 픽셀값 0~1 → 인덱스 0~9 매핑
    img_scaled = (img * (len(chars)-1)).astype(int)

    for i in range(height):
        line = ''
        for j in range(width):
            line += chars[img_scaled[i, j]]
        print(line)


# --- 가중치 및 BN 파라미터 로드 ---
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
    # load_bn(model.bn4, "fc1")

    print(f"Weights and BN parameters loaded from {folder_path}")

def save_output_to_txt(output_tensor, filename="fc3_output.txt"):
    # output_tensor: 1D tensor, e.g. shape (10,)
    arr = output_tensor.squeeze().cpu().numpy()
    with open(filename, "w") as f:
        for i, val in enumerate(arr):
            f.write(f"{val:.6f}")
            if i != len(arr) - 1:
                f.write(",\n")
    print(f"[INFO] FC3 output saved to {filename}")


def infer_single_sample(model, dataset, device, act_override=None):
    model.eval()
    index = np.random.randint(0, len(dataset) - 1)
    img, label = dataset[index]
    input_tensor = img.unsqueeze(0).to(device)

    save_img_to_txt(img, "input_image.txt")

    with torch.no_grad():
        output = model(input_tensor, act_override=act_override)
        save_output_to_txt(output, "fc3_output.txt")
        _, pred = torch.max(output, 1)

    print(f"True Label: {label}")
    print(f"Predicted Label: {pred.item()}")

    print("\nInput image:")
    print_mnist_ascii(img)

def load_fc3_output_and_predict(filename="fc3_output.txt"):
    with open(filename, "r") as f:
        content = f.read()
    # 텍스트가 "val1,\nval2,\n..." 형태라고 가정
    vals = [float(x.strip()) for x in content.split(",") if x.strip()]
    output_tensor = torch.tensor(vals)
    pred = torch.argmax(output_tensor).item()
    print(f"[INFO] Loaded FC3 output from {filename}")
    print(f"Predicted label (from FC3 output): {pred}")
    return pred

def save_img_to_txt(img, filename="input_image.txt"):
    # img: torch.Tensor, shape (H, W) 또는 (C, H, W)
    
    # 만약 채널이 있다면 (C,H,W) → (H,W)로 변환 (예: 첫 채널만)
    if img.dim() == 3:
        img = img[0]  # 첫 채널 선택

    arr = img.cpu().numpy()
    H, W = arr.shape

    with open(filename, "w") as f:
        for i in range(H):
            line = ",".join(f"{x:.6f}" for x in arr[i])
            f.write(line + ",") 

    print(f"[INFO] Image saved to {filename}")

def print_boxed_message(title, message_lines):
    """
    title: 박스 제목 문자열
    message_lines: 메시지 내용 리스트 (문자열들)
    """
    # 박스 너비 계산 (제목과 내용 중 가장 긴 문자열 기준)
    all_lines = [title] + message_lines
    width = max(len(line) for line in all_lines) + 4  # 좌우 여백 포함

    # 윗줄
    print("+" + "-" * width + "+")
    # 제목 (가운데 정렬)
    print("| " + title.center(width - 2) + " |")
    # 구분선
    print("+" + "-" * width + "+")
    # 메시지 내용
    for line in message_lines:
        print("| " + line.ljust(width - 2) + " |")
    # 아랫줄
    print("+" + "-" * width + "+")


def select_activation():
    print("Select Activation function:")
    print(" 0: linear (x)")
    print(" 1: square (x^2)")
    print(" 2: CryptoNet (0.25 + 0.5 * x + 0.125 * x^2)")
    print(" 3: quad (0.234606 + 0.5 * x + 0.204875 * x^2 - 0.0063896 * x^4)")
    print(" 4: ReLU (max(0,x))")
    print(" 5: ReLU-maker (utils_approx.ReLU_maker)")
    print(" 6: student (custom polynomial)")

    choice = input("Enter number (0~6): ")

    try:
        choice_int = int(choice)
        if choice_int not in range(6):
            raise ValueError
    except:
        print("\n[Warning] Invalid input! Defaulting to CryptoNet (2)\n")
        choice_int = 2

    key_list = list(quad_relu_polynomials.keys())
    selected_key = key_list[choice_int]
    func, desc = quad_relu_polynomials[selected_key]
    print()
    print_boxed_message(" Selected Activation Function ", [f"Name: {selected_key}", f"Formula: {desc}"])
    print()
    return func, choice_int


def inference(model, data_loader, device, act_override=None):
    model.eval()
    correct = 0
    total = 0

    # 클래스별 정답-예측 카운터 초기화
    from collections import Counter
    correct_per_class = Counter()
    total_per_class = Counter()

    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)
            y = y.to(device)
            outputs = model(X, act_override=act_override)
            _, predicted = torch.max(outputs, 1)

            total += y.size(0)
            correct += (predicted == y).sum().item()

            # 클래스별 통계 집계
            for t, p in zip(y, predicted):
                total_per_class[int(t)] += 1
                if t == p:
                    correct_per_class[int(t)] += 1

    acc = correct / total
    print(f"Inference Accuracy: {acc*100:.2f}%")

    print("\nClass-wise Accuracy:")
    for cls in range(10):
        if total_per_class[cls] > 0:
            class_acc = correct_per_class[cls] / total_per_class[cls]
            print(f"  Class {cls}: {class_acc*100:.2f}% ({correct_per_class[cls]}/{total_per_class[cls]})")
        else:
            print(f"  Class {cls}: No samples")

    return acc

# Custom Activation Function


quad_relu_polynomials = {
    'linear': (lambda x: x,
                "x"),
    'square': (lambda x: x ** 2,
                "x ** 2"),
    'CryptoNet': (lambda x: 0.125 * x**2 + 0.5 * x + 0.25,
                "0.25 + 0.5 * x + 0.125 * x**2"),            
    # 'quad_v4': (lambda x: 0.125*x**2 + 0.5 * x,
    #             "0.5 * x + 0.125*x**2"),
    'quad': (lambda x: 0.234606 + 0.5 * x + 0.204875 * x ** 2 - 0.0063896 * x ** 4,
                "0.234606 + 0.5 * x + 0.204875 * x ** 2 - 0.0063896 * x ** 4"),
    'Original ReLU': (None,
                "ReLU(x)=max(0,x)"),
    'student': (lambda x: x, "insert your own description"),
    'ReLU-maker' : (lambda x: ReLU_maker({'type':'proposed','alpha':13,'B':50})(x), "ReLU Maker with alpha==13")

    }

# 선택
# quad_relu= select_activation()



class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        # Convolution layers
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)


        # LayerNorm for conv layers
        # normalized_shape = output channel dimension
        self.bn1 = nn.BatchNorm2d(6)   # conv1 output size before pooling (assuming input 32x32)
        self.bn2 = nn.BatchNorm2d(16)  # conv2 output size before pooling


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
        elif act_override is None:
            act = F.relu
        elif self.training:
            act = F.relu
        else:
            act = quad_relu_polynomials[2]['CryptoNet']
            

        x = self.conv1(x)
        x = self.bn1(x)
        x = act(x)
        x = F.avg_pool2d(x, (2, 2))

        # Conv2 + LayerNorm + Activation + Pool
        x = self.conv2(x)
        x = self.bn2(x)
        x = act(x)
        x = F.avg_pool2d(x, 2)
        
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



if __name__ == "__main__":
    DEVICE = torch.device("cpu")

    # MNIST 검증 데이터 로더
    transforms_val = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    valid_dataset = datasets.MNIST(root='./mnist_data', train=False, transform=transforms_val, download=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    # 모델 생성
    model = LeNet5().to(DEVICE)

    # 활성화 함수 선택
    act_fn = select_activation()

    # 저장된 가중치 불러오기
    load_weights_from_npy(model, "./parameters_standard")

    infer_single_sample(model, valid_dataset, device=DEVICE, act_override=act_fn)
    # exit()
    # 추론 실행
    inference(model, valid_loader, DEVICE, act_override=act_fn)

