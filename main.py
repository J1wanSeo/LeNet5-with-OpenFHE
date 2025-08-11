import subprocess
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pytorch_LeNet5 import inference_only  # inference_only.py 모듈 임포트


def run_python_baseline():
    """
    Python baseline inference 실행.
    - 단일 샘플 추론 및 전체 검증셋 정확도 출력
    """
    DEVICE = torch.device("cpu")
    transforms_val = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    valid_dataset = datasets.MNIST(root='./pytorch_LeNet5/mnist_data', train=False,
                                  transform=transforms_val, download=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    model = inference_only.LeNet5().to(DEVICE)
    act_fn = inference_only.select_activation()
    inference_only.load_weights_from_npy(model, "./pytorch_LeNet5/parameters_standard")

    print("=== Python Baseline: Single Sample Inference ===")
    inference_only.infer_single_sample(model, valid_dataset, DEVICE, act_override=act_fn)

    print("=== Python Baseline: Full Validation Inference ===")
    acc = inference_only.inference(model, valid_loader, DEVICE, act_override=act_fn)

    print(f"[RESULT] Python baseline accuracy: {acc*100:.2f}%")
    return acc


def run_cpp_fhe_inference():
    """
    CPP 빌드 및 FHE 추론 수행.
    1) Python에서 무작위 샘플을 골라 input_image.txt로 저장
    2) autotest.sh 실행하여 CPP 빌드 및 실행
    3) fc3_output.txt 읽어 예측 라벨 출력
    """
    DEVICE = torch.device("cpu")
    transforms_val = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    valid_dataset = datasets.MNIST(root='./pytorch_LeNet5/mnist_data', train=False,
                                  transform=transforms_val, download=True)

    model = inference_only.LeNet5().to(DEVICE)
    act_fn, relu_mode = inference_only.select_activation()
    inference_only.load_weights_from_npy(model, "./pytorch_LeNet5/parameters_standard")

    # 1) 무작위 샘플 이미지 저장 (CPP 입력용)
    print("[INFO] Generating input image for CPP FHE...")
    inference_only.infer_single_sample(model, valid_dataset, DEVICE, act_override=act_fn)

    # 2) CPP 빌드 및 실행 (autotest.sh)
    print("[INFO] Running autotest.sh for CPP FHE inference...")
    try:
        subprocess.run(["bash", "./autotest.sh", str(relu_mode)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] autotest.sh execution failed: {e}")
        return None

    # 3) CPP 결과 읽기 및 예측 라벨 출력
    pred = inference_only.load_fc3_output_and_predict("./build/fc3_output.txt")
    print(f"[RESULT] CPP FHE predicted label: {pred}")

    return pred


def main():
    print("=== Execution Mode Selection ===")
    print("1. Python Baseline Inference")
    print("2. CPP FHE Inference")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        run_python_baseline()
    elif choice == "2":
        run_cpp_fhe_inference()
    else:
        print("[ERROR] Invalid input. Please enter 1 or 2.")


if __name__ == "__main__":
    main()
