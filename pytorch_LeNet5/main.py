import torch
from torch.utils.data import DataLoader

from .model import LeNet5
from .loader import load_weights_from_npy, get_mnist_dataset
from .utils import select_activation, inference
from .utils import save_output_to_txt, save_img_to_txt, print_mnist_ascii


if __name__ == "__main__":
    DEVICE = torch.device("cpu")

    # 데이터셋
    valid_dataset = get_mnist_dataset()
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    # 모델
    model = LeNet5().to(DEVICE)

    # activation 선택
    act_fn, _ = select_activation()

    # 가중치 로드
    load_weights_from_npy(model, "./parameters_standard")

    # 샘플 추론
    model.eval()
    img, label = valid_dataset[0]
    with torch.no_grad():
        output = model(img.unsqueeze(0).to(DEVICE), act_override=act_fn)
    save_output_to_txt(output, "fc3_output.txt")
    print(f"True label: {label}, Predicted: {torch.argmax(output).item()}")

    # 전체 추론
    inference(model, valid_loader, DEVICE, act_override=act_fn)
