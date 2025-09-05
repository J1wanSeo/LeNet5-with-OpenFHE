"""
Facade module to expose everything for main.py
"""

from .model import LeNet5
from .loader import load_weights_from_npy, get_mnist_dataset
from .utils import (
    select_activation,
    inference,
    print_mnist_ascii,
    save_output_to_txt,
    save_img_to_txt,
    print_boxed_message,
)
from .relu import quad_relu_polynomials, get_activation, get_description, list_activations


# --- 기존 코드에서 쓰던 infer_single_sample 그대로 유지 ---
import numpy as np
import torch


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
    vals = [float(x.strip()) for x in content.split(",") if x.strip()]
    output_tensor = torch.tensor(vals)
    pred = torch.argmax(output_tensor).item()
    print(f"[INFO] Loaded FC3 output from {filename}")
    print(f"Predicted label (from FC3 output): {pred}")
    return pred
