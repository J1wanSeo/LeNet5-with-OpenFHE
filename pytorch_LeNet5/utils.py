import torch
import numpy as np
from collections import Counter
from .relu import quad_relu_polynomials


def print_mnist_ascii(img_tensor, width=32, height=32):
    img = img_tensor.squeeze().cpu().numpy()
    chars = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@']
    img_scaled = (img * (len(chars)-1)).astype(int)
    for i in range(height):
        print("".join(chars[img_scaled[i, j]] for j in range(width)))


def save_output_to_txt(output_tensor, filename="fc3_output.txt"):
    arr = output_tensor.squeeze().cpu().numpy()
    with open(filename, "w") as f:
        for i, val in enumerate(arr):
            f.write(f"{val:.6f}")
            if i != len(arr) - 1:
                f.write(",\n")
    print(f"[INFO] FC3 output saved to {filename}")


def save_img_to_txt(img, filename="input_image.txt"):
    if img.dim() == 3:
        img = img[0]
    arr = img.cpu().numpy()
    H, W = arr.shape
    with open(filename, "w") as f:
        for i in range(H):
            line = ",".join(f"{x:.6f}" for x in arr[i])
            f.write(line + ",")
    print(f"[INFO] Image saved to {filename}")


def print_boxed_message(title, message_lines):
    all_lines = [title] + message_lines
    width = max(len(line) for line in all_lines) + 4
    print("+" + "-" * width + "+")
    print("| " + title.center(width - 2) + " |")
    print("+" + "-" * width + "+")
    for line in message_lines:
        print("| " + line.ljust(width - 2) + " |")
    print("+" + "-" * width + "+")


def select_activation():
    print("Select Activation function:")
    print(" 0: linear (x)")
    print(" 1: square (x^2)")
    print(" 2: CryptoNet (0.25 + 0.5 * x + 0.125 * x^2)")
    print(" 3: quad (0.234606 + 0.5 * x + 0.204875 * x^2 - 0.0063896 * x^4)")
    print(" 4: ReLU-maker (utils_approx.ReLU_maker)")
    print(" 5: student (custom polynomial)")

    choice = input("Enter number (0~5): ")
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
