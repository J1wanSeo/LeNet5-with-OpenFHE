import numpy as np
import os

def save_npy_to_txt(npy_path, txt_path, delimiter=' '):
    arr = np.load(npy_path)
    arr_flat = arr.flatten()

    with open(txt_path, 'w') as f:
        for i, val in enumerate(arr_flat):
            f.write(f"{val}")
            if i != len(arr_flat) - 1:
                f.write(delimiter)
    print(f"Saved {npy_path} → {txt_path}")

def save_npy_to_txt(npy_path, txt_path, delimiter=' '):
    arr = np.load(npy_path)
    arr_flat = arr.flatten()
    with open(txt_path, 'w') as f:
        f.write(delimiter.join(map(str, arr_flat)))
    print(f"✔ Converted: {npy_path} -> {txt_path}")

def convert_all_in_folder(folder_path, delimiter=' '):
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy') and "Zone.Identifier" not in filename:
            npy_path = os.path.join(folder_path, filename)
            txt_path = os.path.join(folder_path, filename.replace('.npy', '.txt'))
            save_npy_to_txt(npy_path, txt_path, delimiter)

# 사용 예시
convert_all_in_folder("./lenet_weights_epoch(10)", delimiter=',')

