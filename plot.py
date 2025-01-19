import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 文件列表
files = [
    "tst-FedAvg.npy",
    "tst-FedDyn.npy",
]

base_path = "out/T=1000/CIFAR10-Dirichlet0.1-100/active-0.1/Performance"

for file_path in files:
    data = np.load(os.path.join(base_path, file_path))
    accuracy = data[:, 1]

    # 计算移动平均
    window_size = 5
    accuracy = np.convolve(accuracy, np.ones(window_size) / window_size, mode="valid")

    plt.plot(accuracy, linestyle="-", label=file_path)

plt.title("Accuracy per Round")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.legend(loc="best", framealpha=0.5)  # 自动选择位置，设置半透明背景
# 保存图像
plt.savefig("outputs/Cifar10-dir0.1_c100-0.1/ResNet18.png")
