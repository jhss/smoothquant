from smoothquant.observer import OutlierObserver

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import glob

# tensor = torch.load("statistics/140372464529552.pt")
# plt.plot(tensor)
# plt.savefig("1.png")
# print("[DEBUG] shape: ", tensor.shape)

tensor_name_list = glob.glob("statistics/*.pt")

for tensor_name in tensor_name_list:
    cur_tensor = torch.load(tensor_name)
    print(cur_tensor.shape)