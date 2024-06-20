import torch
import numpy as np
data_train = torch.load('Data_tensor_check/f'+str(0)+'_train')
data_test = torch.load('Data_tensor_check/f'+str(0)+'_test')
data_ft = torch.load('Data_tensor_check/ft')
data = data_train + data_test + data_ft

age = [data[n][4][0].item() for n in range(len(data))]
# print(len(age))
# print(age)




# 平均と標準偏差を計算
mean_age = np.mean(age)
std_age = np.std(age)
# 結果を "XX ± XX" の形式で表示
result = f"{mean_age:.2f} ± {std_age:.2f}"
print(result)



data_traval = data_train + data_test
print(len(data_traval))
label_traval = [data[n][3].item() for n in range(len(data_traval))]
print(len(label_traval))
print(label_traval)
zeros_ratio = label_traval.count(0) / len(label_traval)
ones_ratio = label_traval.count(1) / len(label_traval)
print(label_traval.count(0)/5)
print(label_traval.count(1)/5)
print(f"9 の割合: {zeros_ratio:.2f}")
print(f"1 の割合: {ones_ratio:.2f}")

a = 11/43
b = 32/43
print(f"a: {a:.2f}")
print(f"b: {b:.2f}")