import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from network import GM11

raw_data = pd.read_csv("data.csv")
y = raw_data["y"].values

# 标准化
scaler = []
for i in range(len(raw_data.columns) - 1):
    scaler.append(StandardScaler())
for i in range(len(raw_data.columns) - 1):
    raw_data[raw_data.columns[i]] = scaler[i].fit_transform(raw_data[raw_data.columns[i]].values.reshape(-1, 1))

# 构建灰色预测模型，导出2014,2015的特征
raw_data.loc[len(raw_data)] = None
raw_data.loc[len(raw_data)] = None

for i in raw_data.columns[:-1]:
    GM = GM11(raw_data[i][list(range(len(raw_data) - 2))].values)
    f = GM[0]
    c = GM[-2]
    p = GM[-1]
    raw_data[i][len(raw_data) - 2] = f(len(raw_data) - 1)
    raw_data[i][len(raw_data) - 1] = f(len(raw_data))
    raw_data[i] = raw_data[i].round(2)
    if (c < 0.35) & (p > 0.95):
        print('对于模型{}，该模型精度为---好'.format(i))
    elif (c < 0.5) & (p > 0.8):
        print('对于模型{}，该模型精度为---合格'.format(i))
    elif (c < 0.65) & (p > 0.7):
        print('对于模型{}，该模型精度为---勉强合格'.format(i))
    else:
        print('对于模型{}，该模型精度为---不合格'.format(i))
        raw_data.drop(i, inplace=True, axis=1)

# 构建特征矩阵
raw_data.drop("y", inplace=True, axis=1, index=None)

# 保存矩阵
np.save("data/y", y)
np.save("data/x", raw_data[:-2])

np.save("data/x_pre", raw_data.iloc[-2:, :])
