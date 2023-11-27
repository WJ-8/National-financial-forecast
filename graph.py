import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Microsoft YaHei'

pre = np.load("pre.npy").tolist()

target = np.load("data/y.npy").tolist() + [None, None]

x = [i for i in range(1994, 2016)]

plt.plot(x, pre, label='预测')  # 画第一个系列的折线，设置标签为 'Series 1'
plt.plot(x, target, label='真实')  # 画第二个系列的折线，设置标签为 'Series 2'

# 添加标签和标题
plt.xlabel('年份')  # X 轴标签
plt.ylabel('财政收入')  # Y 轴标签
plt.title('财政收入预测')  # 图表标题
plt.box(False)
plt.grid(True)
# 添加图例
plt.legend()

# 显示图表
plt.show()
