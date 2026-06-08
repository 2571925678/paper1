
# @desc: 画单张折线图

import matplotlib.pyplot as plt
import numpy as np

data1 = np.random.choice(30, size=40)
data2 = np.random.choice(30, size=40)
data3 = np.random.choice(30, size=40)

list_color = ['palegreen', 'lime', 'springgreen', 'aquamarine',
              'darkslategray', 'darkcyan', 'green']

# 字体格式设置
font = {'family': 'Times New Roman', 'size': 14}

plt.figure(figsize=(18, 12))
plt.title("Example Figure", family='Times New Roman', fontsize=20)
plt.plot(data1, marker='^', linestyle='--', label='Model 4 prediction', color='lightskyblue')
plt.plot(data2, label='observation', color='lightcoral')
plt.plot(data3, linestyle='-.', label='Model 5 prediction', color='lightpink')
plt.xlabel('Example x label()', family='Times New Roman', fontsize=14)
plt.ylabel('Example y label()', family='Times New Roman', fontsize=14)
plt.xlim(0, len(data1))  # 横坐标的取值范围
plt.ylim()
plt.xticks(fontsize=14, rotation=60)  # 横坐标的刻度
plt.yticks(fontsize=14)
plt.legend(loc='upper left', prop=font)
plt.tight_layout()
# plt.savefig('../Output/Picture1/comparison_Add/CEEMDAN_AE_BiLSTM_LinearAdd', dpi=500)
plt.show()