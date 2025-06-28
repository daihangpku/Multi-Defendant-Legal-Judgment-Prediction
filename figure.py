import json
import matplotlib.pyplot as plt

# 读取数据
with open('F1.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 拆分数据
steps = [item[1] for item in data]
f1_scores = [item[2] for item in data]

# 画图
plt.figure(figsize=(10, 6))
plt.plot(steps, f1_scores, marker='o')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('F1', fontsize=16)
plt.title('Eval F1 Curve', fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.show()