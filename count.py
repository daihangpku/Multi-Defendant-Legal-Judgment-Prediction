import json
import matplotlib.pyplot as plt

filename = 'data/raw/subtask2/train.jsonl'
d = {}
total = 0
with open(filename, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        for item in data["outcomes"]:
            for judgment in item["judgment"]:
                imp = judgment["imprisonment"]
                if imp not in d:
                    d[imp] = 0
                d[imp] += 1
                total += 1

# 按key排序
sorted_d = dict(sorted(d.items(), key=lambda x: x[0]))

# 计算比例
result = []
for k, v in sorted_d.items():
    result.append({
        "imprisonment": k,
        "count": v,
        "ratio": v / total
    })
imprisonment_gt_001 = [item["imprisonment"] for item in result if item["ratio"] > 0.005]
print(imprisonment_gt_001)
# 保存到json文件
with open('imprisonment_distribution.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
x = [k for k in sorted_d.keys()]
y = [v for v in sorted_d.values()]
plt.figure(figsize=(10, 6))
plt.bar(range(len(x)), y, width=0.6)
plt.xlabel('Imprisonment (months)', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.title('Imprisonment Distribution', fontsize=18)

# 只显示每隔n个标签
n = 5  # 可根据实际情况调整
xtick_pos = list(range(len(x)))
xtick_labels = [str(xi) if i % n == 0 else '' for i, xi in enumerate(x)]
plt.xticks(xtick_pos, xtick_labels, rotation=45, fontsize=14)

plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()