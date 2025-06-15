import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-legal-electra-base-discriminator")
lengths = []
with open("data/processed/subtask1/train_ctx.jsonl", encoding="utf8") as fin:
    for line in fin:
        s = json.loads(line)
        txt = s["fact"] 
        #lengths.append(len(tokenizer.tokenize(txt)))
        lengths.append(len(txt))

print("最大长度:", max(lengths))
print("平均长度:", sum(lengths)/len(lengths))
for p in [90, 95, 99]:
    print(f"{p}%样本长度小于:", sorted(lengths)[int(len(lengths)*p/100)])