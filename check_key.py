import json

input_path = "data/processed/subtask1/test_llm.jsonl"      # 替换为你的jsonl文件路径
output_txt = "data/processed/subtask1/finished_test.txt"    # 输出idx的txt文件
output_jsonl = "data/processed/subtask1/test_llm.jsonl"  # 只保留完整条目的新jsonl
required_keys = {'defendant', 'fact', 'charge_ids', 'imprisonment', 'standard_accusation', 'ctx', 'key_facts', 'key_articles', 'idx'}

complete_idx = []
complete_samples = []

with open(input_path, "r", encoding="utf-8") as fin:
    for line in fin:
        if not line.strip():
            continue
        try:
            sample = json.loads(line)
        except Exception as e:
            print("解析失败，跳过一行:", e)
            continue
        if required_keys.issubset(sample.keys()):
            complete_idx.append(str(sample["idx"]))
            complete_samples.append(sample)
        else:
            print(f"缺少字段 idx={sample.get('idx', 'N/A')} 缺少: {required_keys - set(sample.keys())}")

with open(output_txt, "w", encoding="utf-8") as fout:
    for idx in complete_idx:
        fout.write(f"{idx}\n")

with open(output_jsonl, "w", encoding="utf-8") as fout:
    for sample in complete_samples:
        fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"已保存完整条目的idx到 {output_txt}，共{len(complete_idx)}条。")
print(f"已保存完整条目的jsonl到 {output_jsonl}，共{len(complete_samples)}条。")