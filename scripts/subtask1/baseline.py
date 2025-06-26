
import json, re, os, tqdm
from openai import OpenAI
import ipdb
import concurrent.futures
import time
import threading
USER_PROMPT_TMPL = (
    "【案情全文】\n{fact}\n"
    "【被告姓名】\n{defendants}\n"
    "【相关法条】\n{articles}\n"
    "【可选罪名】\n{charges}\n"
    "---------------------\n"
    "请按照下列要求输出：\n"
    "1. 根据案情和相关法条，在可选罪名中预测被告的罪名\n"
    "2. 罪名可能是一个，也可能有多个\n"
    "3. 用 JSON 数组返回，示例：\n"
    '{{"defendant":str,"charges":["罪名1", ...]}}\n'
)
def return_prompt(fact, defendants, articles, charges):
    return USER_PROMPT_TMPL.format(fact=fact, defendants=defendants, articles=articles, charges=charges)

import requests


def query_llm_curl(client, prompt):
    SYSTEM_PROMPT = (
        "你是一名资深中国刑法专家，擅长把冗长案情浓缩成"
        "『判定被告刑期时最关键的事实要点』。"
        "你只输出中文，不解释、不添加法律意见。"
    )

    url = "http://localhost:11435/api/chat"

    data = {
        "model": "deepseek-r1:14b",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    }
    resp = requests.post(url, json=data, timeout=60)
    resp.raise_for_status()
    
    contents = []
    for line in resp.iter_lines(decode_unicode=True):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            # 只拼接 message.content
            if "message" in obj and "content" in obj["message"]:
                contents.append(obj["message"]["content"])
        except Exception as e:
            continue
    #ipdb.set_trace()
    content = "".join(contents).strip()
    # 提取 code block 或第一个合法 JSON
    match = re.search(r"```json\s*([\s\S]*?)```", content)
    if match:
        json_str = match.group(1)
    else:
        match = re.search(r'(\{[\s\S]*?\}|\[[\s\S]*?\])', content)
        json_str = match.group(1) if match else content
    try:
        r = json.loads(json_str)
        return r
    except Exception as e:
        print("解析本地模型输出失败:", e)
        return None


def load_finished_ids(progress_path):
    if not os.path.exists(progress_path):
        return set()
    with open(progress_path, "r", encoding="utf8") as f:
        return set(int(line.strip()) for line in f if line.strip().isdigit())


def llm_preprocess(args, split='train'):
    label2id = json.load(open(f"{args.data_dir}/label2id.json"))
    charges = list(label2id.keys())
    in_path = f"{args.data_dir}/{split}_ctx.jsonl"
    progress_path = f"{args.data_dir}/finished_{split}.txt"

    with open(in_path, encoding="utf8") as fin:
        samples = [json.loads(line) for line in fin]

    finished_ids = load_finished_ids(progress_path)
    tasks = [
        (client, sample, idx)
        for idx, sample in enumerate(samples)
        if idx not in finished_ids
    ]
    all_charges = []
    for client, sample, idx in tqdm.tqdm(tasks, desc=f"llm preprocess {split}"):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                prompt = return_prompt(sample["fact"], sample["defendant"], sample["ctx"])
                result = query_llm_curl(client, prompt)
                if isinstance(result, list) and len(result) > 0 and "charges" in result[0]:
                    for item in result:
                        if item["defendant"] == sample["defendant"]:
                            for charge in item["charges"]:
                                all_charges.append(charge)
                elif isinstance(result, dict) and "charges" in result:
                    if result["defendant"] == sample["defendant"]:
                        for charge in result["charges"]:
                            all_charges.append(charge)
                    else:
                        raise ValueError("Defendant mismatch in LLM response")
                else:
                    raise ValueError("LLM response format error")
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 5
                    print(f"Error on idx {idx}, attempt {attempt+1}/{max_retries}: {e}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed on idx {idx} after {max_retries} attempts: {e}")
                    raise
    id2accusations = {}
    for i, sample in enumerate(samples):
        for j, charge in enumerate(all_charges[i]):
            if j == 0:
                txt = f"{charge}"
            else:
                txt += f",{charge}"
        case_id = sample["case_idx"]
        if case_id not in id2accusations:
            id2accusations[case_id] = txt
        else:
            id2accusations[case_id] += f";{txt}"
    csv_path = os.path.join(args.save_dir, "predictions_accusation.csv")
    with open(csv_path, "w", encoding="utf8", newline="") as f:
        f.write("id,accusations\n")
        for idx in sorted(id2accusations.keys()):
            accusations = id2accusations[idx]
            # 如果有英文逗号，自动加引号
            if ',' in accusations:
                accusations = f'"{accusations}"'
            f.write(f"{idx+1},{accusations}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser("train subtask1 model")
    parser.add_argument("--data_dir", default="data/processed/subtask1", help="数据目录")
    parser.add_argument("--save_dir", default="checkpoints", help="模型检查点目录")
    parser.add_argument("--max_len", type=int, default=512, help="最大序列长度")
    args = parser.parse_args()
    args.save_dir = os.path.join(args.save_dir, f"subtask1_baseline")
    os.makedirs(args.save_dir, exist_ok=True)

    llm_preprocess(args, split='test')
