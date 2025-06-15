import json, re, os, tqdm, jieba, argparse, collections
from openai import OpenAI
import ipdb
USER_PROMPT_TMPL = (
    "【案情全文】\n{fact}\n"
    "【被告姓名】\n{defendants}\n"
    "【相关法条】\n{articles}\n"
    "---------------------\n"
    "请按照下列要求输出：\n"
    "1. 列出和被告有关的有助于定罪的关键事实，≤300字。\n"
    "2. 从给定的相关法条中找出和案情最相关的部分，≤200字。\n"
    "2. 不要复述无关背景；不提及量刑、刑期。\n"
    "3. 用 JSON 数组返回，示例：\n"
    '{{"defendant":str,"key_facts":str, "key_articles":str}}'
)
def return_prompt(fact, defendants, articles):
    return USER_PROMPT_TMPL.format(fact=fact, defendants=defendants, articles=articles)
def query_llm(client, prompt):
    SYSTEM_PROMPT = (
        "你是一名资深中国刑法专家，擅长把冗长案情浓缩成"
        "『判定被告罪名时最关键的事实要点』。"
        "你只输出中文，不解释、不添加法律意见。"
    )

    completion = client.chat.completions.create(
    model="qwen-max",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ],
    )
    # ipdb.set_trace()
    # print(prompt)
    # print(completion.choices[0].message.content.strip())
    result = json.loads(completion.choices[0].message.content.strip())
    # print(len(result["key_facts"]))
    # print(len(result["key_articles"]))
    return result

# def llm_preprocess(OUT_DIR, split='train', num_workers=8):
#     client = OpenAI(
#         api_key="sk-b9283054aa5b4e089d380647972c9c59",
#         base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     ) 
#     in_path = f"{OUT_DIR}/{split}_ctx.jsonl"
#     out_path = f"{OUT_DIR}/{split}_llm.jsonl"
#     with open(in_path, encoding="utf8") as fin, open(out_path, "w", encoding="utf8") as fout:
#         total = sum(1 for _ in open(in_path, encoding="utf8"))
#         for i, line in enumerate(tqdm.tqdm(fin, desc=f"llm preprocess {split}", total=total)):
#             sample = json.loads(line)
#             prompt = return_prompt(sample["fact"], sample["defendant"], sample["ctx"])

#             result = query_llm(client, prompt)
#             if isinstance(result, list) and len(result) > 0 and "key_facts" in result[0]:
#                 sample["key_facts"] = result[0]["key_facts"]
#                 if "key_articles" in result[0]:
#                     sample["key_articles"] = result[0]["key_articles"]
#             elif isinstance(result, dict) and "key_facts" in result:
#                 sample["key_facts"] = result["key_facts"]
#                 if "key_articles" in result:
#                     sample["key_articles"] = result["key_articles"]
#             else:
#                 raise ValueError("LLM response format error")
            
#             fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
import concurrent.futures
import time
import random
import threading

def load_finished_ids(progress_path):
    if not os.path.exists(progress_path):
        return set()
    with open(progress_path, "r", encoding="utf8") as f:
        return set(int(line.strip()) for line in f if line.strip().isdigit())


def llm_preprocess(OUT_DIR, split='train', num_workers=8):
    client = OpenAI(
        api_key="sk-b9283054aa5b4e089d380647972c9c59",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    in_path = f"{OUT_DIR}/{split}_ctx.jsonl"
    out_path = f"{OUT_DIR}/{split}_llm.jsonl"
    progress_path = f"{OUT_DIR}/finished_{split}.txt"
    lock = threading.Lock()

    with open(in_path, encoding="utf8") as fin:
        samples = [json.loads(line) for line in fin]

    finished_ids = load_finished_ids(progress_path)
    tasks = [
        (client, sample, idx)
        for idx, sample in enumerate(samples)
        if idx not in finished_ids
    ]

    def process_and_write(args):
        client, sample, idx = args
        max_retries = 5
        for attempt in range(max_retries):
            try:
                prompt = return_prompt(sample["fact"], sample["defendant"], sample["ctx"])
                result = query_llm(client, prompt)
                if isinstance(result, list) and len(result) > 0 and "key_facts" in result[0]:
                    sample["key_facts"] = result[0]["key_facts"]
                    if "key_articles" in result[0]:
                        sample["key_articles"] = result[0]["key_articles"]
                elif isinstance(result, dict) and "key_facts" in result:
                    sample["key_facts"] = result["key_facts"]
                    if "key_articles" in result:
                        sample["key_articles"] = result["key_articles"]
                else:
                    raise ValueError("LLM response format error")
                sample["idx"] = idx
                with lock:
                    with open(progress_path, "a", encoding="utf8") as fprog:
                        fprog.write(f"{idx}\n")
                    with open(out_path, "a", encoding="utf8") as fout:
                        fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Error on idx {idx}, attempt {attempt+1}/{max_retries}: {e}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed on idx {idx} after {max_retries} attempts: {e}")
                    raise

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm.tqdm(executor.map(process_and_write, tasks), total=len(tasks), desc=f"llm preprocess {split}"))
    except KeyboardInterrupt:
        print("\n检测到 Ctrl+C，等待所有写入操作完成后安全退出...")
        # 这里不需要手动释放 lock，Python 会自动处理
        # 可以做一些清理或日志
        with lock:
            print("清理完成，安全退出。")
            raise