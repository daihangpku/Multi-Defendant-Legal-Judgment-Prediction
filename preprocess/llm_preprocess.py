import json, re, os, tqdm, jieba, argparse, collections
from rank_bm25 import BM25Okapi
from openai import OpenAI
import ipdb
USER_PROMPT_TMPL = (
    "【案情全文】\n{fact}\n"
    "【被告姓名】\n{defendants}\n"
    "【相关法条】\n{articles}\n"
    "---------------------\n"
    "请按照下列要求输出：\n"
    "1. 列出和被告有关的关键事实，≤300字。\n"
    "2. 从给定的相关法条中找出和案情最相关的部分，≤200字。\n"
    "2. 不要复述无关背景；不提及量刑、刑期。\n"
    "3. 用 JSON 数组返回，示例：\n"
    '{{"defendant":str,"key_facts":str, "key_articles":str}}'
)
def return_prompt(fact, defendants, articles):
    """
    返回用户提示模板，包含案情和被告名单。
    :param fact: 案情文本
    :param defendants: 被告名单
    :return: 格式化的用户提示字符串
    """
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
    # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
    # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
    # extra_body={"enable_thinking": False},
    )
    ipdb.set_trace()
    print(prompt)
    print(completion.choices[0].message.content.strip())
    result = json.loads(completion.choices[0].message.content.strip())
    # print(len(result["key_facts"]))
    # print(len(result["key_articles"]))
    return result

def llm_preprocess(OUT_DIR, split='train'):
    client = OpenAI(
    api_key="sk-b9283054aa5b4e089d380647972c9c59",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    ) 
    in_path = f"{OUT_DIR}/{split}_ctx.jsonl"
    out_path = f"{OUT_DIR}/{split}_llm.jsonl"
    with open(in_path, encoding="utf8") as fin, open(out_path, "w", encoding="utf8") as fout:
        for i, line in enumerate(tqdm.tqdm(fin, desc=f"llm preprocess {split}")):
            sample = json.loads(line)
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
            
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
