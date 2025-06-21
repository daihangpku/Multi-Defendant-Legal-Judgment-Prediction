import json, re, os, tqdm, jieba, argparse, collections
from rank_bm25 import BM25Okapi
import concurrent.futures

def flatten_cases(split, label2id, RAW_DIR, OUT_DIR):
    fin  = open(f'{RAW_DIR}/{split}.jsonl', encoding='utf8')
    fout = open(f'{OUT_DIR}/{split}.jsonl', 'w', encoding='utf8')
    missed_cnt = collections.Counter()   # 统计训练中没出现在 charges.json 的罪名

    for case_idx, line in enumerate(tqdm.tqdm(fin, desc=f'flatten {split}')):
        case = json.loads(line)
        fact = case['fact']
        for idx, name in enumerate(case['defendants']):
            sample = {
                'defendant': name,
                'fact': fact,
                'case_idx': case_idx,
            }
           
            # 训练 / 验证集才有标签
            if 'outcomes' in case:
                sample['charge_ids'] = []
                sample['imprisonment'] = []
                sample['standard_accusation'] = []
                for item in case['outcomes']:
                    if item['name'] == name:
                        #ipdb.set_trace()
                        sample["change_num"] = len(item['judgment'])
                        for single_charge in item['judgment']:
                            if single_charge['standard_accusation'] in label2id:
                                sample['charge_ids'].append(label2id[single_charge['standard_accusation']])
                                sample['standard_accusation'].append(single_charge['standard_accusation'])
                                sample['imprisonment'].append(single_charge['imprisonment'])
                            else:
                                missed_cnt.update([single_charge['standard_accusation']])
                        break
                
            fout.write(json.dumps(sample, ensure_ascii=False)+'\n')
    fout.close()
    if missed_cnt:
        print('[WARN] 未在 charges.json 里找到的罪名及次数：', missed_cnt)

def process_sample(line, bm25, article_dict, top_k):
    import jieba
    s = json.loads(line)
    query_tok = list(jieba.cut(s["fact"]))[:3000]
    top_ids = bm25.get_top_n(query_tok, bm25.doc_ids, n=top_k)
    s["ctx"] = "".join(article_dict[aid] for aid in top_ids)
    return s

def add_ctx_to_samples(OUT_DIR, split, article_token_path, article_clean_path, top_k=3, num_workers=8):
    # 加载BM25
    doc_ids, corpus_tokens = [], []
    for ln in open(article_token_path, encoding="utf8"):
        aid, seg = ln.strip().split("\t")
        doc_ids.append(aid)
        corpus_tokens.append(seg.split())
    bm25 = BM25Okapi(corpus_tokens)
    bm25.doc_ids = doc_ids
    article_dict = json.load(open(article_clean_path, encoding="utf8"))

    # 处理样本
    in_path = f'{OUT_DIR}/{split}.jsonl'
    out_path = f'{OUT_DIR}/{split}_ctx.jsonl'
    lines = [line for line in open(in_path, encoding="utf8")]
    total = len(lines)

    # 多进程处理
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor, \
         open(out_path, "w", encoding="utf8") as fout:
        # 需要将 bm25/article_dict/top_k 作为全局变量传递或用 functools.partial
        from functools import partial
        process_fn = partial(process_sample, bm25=bm25, article_dict=article_dict, top_k=top_k)
        for i, s in enumerate(tqdm.tqdm(executor.map(process_fn, lines), total=total)):
            fout.write(json.dumps(s, ensure_ascii=False) + "\n")
            if i == 0:
                print("示例处理后样本：")
                print(json.dumps(s, ensure_ascii=False, indent=2))