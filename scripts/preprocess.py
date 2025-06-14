import json, re, os, tqdm, jieba, argparse, collections
from rank_bm25 import BM25Okapi
import concurrent.futures
RAW_DIR = 'data/raw/subtask1'
OUT_DIR = 'data/processed/subtask1'
os.makedirs(OUT_DIR, exist_ok=True)
import ipdb
# ---------- 1. 法条清洗 ----------
def load_articles():
    art_raw = json.load(open(f'{RAW_DIR}/articles.json', encoding='utf8'))
    art_clean = {}
    for aid, txt in art_raw.items():
        # 去掉书名号、全角空格、换行
        t = re.sub(r'[《》\s]+', '', txt)
        # 只保留条文正文（去掉“《刑法》第XXX条：”前缀）
        t = re.sub(r'^.*?条：', '', t)
        art_clean[aid] = t
    with open(f'{OUT_DIR}/articles_clean.json', 'w', encoding='utf8') as f:
        json.dump(art_clean, f, ensure_ascii=False, indent=2)
    # 预存分词结果，后面 BM25 可直接用
    with open(f'{OUT_DIR}/articles_token.txt', 'w', encoding='utf8') as f:
        for aid, t in art_clean.items():
            f.write(f'{aid}\t{" ".join(jieba.cut(t))}\n')
    return art_clean

# ---------- 2. 罪名 ↔ id 映射 ----------
def load_charges():
    chg = json.load(open(f'{RAW_DIR}/charges.json', encoding='utf8'))
    # 生成反向映射
    id2label = {v:k for k,v in chg.items()}
    with open(f'{OUT_DIR}/label2id.json', 'w', encoding='utf8') as f:
        json.dump(chg, f, ensure_ascii=False, indent=2)
    with open(f'{OUT_DIR}/id2label.json', 'w', encoding='utf8') as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)
    return chg

# ---------- 3. 案件 → 被告 展平 ----------
def flatten_cases(split, label2id):
    fin  = open(f'{RAW_DIR}/{split}.jsonl', encoding='utf8')
    fout = open(f'{OUT_DIR}/{split}.jsonl', 'w', encoding='utf8')
    missed_cnt = collections.Counter()   # 统计训练中没出现在 charges.json 的罪名
    for line in tqdm.tqdm(fin, desc=f'flatten {split}'):
        case = json.loads(line)
        fact = case['fact']
        for idx, name in enumerate(case['defendants']):
            sample = {
                'defendant': name,
                'fact': fact,
            }
            # 训练 / 验证集才有标签
            if 'outcomes' in case:
                sample['charge_ids'] = []
                sample['imprisonment'] = []
                sample['standard_accusation'] = []
                for item in case['outcomes']:
                    if item['name'] == name:
                        #ipdb.set_trace()
                        for single_charge in item['judgment']:
                            if single_charge['standard_accusation'] in label2id:
                                sample['charge_ids'].append(label2id[single_charge['standard_accusation']])
                                sample['standard_accusation'].append(single_charge['standard_accusation'])
                                sample['imprisonment'].append(single_charge['imprisonment'])
                            else:
                                missed_cnt.update([single_charge['standard_accusation']])
                        # sample['standard_accusation'] = item['judgment']['standard_accusation']
                        # sample['imprisonment'] = item['judgment']['imprisonment']
                        # charges = sample['standard_accusation'].split(';')
                        # sample['charge_ids'] = [label2id[c] if c in label2id else -1
                        #                         for c in charges]
                        # sample['charge_ids'] = sorted(sample['charge_ids'])
                        # missed_cnt.update([c for c in charges if c not in label2id])
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

def add_ctx_to_samples(split, article_token_path, article_clean_path, top_k=3, num_workers=8):
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
# ---------- main ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', default='train')
    args = parser.parse_args()
    arts = load_articles()
    label2id = load_charges()
    for sp in args.splits.split(','):
        flatten_cases(sp.strip(), label2id)
        add_ctx_to_samples(
            sp.strip(),
            article_token_path=f"{OUT_DIR}/articles_token.txt",
            article_clean_path=f"{OUT_DIR}/articles_clean.json",
            top_k=5
        )
