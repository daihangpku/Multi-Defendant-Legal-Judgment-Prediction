import json, re, os, tqdm, jieba, argparse, collections
from rank_bm25 import BM25Okapi
import concurrent.futures
def load_articles(RAW_DIR, OUT_DIR):
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

def load_charges(RAW_DIR, OUT_DIR):
    chg = json.load(open(f'{RAW_DIR}/charges.json', encoding='utf8'))
    # 生成反向映射
    id2label = {v:k for k,v in chg.items()}
    with open(f'{OUT_DIR}/label2id.json', 'w', encoding='utf8') as f:
        json.dump(chg, f, ensure_ascii=False, indent=2)
    with open(f'{OUT_DIR}/id2label.json', 'w', encoding='utf8') as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)
    return chg