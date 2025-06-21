import json, re, os, tqdm, jieba, argparse, collections
from rank_bm25 import BM25Okapi
import concurrent.futures
from openai import OpenAI
from clean import load_articles, load_charges
from flatten import flatten_cases, add_ctx_to_samples
from llm_preprocess import llm_preprocess

# ---------- main ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', default='test')
    parser.add_argument('--raw_dir', default='data/raw/subtask1', help='原始数据目录')
    parser.add_argument('--out_dir', default='data/processed/subtask1', help='处理后数据目录')
    parser.add_argument('--top_k', type=int, default=5, help='BM25检索的法条数量')
    args = parser.parse_args()
    RAW_DIR = args.raw_dir
    OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)
    # 预处理文章和法条
    print("Loading articles and charges...")
    # arts = load_articles(RAW_DIR=RAW_DIR, OUT_DIR=OUT_DIR)
    label2id = load_charges(RAW_DIR=RAW_DIR, OUT_DIR=OUT_DIR)


    for sp in args.splits.split(','):
        flatten_cases(sp.strip(), label2id, RAW_DIR, OUT_DIR)
        add_ctx_to_samples(
            OUT_DIR=OUT_DIR,
            split=sp.strip(),
            article_token_path=f"{OUT_DIR}/articles_token.txt",
            article_clean_path=f"{OUT_DIR}/articles_clean.json",
            top_k=args.top_k,
        )
        llm_preprocess(OUT_DIR, split=sp.strip())
