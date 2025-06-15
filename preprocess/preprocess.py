import json, re, os, tqdm, jieba, argparse, collections
from rank_bm25 import BM25Okapi
import concurrent.futures
from openai import OpenAI
from clean import load_articles, load_charges
from flatten import flatten_cases, add_ctx_to_samples
from llm_preprocess import llm_preprocess
RAW_DIR = 'data/raw/subtask1'
OUT_DIR = 'data/processed/subtask1'
os.makedirs(OUT_DIR, exist_ok=True)
import ipdb


# ---------- main ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', default='train')
    args = parser.parse_args()
    arts = load_articles()
    label2id = load_charges()
    for sp in args.splits.split(','):
        flatten_cases(sp.strip(), label2id, RAW_DIR, OUT_DIR)
        add_ctx_to_samples(
            OUT_DIR=OUT_DIR,
            split=sp.strip(),
            article_token_path=f"{OUT_DIR}/articles_token.txt",
            article_clean_path=f"{OUT_DIR}/articles_clean.json",
            top_k=3
        )
        llm_preprocess(OUT_DIR, split=sp.strip())
