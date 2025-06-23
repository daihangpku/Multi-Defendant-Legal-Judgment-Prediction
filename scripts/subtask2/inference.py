import os, json, tqdm, torch, random, numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoTokenizer, AutoModel,
                          get_linear_schedule_with_warmup)
from rank_bm25 import BM25Okapi
from sklearn.metrics import f1_score
import argparse
from dataloader import LawDataset
from models import RegressionModel
from torch.utils.tensorboard import SummaryWriter
import ipdb
import csv

def inference(args, model, dev_dl, device, epoch, writer=None):
    model.eval()
    all_preds = []
    all_case_idx = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dev_dl, desc=f"Inference"):
            idx = batch.pop("idx", None)
            charge_num = batch.pop("charge_num", None)
            case_idx = batch.pop("case_idx", None)  
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model(**batch)
            preds = preds.cpu().numpy()
            all_preds.extend(preds.tolist())
            if case_idx is not None:
                all_case_idx.extend(case_idx.numpy().tolist())
    # 保存预测结果
    csv_path = os.path.join(args.save_dir, "predictions_imprisonment.csv")
    with open(csv_path, "w", encoding="utf8", newline="") as f:
        f.write("id,imprisonment\n")
        for idx, pred in zip(all_case_idx, all_preds):
            # 输出为整数
            f.write(f"{idx},{max(int(round(pred), 0))}\n")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed/subtask1/", help="数据目录")
    parser.add_argument("--backbone", default="google-bert/bert-base-chinese", help="预训练模型")
    parser.add_argument("--batch_size", type=int, default=1, help="批大小")
    parser.add_argument("--max_len", type=int, default=512, help="最大序列长度")
    parser.add_argument("--ckpt_path", type=str, required=True, help="加载的模型检查点路径")
    parser.add_argument("--save_dir", type=str, default=None, help="模型检查点目录")
    args = parser.parse_args()
    
    if args.save_dir is None:
        args.save_dir = os.path.dirname(args.ckpt_path)
    os.makedirs(args.save_dir, exist_ok=True)
    # 加载tokenizer和数据
    tokenizer  = AutoTokenizer.from_pretrained(args.backbone)
    label2id   = json.load(open(f"{args.data_dir}/label2id.json"))
    num_labels = len(label2id)

    eval_datapath = f"{args.data_dir}/test_llm.jsonl"
    all_samples = [json.loads(l) for l in open(eval_datapath, encoding="utf8")]
    total = len(all_samples)
    eval_samples = all_samples
    eval_dataset = LawDataset(samples=eval_samples, tokenizer=tokenizer, MAX_LEN=args.max_len, stage="infer")
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    model = RegressionModel(backbone=args.backbone).to(device)

    # 加载模型权重
    print(f"Loading model from {args.ckpt_path}...")
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    inference(args, model, eval_dataloader, device, epoch=0, )