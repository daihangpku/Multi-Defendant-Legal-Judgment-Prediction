import os, json, tqdm, torch, random, numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoTokenizer, AutoModel,
                          get_linear_schedule_with_warmup)
from rank_bm25 import BM25Okapi
from sklearn.metrics import f1_score
import argparse
from dataloader import LawDataset
from models import MultiLabelClassifier
from torch.utils.tensorboard import SummaryWriter
import ipdb
import csv

def inference(args, model, dev_dl, device, epoch, writer=None):
    model.eval()
    all_logits, all_labels = [], []
    all_case_idx = []
    
    all_charge_num = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dev_dl, desc=f"Eval epoch {epoch}"):
            labels = batch.pop("labels").to(device)
            indices = batch.pop("idx", None)  # 如果LawDataset有idx字段
            case_idx = batch.pop("case_idx", None)  # 如果LawDataset有case_idx字段
            charge_num = batch.pop("charge_num", None)  # 如果LawDataset有change_num字段
            batch  = {k: v.to(device) for k, v in batch.items()}
            logits = torch.sigmoid(model(**batch))
            all_logits.append(logits.cpu())
            # all_labels.append(labels.cpu())
            if case_idx is not None:
                all_case_idx.append(case_idx.cpu())
            if charge_num is not None:
                all_charge_num.append(charge_num.cpu())
    all_logits = torch.cat(all_logits)

    all_case_idx = torch.cat(all_case_idx).numpy()


    # 保存预测结果
    
    
    with open(os.path.join("data/processed/subtask1/id2label.json"), "r", encoding="utf8") as f:
        id2label = json.load(f)

    pred_path = os.path.join(args.save_dir, "predictions.jsonl")
    id2accusations = {}
    last_id = all_case_idx[0]-1
    #ipdb.set_trace()
    with open(pred_path, "w", encoding="utf8") as fout:
        for i, logit in enumerate(all_logits):
            k = all_charge_num[i] if all_charge_num is not None else 1 # 多少个罪名

            topk_idx = torch.topk(logit, k).indices.tolist()
            pred = [0] * len(logit)
            txt = ""
            for cnt,idx in enumerate(topk_idx):
                pred[idx] = 1
                if cnt == 0:
                    txt = f"{id2label[str(idx)]}"
                else:
                    txt += f",{id2label[str(idx)]}"

            sentence = {
                    "idx": int(all_case_idx[i]),
                    "pred": pred,
                    "logits": logit.tolist(),
                    "txt": txt
                }
            # label = all_labels[i].int().tolist()
            fout.write(json.dumps(sentence, ensure_ascii=False) + "\n")

            case_id = int(all_case_idx[i])
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
    tokenizer  = AutoTokenizer.from_pretrained(args.backbone)
    label2id   = json.load(open(f"{args.data_dir}/label2id.json"))
    num_labels = len(label2id)

    eval_datapath = f"{args.data_dir}/test_llm.jsonl"
    all_samples = [json.loads(l) for l in open(eval_datapath, encoding="utf8")]
    total = len(all_samples)
    eval_samples = all_samples
    eval_dataset = LawDataset(samples=eval_samples, tokenizer=tokenizer, num_labels=num_labels, MAX_LEN=args.max_len, stage="test")
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiLabelClassifier(backbone=args.backbone, num_labels=num_labels).to(device)

    # 加载模型权重
    print(f"Loading model from {args.ckpt_path}...")
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    inference(args, model, eval_dataloader, device, epoch=0, )