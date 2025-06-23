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
import time


def compute_pos_weight(train_ds, num_labels):
    freq = torch.zeros(num_labels)
    for s in train_ds.samples:
        for cid in s.get("charge_ids", []):
            if cid >= 0:
                freq[cid] += 1
    return (len(train_ds) - freq) / (freq + 1e-3)

def evaluate(args, model, dev_dl, device, epoch, writer=None):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm.tqdm(dev_dl, desc=f"Eval epoch {epoch}"):
            labels = batch.pop("labels").to(device)
            batch  = {k: v.to(device) for k, v in batch.items()}
            preds = model(**batch)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    mae = torch.mean(torch.abs(all_preds - all_labels)).item()
    mse = torch.mean((all_preds - all_labels) ** 2).item()
    print(f"[eval] epoch {epoch} | MAE={mae:.4f} | MSE={mse:.4f}")
    if writer:
        writer.add_scalar("eval/MAE", mae, epoch)
        writer.add_scalar("eval/MSE", mse, epoch)

def main(args):
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "logs"))
    print("loading tokenizer and model...")
    tokenizer  = AutoTokenizer.from_pretrained(args.backbone)
    train_datapath = f"{args.data_dir}/train_llm.jsonl"
    all_samples = [json.loads(l) for l in open(train_datapath, encoding="utf8")]
    total = len(all_samples)
    indices = list(range(total))
    random.shuffle(indices)
    split = int(total * 0.9)
    train_indices = indices[:split]
    eval_indices = indices[split:]
    train_samples = [all_samples[i] for i in train_indices]
    eval_samples = [all_samples[i] for i in eval_indices]
    train_dataset = LawDataset(samples=train_samples, tokenizer=tokenizer, MAX_LEN=args.max_len, stage="train")
    train_dataloader = DataLoader(train_dataset,  batch_size=args.batch_size,  shuffle=True, num_workers=4)
    eval_dataset = LawDataset(samples=eval_samples, tokenizer=tokenizer, MAX_LEN=args.max_len, stage="eval")
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    model = RegressionModel(backbone=args.backbone).to(device)
    crit = torch.nn.MSELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=int(args.warmup_ratio * len(train_dataloader) * args.epochs),
        num_training_steps=len(train_dataloader) * args.epochs,
    )
    start_epoch = 0
    if args.ckpt_path:
        print(f"Loading model from {args.ckpt_path}...")
        checkpoint = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1 
    print("start training...")
    for num_epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm.tqdm(train_dataloader, desc=f"Epoch {num_epoch}")
        epoch_loss = 0
        step = 0
        for batch in pbar:
            labels = batch.pop("labels").to(device)
            batch  = {k: v.to(device) for k, v in batch.items()}
            preds = model(**batch)
            loss   = crit(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step(); optim.zero_grad()
            pbar.set_postfix(loss=loss.item())
            epoch_loss += loss.item()
            step += 1
        avg_loss = epoch_loss / step
        print(f"[Epoch {num_epoch}] Average loss: {avg_loss:.4f}")
        writer.add_scalar("train/loss", avg_loss, num_epoch)
        if (num_epoch + 1) % args.save_interval == 0:
            print(f"Saving checkpoint for epoch {num_epoch + 1}...")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'epoch': num_epoch,
            }, f"{args.save_dir}/{num_epoch + 1}.pt")
        if (num_epoch + 1) % args.eval_interval == 0:
            print(f"Evaluating model at epoch {num_epoch + 1}...")
            evaluate(args, model, eval_dataloader, device, num_epoch, writer = writer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("train subtask1 model")
    parser.add_argument("--data_dir", default="data/processed/subtask1", help="数据目录")
    parser.add_argument("--save_dir", default="checkpoints", help="模型检查点目录")
    parser.add_argument("--backbone", default="google-bert/bert-base-chinese", help="预训练模型")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批大小")
    parser.add_argument("--lr", type=float, default=3e-5, help="学习率")
    parser.add_argument("--max_len", type=int, default=512, help="最大序列长度")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="学习率预热比例")
    parser.add_argument("--save_interval", type=int, default=10, help="保存模型的间隔轮数")
    parser.add_argument("--eval_interval", type=int, default=1, help="评估模型的间隔轮数")
    parser.add_argument("--ckpt_path", type=str, default=None, help="加载的模型检查点路径")
    args = parser.parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    args.save_dir = os.path.join(args.save_dir, f"subtask2_{timestamp}")
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
