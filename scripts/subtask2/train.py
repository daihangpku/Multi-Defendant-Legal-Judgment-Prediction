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
import time

IMPRISONMENT_LABELS = [0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 24, 27, 30, 36, 39, 42, 48, 54, 60, 72, 84, 180]

def compute_pos_weight(train_ds, num_labels):
    freq = torch.zeros(num_labels)
    for s in train_ds.samples:
        for cid in s.get("imprisonment_labels", []):
            if cid >= 0:
                freq[cid] += 1
    return (len(train_ds) - freq) / (freq + 1e-3)

def evaluate(args, model, dev_dl, device, epoch, writer=None):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm.tqdm(dev_dl, desc=f"Eval epoch {epoch}"):
            labels = batch.pop("labels").to(device)
            charge_num = batch.pop("charge_num")
            case_idx = batch.pop("case_idx", None)
            batch  = {k: v.to(device) for k, v in batch.items()}
            logits = torch.sigmoid(model(**batch))
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    best_f1, best_t = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.05):
        pred = (all_logits > t).int()
        f1   = f1_score(all_labels.numpy(), pred.numpy(), average="micro")
        if f1 > best_f1:
            best_f1, best_t = f1, t
        writer.add_scalar(f"eval/F1_t={t:.2f}", best_f1, epoch)
    print(f"[eval] epoch {epoch} | F1={best_f1:.4f} at t={best_t:.2f}")

    with open(f"{args.save_dir}/threshold.json", "w") as fp:
        json.dump({"t": float(best_t)}, fp)

def main(args):
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "logs"))
    print("loading tokenizer and model...")
    
    tokenizer  = AutoTokenizer.from_pretrained(args.backbone)
    id2label   = args.labels
    label2id   = {}
    for i, label in enumerate(id2label):
        label2id[label] = i
    
    num_labels = len(label2id)

    train_datapath = f"{args.data_dir}/train_llm.jsonl"

    all_samples = [json.loads(l) for l in open(train_datapath, encoding="utf8")]
    selected_samples = []
    for s in all_samples:
        s["imprisonment_labels"] = [label2id.get(cid, -1) for cid in s.get("imprisonment", [])]
        if any(cid >= 0 for cid in s["imprisonment_labels"]):
            selected_samples.append(s)
    total = len(selected_samples)
    indices = list(range(total))
    random.shuffle(indices)  # 随机打乱编号
    split = int(total * 0.9)
    train_indices = indices[:split]
    eval_indices = indices[split:]

    train_samples = [selected_samples[i] for i in train_indices]
    eval_samples = [selected_samples[i] for i in eval_indices]
    train_dataset = LawDataset(samples=train_samples, tokenizer=tokenizer, num_labels=num_labels, MAX_LEN=args.max_len, stage="train")
    train_dataloader = DataLoader(train_dataset,  batch_size=args.batch_size,  shuffle=True, num_workers=4)
    
    eval_dataset = LawDataset(samples=eval_samples, tokenizer=tokenizer, num_labels=num_labels, MAX_LEN=args.max_len, stage="eval")
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # dev_dl   = build_dataloader("dev")
    device   = "cuda" if torch.cuda.is_available() else "cpu"

    model = MultiLabelClassifier(backbone=args.backbone, num_labels=num_labels).to(device)

    # pos_weight 
    pos_weight = compute_pos_weight(train_dataloader.dataset, num_labels).to(device)
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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
            charge_num = batch.pop("charge_num")
            case_idx = batch.pop("case_idx", None)
            batch  = {k: v.to(device) for k, v in batch.items()}

            logits = model(**batch)
            loss   = crit(logits, labels)

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
                # 还可以加其他内容，比如学习率调度器等
            }, f"{args.save_dir}/{num_epoch + 1}.pt")

        if (num_epoch + 1) % args.eval_interval == 0:
            print(f"Evaluating model at epoch {num_epoch + 1}...")
            evaluate(args, model, eval_dataloader, device, num_epoch, writer = writer)

IMPRISONMENT_LABELS = [0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 24, 27, 30, 36, 39, 42, 48, 54, 60, 72, 84, 180]
if __name__ == "__main__":
    parser = argparse.ArgumentParser("train subtask2 model")
    parser.add_argument("--data_dir", default="data/processed/subtask2", help="数据目录")
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
    parser.add_argument("--labels", type=list, default=IMPRISONMENT_LABELS, help="标签列表")
    args = parser.parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    args.save_dir = os.path.join(args.save_dir, f"subtask2_{timestamp}")
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
