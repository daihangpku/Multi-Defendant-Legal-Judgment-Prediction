import os, json, tqdm, torch, random, numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoTokenizer, AutoModel,
                          get_linear_schedule_with_warmup)
from rank_bm25 import BM25Okapi
from sklearn.metrics import f1_score
import argparse
from dataloader import LawDataset
from models import MultiLabelClassifier



def compute_pos_weight(train_ds, num_labels):
    freq = torch.zeros(num_labels)
    for s in train_ds.samples:
        for cid in s.get("charge_ids", []):
            if cid >= 0:
                freq[cid] += 1
    return (len(train_ds) - freq) / (freq + 1e-3)

def evaluate(args, model, dev_dl, device, epoch):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm.tqdm(dev_dl, desc=f"dev-{epoch}"):
            labels = batch.pop("labels").to(device)
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
    print(f"[Dev] epoch {epoch} | F1={best_f1:.4f} at t={best_t:.2f}")

    with open(f"{args.checkpoint_dir}/threshold.json", "w") as fp:
        json.dump({"t": float(best_t)}, fp)

def main(args):
    print("loading tokenizer and model...")
    
    tokenizer  = AutoTokenizer.from_pretrained(args.backbone)
    label2id   = json.load(open(f"{args.data_dir}/label2id.json"))
    id2label   = {v: k for k, v in label2id.items()}
    num_labels = len(label2id)
    # article_dict = json.load(open(f"{args.data_dir}/articles_clean.json"))
    doc_ids, corpus_tokens = [], []
    for ln in open(f"{args.data_dir}/articles_token.txt", encoding="utf8"):
        aid, seg = ln.strip().split("\t")
        doc_ids.append(aid)
        corpus_tokens.append(seg.split())
    bm25 = BM25Okapi(corpus_tokens)
    bm25.doc_ids = doc_ids
    # load data
    train_datapath = f"{args.data_dir}/train_ctx.jsonl"
    article_path = f"{args.data_dir}/articles_clean.json"
    train_dataset = LawDataset(path=train_datapath, 
                                article_path=article_path,
                                bm25=bm25,
                                tokenizer=tokenizer,
                                num_labels=num_labels,
                                stage="train"),
    train_dataloader = DataLoader(train_dataset,
                      batch_size=args.batch_size,
                      shuffle=True,
                      num_workers=4)
    

    # dev_dl   = build_dataloader("dev")
    device   = "cuda" if torch.cuda.is_available() else "cpu"

    model = MultiLabelClassifier().to(device)

    # pos_weight 
    pos_weight = compute_pos_weight(train_dataloader.dataset, num_labels).to(device)
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=int(args.warn_up_ratio * len(train_dataloader) * args.epochs),
        num_training_steps=len(train_dataloader) * args.epochs,
    )
    print("start training...")
    for num_epoch in range(args.epochs):
        model.train()
        pbar = tqdm.tqdm(train_dataloader, desc=f"train-{num_epoch}")
        epoch_loss = 0
        step = 0
        for batch in pbar:
            labels = batch.pop("labels").to(device)
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

        torch.save(model.state_dict(), f"{args.checkpoint_dir}/{num_epoch}.pt")
        # evaluate(model, dev_dl, device, ep)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed/subtask1", help="数据目录")
    parser.add_argument("--checkpoint_dir", default="checkpoints", help="模型检查点目录")
    parser.add_argument("--backbone", default="hfl/chinese-legal-electra-base-discriminator", help="预训练模型")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批大小")
    parser.add_argument("--lr", type=float, default=3e-5, help="学习率")
    parser.add_argument("--max_len", type=int, default=512, help="最大序列长度")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="学习率预热比例")
    parser.add_argument("--top_k_article", type=int, default=5, help="BM25检索的法条数量")
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    main(args)
