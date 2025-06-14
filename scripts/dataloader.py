import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

class LawDataset(Dataset):
    def __init__(self, path, tokenizer, num_labels, MAX_LEN=512, stage="train"):
        self.samples = [json.loads(l) for l in open(path, encoding="utf8")]
        self.stage   = stage
        self.max_len = MAX_LEN
        self.tokenizer = tokenizer
        self.NUM_LABELS = num_labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        txt = s["fact"].replace(s["defendant"], f"[DEF]{s['defendant']}[/DEF]")
        txt += self.tokenizer.sep_token + s["ctx"]

        enc = self.tokenizer(
            txt,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}

        if self.stage != "infer":
            y = torch.zeros(self.NUM_LABELS)
            for cid in s.get("charge_ids", []):
                if cid >= 0:
                    y[cid] = 1
            item["labels"] = y
        return item