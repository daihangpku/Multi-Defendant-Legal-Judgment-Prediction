import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

class LawDataset(Dataset):
    def __init__(self, samples, tokenizer, num_labels, MAX_LEN=512, stage="train"):
        self.samples = samples
        self.stage   = stage
        self.max_len = MAX_LEN
        self.tokenizer = tokenizer
        self.NUM_LABELS = num_labels
        # Filter samples to only include those with all required keys
        required_keys = {'defendant', 'fact', 'charge_ids', 'imprisonment', 'standard_accusation', 'ctx', 'key_facts', 'key_articles', 'idx'}
        self.samples = [s for s in self.samples if required_keys.issubset(s.keys())]
        # Remove samples where 'key_facts' or 'key_articles' are not strings
        self.samples = [
            s for s in self.samples
            if isinstance(s["key_facts"], str) and isinstance(s["key_articles"], str)
        ]
        if stage=="test":
            print("loading test dataset, total samples:", len(self.samples))
            self.samples.sort(key=lambda x: x["idx"])
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        txt = s["key_facts"].replace(s["defendant"], f"[DEF]{s['defendant']}[/DEF]")
        txt += self.tokenizer.sep_token + s["key_articles"]

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
        if "change_num" in s:
            item["charge_num"] = s["change_num"]
        if "case_idx" in s:
            item["case_idx"] = s["case_idx"]

        return item