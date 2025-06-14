import torch
from transformers import AutoModel

class MultiLabelClassifier(torch.nn.Module):
    def __init__(self, backbone, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone)
        self.dropout = torch.nn.Dropout(0.2)
        self.cls     = torch.nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # 取 [CLS] 向量
        x = outputs.last_hidden_state[:, 0]
        x = self.dropout(x)
        return self.cls(x)