import torch
from transformers import AutoModel

class MultiLabelClassifier(torch.nn.Module):
    def __init__(self, backbone, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone)
        self.dropout = torch.nn.Dropout(0.2)
        self.cls     = torch.nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        x = self.encoder(input_ids, attention_mask=attention_mask).pooler_output
        x = self.dropout(x)
        return self.cls(x)