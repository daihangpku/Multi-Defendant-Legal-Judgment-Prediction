import torch
from transformers import AutoModel

class MultiLabelClassifier(torch.nn.Module):
    def __init__(self, backbone, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone)
        self.dropout = torch.nn.Dropout(0.2)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.encoder.config.hidden_size, num_labels),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(0.2),
            # torch.nn.Linear(512, num_labels)
        )
        self._init_weights()

    def _init_weights(self):
        # 初始化MLP层权重
        for m in self.mlp:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        x = outputs.last_hidden_state[:, 0]
        #x = outputs.last_hidden_state.mean(dim=1)
        x = self.dropout(x)
        return self.mlp(x)