import torch
from transformers import AutoModel

class RegressionModel(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone)
        self.dropout = torch.nn.Dropout(0.2)
        self.regressor = torch.nn.Linear(self.encoder.config.hidden_size, 1)
        self._init_weights()

    def _init_weights(self):
        torch.nn.init.xavier_uniform_(self.regressor.weight)
        if self.regressor.bias is not None:
            torch.nn.init.zeros_(self.regressor.bias)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        x = outputs.last_hidden_state[:, 0]
        x = self.dropout(x)
        return self.regressor(x).squeeze(-1)  # 输出(batch_size,)
