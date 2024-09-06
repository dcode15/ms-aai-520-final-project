from typing import Optional

import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, BartConfig


class ChatbotModel(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.bart = BartForConditionalGeneration(config)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.Tensor] = None,
            decoder_attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
    ):
        return self.bart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            return_dict=True
        )

    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        return self.bart.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
