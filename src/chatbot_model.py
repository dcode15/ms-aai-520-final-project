from transformers import GPT2LMHeadModel, GPT2Config


class ChatbotModel(GPT2LMHeadModel):
    def __init__(self, config: GPT2Config):
        super().__init__(config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
