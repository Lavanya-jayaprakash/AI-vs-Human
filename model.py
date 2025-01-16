import torch
from torch import nn
from transformers import BertModel, BertConfig

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        # Include attention dropout in the configuration
        config = BertConfig.from_pretrained(bert_model_name)
        self.bert = BertModel.from_pretrained(bert_model_name, config=config)
        # self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        # x = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits