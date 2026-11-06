"""
BERT text classifier with the same settings as the paper.

- dropout 0.2 (inside BERT and before the final layer)
- a small linear layer on top of the [CLS] token
- optional "word freeze": keep the word embeddings fixed and train the rest
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class BertClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_labels=2,
                 dropout=0.2, word_freeze=True):
        super().__init__()

        # Load BERT settings, then change the few we care about.
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = num_labels
        config.hidden_dropout_prob = dropout
        config.attention_probs_dropout_prob = dropout

        # The BERT encoder.
        self.bert = AutoModel.from_pretrained(model_name, config=config)

        # Dropout and a linear layer that turns BERT output into class scores.
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        if word_freeze:
            # Freeze only the word embeddings. The rest is still trained.
            self.bert.embeddings.word_embeddings.weight.requires_grad = False

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Run BERT, then take the [CLS] token (first position) as a summary.
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls = outputs.last_hidden_state[:, 0]

        # Turn the summary into class scores.
        return self.classifier(self.dropout(cls))
