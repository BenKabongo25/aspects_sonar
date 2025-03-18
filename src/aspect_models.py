# Ben Kabongo
# March 2025


import torch
import torch.nn as nn

from typing import List


class AspectEncoder(nn.Module):

    def __init__(self, n_aspects: int, n_classes: int=4, d_model: int=1024, pooling: str='mean'):
        super().__init__()
        self.n_aspects = n_aspects
        self.n_classes = n_classes
        self.d_model = d_model
        self.pooling = pooling

        self.aspect_embeddings = nn.Embedding(n_aspects * n_classes + 1, d_model)

    def forward(self, aspect_ids: torch.LongTensor, class_ids: torch.LongTensor) -> torch.FloatTensor:
        # aspect_ids: (batch_size, seq_len)
        # class_ids: (batch_size, seq_len)
        aspect_ids = aspect_ids * self.n_classes + class_ids
        aspect_ids[aspect_ids != 0] += 1

        aspect_embeddings = self.aspect_embeddings(aspect_ids) # (batch_size, seq_len, d_model)
        mask = (aspect_ids != self.pad_idx).unsqueeze(2)
        if self.pooling == 'mean':
            aspect_embeddings = aspect_embeddings * mask
            aspect_embeddings = aspect_embeddings.sum(1) / mask.sum(1)
        else: # max pooling
            aspect_embeddings = aspect_embeddings.masked_fill(~mask, float('-inf'))
            aspect_embeddings = aspect_embeddings.max(1).values
        return aspect_embeddings


class AspectDecoder(nn.Module):

    def __init__(self, n_aspects: int, n_classes: int=4, d_model: int=1024, dropout: float=0.1):
        super().__init__()
        self.n_aspects = n_aspects
        self.n_classes = n_classes
        self.d_model = d_model

        self.shared = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

        self.aspect_heads = nn.ModuleList([nn.Linear(d_model, n_classes) for _ in range(n_aspects)])

    def forward(self, embeddings: torch.FloatTensor) -> torch.FloatTensor:
        # embeddings: (batch_size, d_model)
        shared = self.shared(embeddings)
        aspect_logits = [head(shared) for head in self.aspect_heads]
        aspect_logits = torch.stack(aspect_logits, dim=1) # (batch_size, n_aspects, n_classes)
        return aspect_logits


class AspectClassificationLoss(nn.Module):

    def __init__(self, n_aspects: int, aspect_weights: List[float]=None):
        super().__init__()
        self.n_aspects = n_aspects
        if aspect_weights is None:
            aspect_weights = [1.0] * n_aspects
        self.aspect_weights = aspect_weights
        self.losses = nn.ModuleList([nn.CrossEntropyLoss() for _ in range(n_aspects)])

    def forward(self, aspect_logits: torch.FloatTensor, ordered_class_ids: torch.LongTensor) -> torch.FloatTensor:
        # aspect_logits: (batch_size, n_aspects, n_classes)
        # ordered_class_ids: (batch_size, n_aspects)
        total_loss = .0
        for i in range(self.n_aspects):
            loss = self.losses[i](aspect_logits[:, i], ordered_class_ids[:, i])
            total_loss += self.aspect_weights[i] * loss
        return total_loss


class AspectEmbeddingLoss(nn.MSELoss):
    pass

