# Ben Kabongo
# March 2025


import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import Optimizer

from data import *
from aspect_models import *


def train_epoch(
    aspect_encoder: AspectEncoder,
    aspect_decoder: AspectDecoder,
    dataloader: DataLoader, 
    encoder_optimizer: Optimizer, 
    decoder_optimizer: Optimizer,
    embedding_criterion: AspectEmbeddingLoss,
    aspect_criterion: AspectClassificationLoss,
    text_criterion: AspectClassificationLoss,
    device: torch.device,
    alpha: float=1.0,
    beta: float=1.0,
    gamma: float=1.0,
) -> dict:
    aspect_encoder.train()
    aspect_decoder.train()

    total_loss = .0
    total_embedding_loss = .0
    total_aspect_loss = .0
    total_text_loss = .0

    for batch in dataloader:
        aspect_ids = batch['aspect_ids'].to(device)
        class_ids = batch['class_ids'].to(device)
        ordered_class_ids = batch['ordered_class_ids'].to(device)
        text_embeddings = batch['text_embeddings'].to(device)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        
        aspect_embeddings = aspect_encoder(aspect_ids, class_ids)
        loss = .0
        if alpha > 0:
            embedding_loss = embedding_criterion(aspect_embeddings, text_embeddings)
            total_embedding_loss += embedding_loss.item()
            loss += alpha * embedding_loss

        if beta > 0:
            aspect_logits = aspect_decoder(aspect_embeddings)
            aspect_loss = aspect_criterion(aspect_logits, ordered_class_ids)
            total_aspect_loss += aspect_loss.item()
            loss += beta * aspect_loss

        if gamma > 0:
            text_logits = aspect_decoder(text_embeddings)
            text_loss = text_criterion(text_logits, class_ids)
            total_text_loss += text_loss.item()
            loss += gamma * text_loss

        total_loss += loss.item()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

    total_loss /= len(dataloader)
    total_embedding_loss /= len(dataloader)
    total_aspect_loss /= len(dataloader)
    total_text_loss /= len(dataloader)

    return {
        'total_loss': total_loss,
        'embedding_loss': total_embedding_loss,
        'aspect_loss': total_aspect_loss,
        'text_loss': total_text_loss,
    }
