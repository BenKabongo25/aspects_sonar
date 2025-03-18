# Ben Kabongo
# March 2025


import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm

from data import *
from metrics import *
from models import *


def save_checkpoints(
    aspect_decoder: AspectDecoder,
    optimizer: Optimizer,
    epoch: int,
    best_f1: float,
    path: str='aspect_decoder.pth'
) -> None:
    torch.save({
        'epoch': epoch,
        'model_state_dict': aspect_decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_f1': best_f1
    }, path)


def load_checkpoints(
    aspect_decoder: AspectDecoder,
    optimizer: Optimizer,
    path: str='aspect_decoder.pth'
) -> Tuple[AspectDecoder, Optimizer, int, float]: 
    checkpoint = torch.load(path)
    aspect_decoder.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_f1 = checkpoint['best_f1']
    return aspect_decoder, optimizer, epoch, best_f1




def train(
    aspect_decoder: AspectDecoder,
    dataloader: DataLoader, 
    optimizer: Optimizer,
    aspect_criterion: AspectClassificationLoss,
    device: torch.device
):
    aspect_decoder.train()
    total_loss = .0

    for batch in dataloader:
        ordered_class_ids = batch['ordered_class_ids'].to(device)
        text_embeddings = batch['text_embeddings'].to(device)
        
        aspect_logits = aspect_decoder(text_embeddings)
        loss = aspect_criterion(aspect_logits, ordered_class_ids)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return {'loss': total_loss / len(dataloader)}


def eval(
    aspect_decoder: AspectDecoder,
    dataloader: DataLoader,
    aspect_criterion: AspectClassificationLoss,
    vocab: AspectVocabulary,
    device: torch.device
):
    aspect_decoder.eval()
    total_loss = .0

    all_annotations = []
    all_predict_annotations = []

    with torch.no_grad():
        for batch in dataloader:
            ordered_class_ids = batch['ordered_class_ids'].to(device)
            text_embeddings = batch['text_embeddings'].to(device)
            annotations = batch['annotations']
            
            aspect_logits = aspect_decoder(text_embeddings)
            loss = aspect_criterion(aspect_logits, ordered_class_ids)

            total_loss += loss.item()

            predict_ordered_class_ids = torch.argmax(aspect_logits, dim=-1)
            predict_ordered_class_ids = predict_ordered_class_ids.detach().cpu().tolist()
            predict_annotations = [vocab.ordered_ids_to_annotations(ids) for ids in predict_ordered_class_ids]

            all_annotations.extend(annotations)
            all_predict_annotations.extend(predict_annotations)

    aspect_scores = aspect_evaluation(predictions=all_predict_annotations, references=all_annotations)
    aspect_scores['loss'] = total_loss / len(dataloader)
    return aspect_scores


def trainer(
    aspect_decoder: AspectDecoder,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    test_dataloader: DataLoader,
    optimizer: Optimizer,
    aspect_criterion: AspectClassificationLoss,
    device: torch.device,
    n_epochs: int=10
):
    progress_bar = tqdm(range(1, 1 + n_epochs), desc="Training", colour="cyan")

    logs = {}
    best_f1 = .0


    for epoch in progress_bar:
        train_results = train(aspect_decoder, train_dataloader, optimizer, aspect_criterion, device)
        val_results = eval(aspect_decoder, eval_dataloader, aspect_criterion, train_dataloader.dataset.vocab, device)
        print(f"Epoch {epoch + 1}/{n_epochs}:")
        print(f"Train Loss: {train_results['loss']:.4f}")
        print(f"Val Loss: {val_results['loss']:.4f}")
        print(f"Val Precision: {val_results['precision']:.4f}")
        print(f"Val Recall: {val_results['recall']:.4f}")
        print(f"Val F1: {val_results['f1']:.4f}")
        print()