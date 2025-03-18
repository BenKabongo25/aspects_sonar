# Ben Kabongo
# March 2025


import ast
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm

from aspect_models import *
from data import *
from metrics import *
from text_models import *


def save_checkpoints(
    aspect_decoder: AspectDecoder,
    optimizer: Optimizer,
    epoch: int,
    best_f1: float,
    path: str='checkpoint.pth'
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
    path: str='checkpoint.pth'
) -> Tuple[AspectDecoder, Optimizer, int, float]: 
    checkpoint = torch.load(path)
    aspect_decoder.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_f1 = checkpoint['best_f1']
    return aspect_decoder, optimizer, epoch, best_f1


def save_results(results: Dict[str, Any], path: str='results.json') -> None:
    with open(path, 'w') as f:
        json.dump(results, f)


def train(
    aspect_decoder: AspectDecoder,
    dataloader: DataLoader, 
    optimizer: Optimizer,
    aspect_criterion: AspectClassificationLoss,
    device: torch.device
) -> Dict[str, float]:
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
) -> Dict[str, Union[int, float]]:
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


def plot_infos(
    train_infos: Dict[str, List[float]],
    eval_infos: Dict[str, List[float]],
    save_dir: Optional[str]=None,
) -> None:
    for key in eval_infos:
        plt.figure(figsize=(8, 6))
        
        if key in train_infos:
            plt.plot(x=range(1, 1 + len(train_infos[key])), y=train_infos[key], label='Train')
            plt.plot(x=range(1, 1 + len(eval_infos[key])), y=eval_infos[key], label='Eval')
            plt.legend()
        else:
            plt.plot(x=range(1, 1 + len(eval_infos[key])), y=eval_infos[key])
        
        plt.xlabel('Epoch')
        plt.ylabel(key)

        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f"{key}.pdf"))


def trainer(
    aspect_decoder: AspectDecoder,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    optimizer: Optimizer,
    aspect_criterion: AspectClassificationLoss,
    device: torch.device,
    n_epochs: int=10,
    save_dir: str='decoder_absa/'
):
    checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
    results_path = os.path.join(save_dir, 'results.json')

    progress_bar = tqdm(range(1, 1 + n_epochs), desc="Training", colour="cyan")

    train_infos = {}
    eval_infos = {}
    best_f1 = .0

    for epoch in progress_bar:
        train_results = train(aspect_decoder, train_dataloader, optimizer, aspect_criterion, device)
        eval_results = eval(aspect_decoder, eval_dataloader, aspect_criterion, train_dataloader.dataset.vocab, device)

        for key, value in train_results.items():
            if key not in train_infos:
                train_infos[key] = []
            train_infos[key].append(value)

        for key, value in eval_results.items():
            if key not in eval_infos:
                eval_infos[key] = []
            eval_infos[key].append(value)
        f1 = eval_results['f1']

        if f1 > best_f1:
            best_f1 = f1
            save_checkpoints(aspect_decoder, optimizer, epoch, best_f1, path=checkpoint_path)
        save_results({'train': train_infos, 'eval': eval_infos}, path=results_path)

        desc = (
            f"Epoch {epoch}/{n_epochs} - Train Loss: {train_results['loss']:.4f} " 
            "- Val Loss: {eval_results['loss']:.4f} - Val F1: {eval_results['f1']:.4f}"
        )
        progress_bar.set_description(desc)

    return train_infos, eval_infos


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(
    train_path: str,
    eval_path: str,
    test_path: str,
    dropout: float=0.1,
    encoder_type: str='sonar',
    d_model: int=1024,
    unk_aspect_flag: bool=True,
    sonar_encoder: str='text_sonar_basic_encoder', 
    sonar_tokenizer: str='text_sonar_basic_encoder',
    sonar_source_lang: str="eng_Latn",
    sonar_dtype: torch.dtype=torch.float16,
    batch_size: int=128,
    lr: float=1e-3,
    n_epochs: int=10,
    save_dir: str='decoder_absa/',
    seed: int=42,
):
    set_seed(seed)

    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)
    test_df = pd.read_csv(test_path)

    train_df['annotations'] = train_df['annotations'].apply(ast.literal_eval)
    eval_df['annotations'] = eval_df['annotations'].apply(ast.literal_eval)
    test_df['annotations'] = test_df['annotations'].apply(ast.literal_eval)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if encoder_type == 'sonar':
        text_encoder = SonarEncoder(
            encoder=sonar_encoder, 
            tokenizer=sonar_tokenizer,
            source_lang=sonar_source_lang,
            device=device,
            dtype=sonar_dtype
        )
    else:
        raise ValueError(f"Encoder type '{encoder_type}' is not supported!")

    all_annotations = [annotation for annotations in train_df['annotations'] for annotation in annotations]
    vocab = AspectVocabulary(all_annotations, unk_aspect_flag=unk_aspect_flag)
    n_aspects = vocab.get_n_aspects()
    n_classes = vocab.get_n_classes()

    train_dataset = AspectDataset(
        vocab=vocab, text_encoder=text_encoder, texts=train_df['text'], annotations=train_df['annotations']
    )
    eval_dataset = AspectDataset(
        vocab=vocab, text_encoder=text_encoder, texts=eval_df['text'], annotations=eval_df['annotations']
    )
    test_dataset = AspectDataset(
        vocab=vocab, text_encoder=text_encoder, texts=test_df['text'], annotations=test_df['annotations']
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )

    aspect_decoder = AspectDecoder(n_aspects=n_aspects, n_classes=n_classes, d_model=d_model, dropout=dropout)
    aspect_criterion = AspectClassificationLoss()
    optimizer = torch.optim.Adam(aspect_decoder.parameters(), lr=lr)

    aspect_decoder.to(device)
    aspect_criterion.to(device)

    train_infos, eval_infos = trainer(
        aspect_decoder=aspect_decoder,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        aspect_criterion=aspect_criterion,
        device=device,
        n_epochs=n_epochs,
        checkpoint_path=checkpoint_path
    )
    
    plot_infos(train_infos, eval_infos, save_dir=plots_dir)

    aspect_decoder, _, _, _ = load_checkpoints(aspect_decoder, optimizer)
    test_infos = eval(aspect_decoder, test_dataloader, aspect_criterion, train_dataloader.dataset.vocab, device)
    save_results({'test': test_infos}, path=os.path.join(save_dir, 'test_results.json'))
    