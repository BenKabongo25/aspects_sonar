# Ben Kabongo
# March 2025


import re
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Optional, Tuple

from text_models import TextBaseEncoder


PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
PAD_IDX = 0
UNK_IDX = 1


def preprocess_text(text: str, max_length: int=-1) -> str:
    text = str(text).strip()
    text = text.lower()
    text = re.sub('( )+', ' ', text)
    text = re.sub(r" \'(s|m|ve|d|ll|re)", r"'\1", text)
    text = re.sub(r" \(", "(", text)
    text = re.sub(r" \)", ")", text)
    text = re.sub(r" ,", ",", text)
    text = re.sub(r" \.", ".", text)
    text = re.sub(r" !", "!", text)
    text = re.sub(r" \?", "?", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    if max_length > 0:
        text = str(text).strip().split()
        if len(text) > max_length:
            text = text[:max_length]
        text = " ".join(text)
    return text


class AspectVocabulary:

    def __init__(self, all_annotations: List[Tuple[str]], unk_aspect_flag: bool=True):
        self.aspects = set()
        self.classes = set()

        for annotations in all_annotations:
            for annotation in annotations:
                aspect, class_ = annotation
                self.aspects.add(aspect)
                self.classes.add(class_)

        self.aspects = list(self.aspects)
        self.classes = list(self.classes)
        self.aspects.insert(PAD_IDX, PAD_TOKEN)
        self.classes.insert(PAD_IDX, UNK_TOKEN)

        self.unk_aspect_flag = unk_aspect_flag
        if self.unk_aspect_flag:
            self.aspects.insert(UNK_IDX, UNK_TOKEN)

        self.n_aspects = len(self.aspects) - 1
        self.n_classes = len(self.classes)

    def get_n_aspects(self) -> int:
        return self.n_aspects
    
    def get_n_classes(self) -> int:
        return self.n_classes
    
    def get_aspects(self) -> List[str]:
        return self.aspects
    
    def get_classes(self) -> List[str]:
        return self.classes
    
    def aspect_to_id(self, aspect: str) -> int:
        return self.aspects.index(aspect)
    
    def class_to_id(self, class_: str) -> int:
        return self.classes.index(class_)
    
    def id_to_aspect(self, id_: int) -> str:
        return self.aspects[id_]
    
    def id_to_class(self, id_: int) -> str:
        return self.classes[id_]
    
    def annotation_to_ids(self, annotation: Tuple[str]) -> List[int]:
        aspect, class_ = annotation
        return self.aspect_to_id(aspect), self.class_to_id(class_)
    
    def ids_to_annotation(self, aspect_id: int, class_id: int) -> Tuple[str]:
        aspect = self.id_to_aspect(aspect_id)
        class_ = self.id_to_class(class_id)
        return aspect, class_
    
    def annotations_to_ids(self, annotations: List[Tuple[str]]) -> Tuple[List[int]]:
        aspect_ids = []
        class_ids = []

        for aspect, class_ in annotations:
            if aspect not in self.aspects:
                if self.unk_aspect_flag:
                    aspect = '<unk>'
                else:
                    continue # ignore unknown aspects
            aspect_ids.append(self.aspect_to_id(aspect))
            class_ids.append(self.class_to_id(class_))
                
        return aspect_ids, class_ids
    
    def ids_to_annotations(self, aspect_ids: List[int], class_ids: List[int], ignore_unk_class: bool=True) -> List[Tuple[str]]:
        aspects = []
        classes = []

        for aspect_id, class_id in zip(aspect_ids, class_ids):
            if class_id == PAD_IDX and ignore_unk_class:
                continue # ignore unknown classes
            aspects.append(self.id_to_aspect(aspect_id))
            classes.append(self.id_to_class(class_id))

        return list(zip(aspects, classes))
    
    def ordered_ids_to_annotations(self, ordered_class_ids: List[int]) -> List[Tuple[str]]:
        assert len(ordered_class_ids) == self.n_aspects, 'The number of aspects does not match the number of class ids!'
        aspects = list(self.aspects[1:])
        classes = [self.id_to_class(class_id) for class_id in ordered_class_ids]
        return list(zip(aspects, classes))


class AspectDataset(Dataset):

    def __init__(
        self, 
        vocab: AspectVocabulary,
        text_encoder: TextBaseEncoder,
        texts: List[str], 
        annotations: List[List[Tuple[str]]], 
        aspect_ids: Optional[List[List[int]]]=None,
        class_ids: Optional[List[List[int]]]=None,
        text_embeddings: Optional[torch.FloatTensor]=None,
        max_length: int=-1
    ):
        super().__init__()
        self.vocab = vocab
        self.text_encoder = text_encoder
        self.texts = texts
        self.annotations = annotations
        self.text_embeddings = text_embeddings
        self.aspect_ids = aspect_ids
        self.class_ids = class_ids
        self.max_length = max_length
        self._preprocess()

    def _preprocess(self) -> None:
        if self.text_embeddings is None:
            assert self.text_encoder is not None, 'Text encoder is required to encode texts!'

        if aspect_ids is None and class_ids is None:
            annotation_flag = True
            self.aspect_ids = []
            self.class_ids = []

        text_embeddings = []
        for i in range(len(self.texts)):
            self.texts[i] = preprocess_text(self.texts[i], max_length=self.max_length)
            with torch.no_grad():
                text_embedding = self.text_encoder(self.texts[i])
            text_embeddings.append(text_embedding)

            if annotation_flag:
                annotations = self.annotations[i]
                aspect_ids, class_ids = self.vocab.annotations_to_ids(annotations)
                self.aspect_ids.append(aspect_ids)
                self.class_ids.append(class_ids)

        self.text_embeddings = torch.cat(text_embeddings, dim=0)

    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> dict:
        _out = {}
        _out['text'] = self.texts[idx]
        _out['annotations'] = self.annotations[idx]

        _out['aspect_ids'] = torch.LongTensor(self.aspect_ids[idx])
        _out['class_ids'] = torch.LongTensor(self.class_ids[idx])
        ordered_class_ids = torch.zeros(self.vocab.get_n_aspects(), dtype=torch.long)
        ordered_class_ids[_out['aspect_ids']] = _out['class_ids']
        _out['ordered_class_ids'] = ordered_class_ids

        if self.text_embeddings is not None:
            _out['text_embeddings'] = self.text_embeddings[idx]

        return _out
    

def collate_fn(batch: List[dict]) -> dict:
    _out = {}
    _out['texts'] = [sample['text'] for sample in batch]
    _out['annotations'] = [sample['annotations'] for sample in batch]

    _out['aspect_ids'] = pad_sequence([sample['aspect_ids'] for sample in batch], batch_first=True, padding_value=PAD_IDX)
    _out['class_ids'] = pad_sequence([sample['class_ids'] for sample in batch], batch_first=True, padding_value=PAD_IDX)
    _out['ordered_class_ids'] = torch.stack([sample['ordered_class_ids'] for sample in batch], dim=0)

    if 'text_embeddings' in batch[0]:
        _out['text_embeddings'] = torch.stack([sample['text_embeddings'] for sample in batch], dim=0)

    return _out
