# Ben Kabongo
# March 2025


import torch
import torch.nn as nn

from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline

from typing import List


class TextBaseEncoder(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, texts: List[str]) -> torch.FloatTensor:
        raise NotImplementedError
    

class TextBaseDecoder(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, embeddings: torch.FloatTensor) -> List[str]:
        raise NotImplementedError
    

class SonarEncoder(TextBaseEncoder):

    def __init__(self, 
        encoder: str='text_sonar_basic_encoder', 
        tokenizer: str='text_sonar_basic_encoder',
        source_lang: str="eng_Latn",
        device: torch.device=torch.device("cuda"),
        dtype: torch.dtype=torch.float16
    ):
        super().__init__()
        self.model = TextToEmbeddingModelPipeline(
            encoder=encoder, 
            tokenizer=tokenizer,
            device=device,
            dtype=dtype
        )
        self.source_lang = source_lang

    def forward(self, texts: List[str]) -> torch.FloatTensor:
        return self.model.predict(texts, source_lang=self.source_lang)
    

class SonarDecoder(TextBaseDecoder):

    def __init__(self, 
        decoder: str='text_sonar_basic_encoder', 
        tokenizer: str='text_sonar_basic_encoder',
        target_lang: str="eng_Latn",
        max_seq_len: int=128,
        device: torch.device=torch.device("cuda"),
        dtype: torch.dtype=torch.float16
    ):
        super().__init__()
        self.model = EmbeddingToTextModelPipeline(
            decoder=decoder, 
            tokenizer=tokenizer,
            device=device,
            dtype=dtype
        )
        self.target_lang = target_lang
        self.max_seq_len = max_seq_len

    def forward(self, embeddings: torch.FloatTensor) -> List[str]:
        return self.model.predict(embeddings, target_lang=self.target_lang, max_seq_len=self.max_seq_len)
    