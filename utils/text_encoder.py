import os
from typing import Tuple, Union, List

import numpy as np
import torch


class T5TextEncoder(torch.nn.Module):
    def __init__(self, model_name: str, device: torch.device, max_length: int = 60):
        super().__init__()
        from transformers import T5Tokenizer, T5EncoderModel

        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=True)
        self.model = T5EncoderModel.from_pretrained(model_name).to(device)
        self.device = device
        self.model.eval()
        self.output_dim = self.model.config.d_model
        self.max_length = max_length

    def encode(self, texts: Union[str, List[str]]) -> Tuple[np.ndarray, np.ndarray]:
        """Encode text to token-level features.
        
        Args:
            texts: Single string or list of strings
            
        Returns:
            feats: (B, max_length, dim) or (max_length, dim) if single string
            mask: (B, max_length) or (max_length,) if single string - True=valid, False=padding
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        with torch.no_grad():
            inputs = self.tokenizer(
                texts, return_tensors="pt",
                padding="max_length", max_length=self.max_length, truncation=True
            ).to(self.device)
            outputs = self.model(**inputs)
            feats = outputs.last_hidden_state.float()  # (B, max_length, dim)
            mask = inputs.attention_mask.bool()  # (B, max_length), True=valid

        feats = feats.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()
        if single:
            return feats[0], mask[0]
        return feats, mask


def resolve_text_encoder_path(text_encoder_type: str, text_encoder: str, repo_root: str) -> str:
    """Resolve text encoder path with auto-detection."""
    if text_encoder:
        return text_encoder
    
    # Auto-detect from project text_encoders directory
    encoders_dir = os.path.join(repo_root, "text_encoders")
    
    if text_encoder_type == "t5":
        local_t5 = os.path.join(encoders_dir, "flan-t5-small")
        if os.path.isdir(local_t5):
            return local_t5
        return "google/flan-t5-small"
    
    # BGE
    local_bge = os.path.join(encoders_dir, "bge-small-en-v1.5")
    if os.path.isdir(local_bge):
        return local_bge
    return "BAAI/bge-small-en-v1.5"


def load_text_encoder(
    text_encoder_type: str, model_name_or_path: str, device: Union[str, torch.device],
    max_length: int = 60
) -> Tuple[torch.nn.Module, int]:
    device = torch.device(device)

    if text_encoder_type == "t5":
        encoder = T5TextEncoder(model_name_or_path, device, max_length=max_length)
        return encoder, encoder.output_dim

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model_name_or_path, device=device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    try:
        dim = int(encoder.get_sentence_embedding_dimension())
    except Exception:
        dim = int(encoder.encode("").shape[-1])
    return encoder, dim
