"""
HuggingFace Transformers Integration
EIT Lossless wrapper for compatible models
"""

import torch
import torch.nn as nn
from typing import Optional
from .core import AdvancedEITLossless


class EITTransformerWrapper(nn.Module):
    """Wrap any HuggingFace transformer with EIT optimization"""
    
    def __init__(self, model: nn.Module, freeze_ratio: float = 0.9):
        super().__init__()
        self.model = model
        self.eit = AdvancedEITLossless(freeze_ratio=freeze_ratio)
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                freeze_embeddings: bool = True,
                **kwargs):
        """Forward pass with optional EIT freezing"""
        
        if freeze_embeddings and hasattr(self.model, 'embeddings'):
            embeddings = self.model.embeddings(input_ids)
            frozen_embeddings, _ = self.eit.freeze(embeddings)
            
            outputs = self.model.encoder(frozen_embeddings, 
                                        attention_mask=attention_mask,
                                        **kwargs)
            
            restored_output = self.eit.restore(outputs.last_hidden_state)
            outputs.last_hidden_state = restored_output
        else:
            outputs = self.model(input_ids, 
                               attention_mask=attention_mask,
                               **kwargs)
        
        return outputs