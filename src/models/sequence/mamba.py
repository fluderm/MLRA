""" Wrapper around nn.MultiheadAttention to adhere to SequenceModule interface. """

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import hydra
from src.models.sequence.base import SequenceModule, TransposedModule
import src.models.nn.utils as U

from opt_einsum import contract as einsum
from einops import rearrange, repeat
from torch.nn.functional import scaled_dot_product_attention as sdpa
import copy

from local_attention.local_attention import LocalAttention, LocalFlashAttention
from local_attention.pykeops.local_attention import LocalAttention as LocalAttention_pykeops

try:
    import pykeops
    from pykeops.torch import LazyTensor

    pykeops.set_verbose(False)
    has_pykeops = True

except ImportError:
    has_pykeops = False


from mamba_ssm import Mamba


@TransposedModule
class CudaMambaBlock(SequenceModule):
    """Wrapper for MultiheadAttention using Mamba for efficient attention processing."""
    def __init__(self, 
                 d_model, 
                 dropout=0.0, 
                 ss_state=16, 
                 d_conv=4, 
                 expand=2, 
                 *args, 
                 bias=True, 
                 causal=True, 
                 rotary=False, 
                 **kwargs):

        super().__init__()
        # Sequence model necessary attributes
        self.d_model = d_model
        self.d_output = d_model

        #self.d_k = d_model // n_heads
        #self.num_heads = n_heads
        
        self.causal = causal
        self.dropout = nn.Dropout(dropout)
        self.ss_state = ss_state
        self.d_conv = d_conv
        self.expand = expand

        if rotary:
            self.rope = RotaryEmbedding(self.d_k)

        # Initialize Mamba block
        print(f'-------------------------------')
        print(f'Mamba d_state = {self.ss_state}')
        self.mamba = Mamba(d_model = d_model,
                           d_state = self.ss_state,
                           d_conv = self.d_conv,
                           expand = self.expand,
                           dt_rank = 'auto',
                           dt_min = 0.001,
                           dt_max = 0.1,
                           dt_init = 'random',
                           dt_scale = 1.0,
                           dt_init_floor = 1e-4,
                           conv_bias = True,
                           bias = False,
                           use_fast_path = True,
                           layer_idx = None,
                           device = None,
                           dtype = None,)

    def forward(self, src, attn_mask=None, key_padding_mask=None, state=None, **kwargs):
        """
        src: (B, L, D)
        attn_mask: (B, L, L)
        """
        if key_padding_mask is not None:
            raise NotImplementedError("Key padding not implemented for now with Mamba")
        if state is not None:
            raise NotImplementedError("State not implemented for now with Mamba")

        # Apply rotary positional embeddings if used
        if hasattr(self, 'rope'):
            src = self.rope(src)

        # Process the input through the Mamba block
        src = self.mamba(src)

        # Apply dropout
        src = self.dropout(src)

        return src, None  # None to match output signature of original MultiheadAttention

    def step(self, x, state):
        raise NotImplementedError("Not implemented for now with Mamba")
