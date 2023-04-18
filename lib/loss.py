# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.
    https://github.com/RElbers/info-nce-pytorch/

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(
        self,
        temperature=0.1,
        reduction="mean",
        negative_mode="unpaired",
        symmetric_loss=False,
    ):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode
        self.symmetric_loss = symmetric_loss

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(
            query,
            positive_key,
            negative_keys,
            temperature=self.temperature,
            reduction=self.reduction,
            negative_mode=self.negative_mode,
            symmetric_loss=self.symmetric_loss,
        )


def info_nce(
    query,
    positive_key,
    negative_keys=None,
    temperature=0.1,
    reduction="mean",
    negative_mode="unpaired",
    symmetric_loss=False,
):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError("<query> must have 2 dimensions.")
    if positive_key.dim() != 2:
        raise ValueError("<positive_key> must have 2 dimensions.")
    if negative_keys is not None:
        if negative_mode == "unpaired" and negative_keys.dim() != 2:
            raise ValueError(
                "<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'."
            )
        if negative_mode == "paired" and negative_keys.dim() != 3:
            raise ValueError(
                "<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'."
            )

    # Check matching number of samples.
    if len(query) != len(positive_key):
        print("Query shape:", query.shape)
        print("Positive key shape:", positive_key.shape)
        raise ValueError(
            "<query> and <positive_key> must must have the same number of samples."
        )
    if negative_keys is not None:
        if negative_mode == "paired" and len(query) != len(negative_keys):
            raise ValueError(
                "If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>."
            )

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError(
            "Vectors of <query> and <positive_key> should have the same number of components."
        )
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError(
                "Vectors of <query> and <negative_keys> should have the same number of components."
            )

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)

    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == "unpaired":
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == "paired":
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    if symmetric_loss:
        # TODO: consider use learned temperature
        loss_i = F.nll_loss(F.log_softmax(logits / temperature, dim=0), labels)
        loss_t = F.nll_loss(F.log_softmax(logits / temperature, dim=1), labels)
        return loss_i + loss_t
    else:
        return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

class KDSP(nn.Module):
	'''
	Similarity-Preserving Knowledge Distillation
	'''
	def __init__(self):
		super(KDSP, self).__init__()

	def forward(self, fm_s, fm_t):
		fm_s = fm_s.view(fm_s.size(0), -1)
		G_s  = torch.mm(fm_s, fm_s.t())
		norm_G_s = F.normalize(G_s, p=2, dim=1)

		fm_t = fm_t.view(fm_t.size(0), -1)
		G_t  = torch.mm(fm_t, fm_t.t())
		norm_G_t = F.normalize(G_t, p=2, dim=1)

		loss = F.mse_loss(norm_G_s, norm_G_t)

		return loss

class NST(nn.Module):
	'''
	Like What You Like: Knowledge Distill via Neuron Selectivity Transfer
	'''
	def __init__(self):
		super(NST, self).__init__()

	def forward(self, fm_s, fm_t):
		fm_s = fm_s.view(fm_s.size(0), fm_s.size(1), -1)
		fm_s = F.normalize(fm_s, dim=2)

		fm_t = fm_t.view(fm_t.size(0), fm_t.size(1), -1)
		fm_t = F.normalize(fm_t, dim=2)

		loss = self.poly_kernel(fm_t, fm_t).mean() \
			 + self.poly_kernel(fm_s, fm_s).mean() \
			 - 2 * self.poly_kernel(fm_s, fm_t).mean()

		return loss

	def poly_kernel(self, fm1, fm2):
		fm1 = fm1.unsqueeze(1)
		fm2 = fm2.unsqueeze(2)
		out = (fm1 * fm2).sum(-1).pow(2)

		return out
