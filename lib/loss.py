# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

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

class Attention(nn.Module):
	'''
	Paying More Attention to Attention: Improving the Performance of Convolutional
	Neural Netkworks wia Attention Transfer
	'''
	def __init__(self, p=2):
		super(Attention, self).__init__()
		self.p = p

	def forward(self, fm_s, fm_t):
		loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))

		return loss

	def attention_map(self, fm, eps=1e-6):
		am = torch.pow(torch.abs(fm), self.p)
		am = torch.sum(am, dim=1, keepdim=True)
		norm = torch.norm(am, dim=(-2, -1), keepdim=True)
		am = torch.div(am, norm+eps)

		return am

def conv1x1(in_channels, out_channels):
	return nn.Conv2d(in_channels, out_channels,
					 kernel_size=1, stride=1,
					 padding=0, bias=False)

class CC(nn.Module):
	'''
	Correlation Congruence for Knowledge Distillation
	'''
	def __init__(self, gamma, P_order):
		super(CC, self).__init__()
		self.gamma = gamma
		self.P_order = P_order

	def forward(self, feat_s, feat_t):
		corr_mat_s = self.get_correlation_matrix(feat_s)
		corr_mat_t = self.get_correlation_matrix(feat_t)

		loss = F.mse_loss(corr_mat_s, corr_mat_t)

		return loss

	def get_correlation_matrix(self, feat):
		feat = F.normalize(feat, p=2, dim=-1)
		sim_mat  = torch.matmul(feat, feat.t())
		corr_mat = torch.zeros_like(sim_mat)

		for p in range(self.P_order+1):
			corr_mat += math.exp(-2*self.gamma) * (2*self.gamma)**p / \
						math.factorial(p) * torch.pow(sim_mat, p)

		return corr_mat
	
class PKT(nn.Module):
	'''
	Learning Deep Representations with Probabilistic Knowledge Transfer
	'''
	def __init__(self):
		super(PKT, self).__init__()

	def forward(self, feat_s, feat_t, eps=1e-6):
		# Normalize each vector by its norm
		feat_s_norm = torch.sqrt(torch.sum(feat_s ** 2, dim=1, keepdim=True))
		feat_s = feat_s / (feat_s_norm + eps)
		feat_s[feat_s != feat_s] = 0

		feat_t_norm = torch.sqrt(torch.sum(feat_t ** 2, dim=1, keepdim=True))
		feat_t = feat_t / (feat_t_norm + eps)
		feat_t[feat_t != feat_t] = 0

		# Calculate the cosine similarity
		feat_s_cos_sim = torch.mm(feat_s, feat_s.transpose(0, 1))
		feat_t_cos_sim = torch.mm(feat_t, feat_t.transpose(0, 1))

		# Scale cosine similarity to [0,1]
		feat_s_cos_sim = (feat_s_cos_sim + 1.0) / 2.0
		feat_t_cos_sim = (feat_t_cos_sim + 1.0) / 2.0

		# Transform them into probabilities
		feat_s_cond_prob = feat_s_cos_sim / torch.sum(feat_s_cos_sim, dim=1, keepdim=True)
		feat_t_cond_prob = feat_t_cos_sim / torch.sum(feat_t_cos_sim, dim=1, keepdim=True)

		# Calculate the KL-divergence
		loss = torch.mean(feat_t_cond_prob * torch.log((feat_t_cond_prob + eps) / (feat_s_cond_prob + eps)))

		return loss

class RKD(nn.Module):
	'''
	Relational Knowledge Distillation
	'''
	def __init__(self, w_dist, w_angle):
		super(RKD, self).__init__()

		self.w_dist  = w_dist
		self.w_angle = w_angle

	def forward(self, feat_s, feat_t):
		loss = self.w_dist * self.rkd_dist(feat_s, feat_t) + \
			   self.w_angle * self.rkd_angle(feat_s, feat_t)

		return loss

	def rkd_dist(self, feat_s, feat_t):
		feat_t_dist = self.pdist(feat_t, squared=False)
		mean_feat_t_dist = feat_t_dist[feat_t_dist>0].mean()
		feat_t_dist = feat_t_dist / mean_feat_t_dist

		feat_s_dist = self.pdist(feat_s, squared=False)
		mean_feat_s_dist = feat_s_dist[feat_s_dist>0].mean()
		feat_s_dist = feat_s_dist / mean_feat_s_dist

		loss = F.smooth_l1_loss(feat_s_dist, feat_t_dist)

		return loss

	def rkd_angle(self, feat_s, feat_t):
		# N x C --> N x N x C
		feat_t_vd = (feat_t.unsqueeze(0) - feat_t.unsqueeze(1))
		norm_feat_t_vd = F.normalize(feat_t_vd, p=2, dim=2)
		feat_t_angle = torch.bmm(norm_feat_t_vd, norm_feat_t_vd.transpose(1, 2)).view(-1)

		feat_s_vd = (feat_s.unsqueeze(0) - feat_s.unsqueeze(1))
		norm_feat_s_vd = F.normalize(feat_s_vd, p=2, dim=2)
		feat_s_angle = torch.bmm(norm_feat_s_vd, norm_feat_s_vd.transpose(1, 2)).view(-1)

		loss = F.smooth_l1_loss(feat_s_angle, feat_t_angle)

		return loss

	def pdist(self, feat, squared=False, eps=1e-12):
		feat_square = feat.pow(2).sum(dim=1)
		feat_prod   = torch.mm(feat, feat.t())
		feat_dist   = (feat_square.unsqueeze(0) + feat_square.unsqueeze(1) - 2 * feat_prod).clamp(min=eps)

		if not squared:
			feat_dist = feat_dist.sqrt()

		feat_dist = feat_dist.clone()
		feat_dist[range(len(feat)), range(len(feat))] = 0

		return feat_dist

class FitNets(nn.Module):
	'''
	FitNets: Hints for Thin Deep Nets
	'''
	def __init__(self):
		super(FitNets, self).__init__()

	def forward(self, fm_s, fm_t):
		loss = F.mse_loss(fm_s, fm_t)

		return loss

class AB(nn.Module):
	'''
	Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons
	'''
	def __init__(self, margin):
		super(AB, self).__init__()
		self.margin = margin

	def forward(self, fm_s, fm_t):
		# fm befor activation
		loss = ((fm_s + self.margin).pow(2) * ((fm_s > -self.margin) & (fm_t <= 0)).float() +
			    (fm_s - self.margin).pow(2) * ((fm_s <= self.margin) & (fm_t > 0)).float())
		loss = loss.mean()

		return loss

"""					
class VID(nn.Module):
	'''
	Variational Information Distillation for Knowledge Transfer
	'''
	def __init__(self, in_channels, mid_channels, out_channels, init_var, eps=1e-6):
		super(VID, self).__init__()
		self.eps = eps
		self.regressor = nn.Sequential(*[
				conv1x1(in_channels, mid_channels),
				# nn.BatchNorm2d(mid_channels),
				nn.ReLU(),
				conv1x1(mid_channels, mid_channels),
				# nn.BatchNorm2d(mid_channels),
				nn.ReLU(),
				conv1x1(mid_channels, out_channels),
			])
		self.alpha = nn.Parameter(
				np.log(np.exp(init_var-eps)-1.0) * torch.ones(out_channels)
			)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			# elif isinstance(m, nn.BatchNorm2d):
			# 	nn.init.constant_(m.weight, 1)
			# 	nn.init.constant_(m.bias, 0)

	def forward(self, fm_s, fm_t):
		pred_mean = self.regressor(fm_s)
		pred_var  = torch.log(1.0+torch.exp(self.alpha)) + self.eps
		pred_var  = pred_var.view(1, -1, 1, 1)
		neg_log_prob = 0.5 * (torch.log(pred_var) + (pred_mean-fm_t)**2 / pred_var)
		loss = torch.mean(neg_log_prob)

		return loss

class CRD(nn.Module):
	'''
	Contrastive Representation Distillation
	https://openreview.net/pdf?id=SkgpBJrtvS
	includes two symmetric parts:
	(a) using teacher as anchor, choose positive and negatives over the student side
	(b) using student as anchor, choose positive and negatives over the teacher side
	Args:
		s_dim: the dimension of student's feature
		t_dim: the dimension of teacher's feature
		feat_dim: the dimension of the projection space
		nce_n: number of negatives paired with each positive BS-1 255
		nce_t: the temperature
		nce_mom: the momentum for updating the memory buffer
		n_data: the number of samples in the training set, which is the M in Eq.(19) 2...
	'''
	def __init__(self, s_dim, t_dim, feat_dim, nce_n, nce_t, nce_mom, n_data):
		super(CRD, self).__init__()
		self.embed_s = Embed(s_dim, feat_dim)
		self.embed_t = Embed(t_dim, feat_dim)
		self.contrast = ContrastMemory(feat_dim, n_data, nce_n, nce_t, nce_mom)
		self.criterion_s = ContrastLoss(n_data)
		self.criterion_t = ContrastLoss(n_data)

	def forward(self, feat_s, feat_t, idx, sample_idx):
		feat_s = self.embed_s(feat_s)
		feat_t = self.embed_t(feat_t)
		out_s, out_t = self.contrast(feat_s, feat_t, idx, sample_idx)
		loss_s = self.criterion_s(out_s)
		loss_t = self.criterion_t(out_t)
		loss = loss_s + loss_t

		return loss


class Embed(nn.Module):
	def __init__(self, in_dim, out_dim):
		super(Embed, self).__init__()
		self.linear = nn.Linear(in_dim, out_dim)

	def forward(self, x):
		x = x.view(x.size(0), -1)
		x = self.linear(x)
		x = F.normalize(x, p=2, dim=1)

		return x


class ContrastLoss(nn.Module):
	'''
	contrastive loss, corresponding to Eq.(18)
	'''
	def __init__(self, n_data, eps=1e-7):
		super(ContrastLoss, self).__init__()
		self.n_data = n_data
		self.eps = eps

	def forward(self, x):
		bs = x.size(0)
		N  = x.size(1) - 1
		M  = float(self.n_data)

		# loss for positive pair
		pos_pair = x.select(1, 0)
		log_pos  = torch.div(pos_pair, pos_pair.add(N / M + self.eps)).log_()

		# loss for negative pair
		neg_pair = x.narrow(1, 1, N)
		log_neg  = torch.div(neg_pair.clone().fill_(N / M), neg_pair.add(N / M + self.eps)).log_()

		loss = -(log_pos.sum() + log_neg.sum()) / bs

		return loss


class ContrastMemory(nn.Module):
	def __init__(self, feat_dim, n_data, nce_n, nce_t, nce_mom):
		super(ContrastMemory, self).__init__()
		self.N = nce_n
		self.T = nce_t
		self.momentum = nce_mom
		self.Z_t = None
		self.Z_s = None

		stdv = 1. / math.sqrt(feat_dim / 3.)
		self.register_buffer('memory_t', torch.rand(n_data, feat_dim).mul_(2 * stdv).add_(-stdv))
		self.register_buffer('memory_s', torch.rand(n_data, feat_dim).mul_(2 * stdv).add_(-stdv))

	def forward(self, feat_s, feat_t, idx, sample_idx):
		bs = feat_s.size(0)
		feat_dim = self.memory_s.size(1)
		n_data = self.memory_s.size(0)

		# using teacher as anchor
		weight_s = torch.index_select(self.memory_s, 0, sample_idx.view(-1)).detach()
		weight_s = weight_s.view(bs, self.N + 1, feat_dim)
		out_t = torch.bmm(weight_s, feat_t.view(bs, feat_dim, 1))
		out_t = torch.exp(torch.div(out_t, self.T)).squeeze().contiguous()

		# using student as anchor
		weight_t = torch.index_select(self.memory_t, 0, sample_idx.view(-1)).detach()
		weight_t = weight_t.view(bs, self.N + 1, feat_dim)
		out_s = torch.bmm(weight_t, feat_s.view(bs, feat_dim, 1))
		out_s = torch.exp(torch.div(out_s, self.T)).squeeze().contiguous()

		# set Z if haven't been set yet
		if self.Z_t is None:
			self.Z_t = (out_t.mean() * n_data).detach().item()
		if self.Z_s is None:
			self.Z_s = (out_s.mean() * n_data).detach().item()

		out_t = torch.div(out_t, self.Z_t)
		out_s = torch.div(out_s, self.Z_s)

		# update memory
		with torch.no_grad():
			pos_mem_t = torch.index_select(self.memory_t, 0, idx.view(-1))
			pos_mem_t.mul_(self.momentum)
			pos_mem_t.add_(torch.mul(feat_t, 1 - self.momentum))
			pos_mem_t = F.normalize(pos_mem_t, p=2, dim=1)
			self.memory_t.index_copy_(0, idx, pos_mem_t)

			pos_mem_s = torch.index_select(self.memory_s, 0, idx.view(-1))
			pos_mem_s.mul_(self.momentum)
			pos_mem_s.add_(torch.mul(feat_s, 1 - self.momentum))
			pos_mem_s = F.normalize(pos_mem_s, p=2, dim=1)
			self.memory_s.index_copy_(0, idx, pos_mem_s)

		return out_s, out_t

"""	