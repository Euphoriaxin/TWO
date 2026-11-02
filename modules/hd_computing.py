# modules/hd_computing.py
import torch
import torch.nn as nn
import numpy as np


class LSHProjection(nn.Module):
    """LSH投影层：随机投影 + sign 二值化，将低维实值映射到高维二值空间"""

    def __init__(self, input_dim, hd_dim=5000):
        super().__init__()
        self.hd_dim = hd_dim
        # input_dim 对应 CMN 的特征维度（默认 512，与 d_model 对齐）
        self.projection_matrix = nn.Parameter(
            torch.randn(hd_dim, input_dim, dtype=torch.float16) / np.sqrt(input_dim),
            requires_grad=False
        )

    def forward(self, x):
        # x: (B, S, input_dim) 或 (B, input_dim)
        x = x.to(torch.float16)
        orig_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])                     # (N, input_dim)
        proj = torch.matmul(x_flat, self.projection_matrix.T)   # (N, hd_dim)
        proj = proj.reshape(*orig_shape[:-1], self.hd_dim)      # 恢复形状
        return torch.sign(proj + 1e-8)


class HyperdimensionalComputing(nn.Module):
    """超维计算：适配 CMN 的 512 维特征空间"""

    def __init__(self, img_feat_dim=512, txt_feat_dim=512, hd_dim=5000, num_patches=16):
        super().__init__()
        self.img_lsh = LSHProjection(input_dim=img_feat_dim, hd_dim=hd_dim)
        self.txt_lsh = LSHProjection(input_dim=txt_feat_dim, hd_dim=hd_dim)
        self.hd_dim = hd_dim
        self.num_patches = num_patches
        # 为每个 patch 位置生成固定的随机超维向量（二值化）
        self.patch_pos_vectors = nn.Parameter(
            torch.sign(torch.randn(self.num_patches, hd_dim, dtype=torch.float16) + 1e-8),
            requires_grad=False
        )

    def binding(self, a, b):
        """修复 batch bug：显式处理 2D/3D 广播"""
        if a.dim() == 3 and b.dim() == 2:
            min_patch = min(a.shape[1], b.shape[0])
            a = a[:, :min_patch, :]
            b = b[:min_patch, :].unsqueeze(0)
        elif a.dim() == 2 and b.dim() == 2:
            if a.shape[0] != b.shape[0]:
                m = min(a.shape[0], b.shape[0])
                a, b = a[:m, :], b[:m, :]
        elif a.dim() == 3 and b.dim() == 3:
            if a.shape[1] != b.shape[1]:
                m = min(a.shape[1], b.shape[1])
                a, b = a[:, :m, :], b[:, :m, :]
        return torch.clamp(a * b, -500, 500).to(torch.float16)

    def bundling(self, vectors):
        """vectors:(B,S,D)->(B,D) 或 (B,D)->(B,D)"""
        if vectors.dim() == 3:
            return torch.sign(torch.sum(vectors, dim=1) + 1e-8).to(torch.float16)
        return torch.sign(torch.sum(vectors, dim=0) + 1e-8).to(torch.float16)

    # ---- 序列级（用于记忆响应注入，保留位置信息）----
    def compute_img_hd_seq(self, img_patch_feats):
        """(B,S,512) -> (B,S,hd_dim)，保持序列长度与输入一致（不截断）"""
        x = img_patch_feats.half()
        S = x.shape[1]
        # 获取与序列长度匹配的 patch 位置向量（超出池大小则循环复用）
        if S <= self.num_patches:
            pos = self.patch_pos_vectors[:S, :]
        else:
            reps = (S + self.num_patches - 1) // self.num_patches
            pos = self.patch_pos_vectors.repeat(reps, 1)[:S, :]
        proj = self.img_lsh(x)                       # (B, S, hd)
        bound = self.binding(proj, pos)              # (B, S, hd)
        return bound

    def compute_txt_hd_seq(self, txt_feats):
        """(B,T,512) -> (B,T,hd_dim)"""
        return self.txt_lsh(txt_feats.half())

    # ---- 句子级（用于原型更新/检索）----
    def compute_img_hd(self, img_patch_feats):
        seq = self.compute_img_hd_seq(img_patch_feats)
        return self.bundling(seq)

    def compute_txt_hd(self, txt_feats):
        seq = self.compute_txt_hd_seq(txt_feats)
        return self.bundling(seq)

    def compute_combined_hd(self, img_patch_feats, txt_feats):
        img_hd = self.compute_img_hd(img_patch_feats)
        txt_hd = self.compute_txt_hd(txt_feats)
        return self.binding(img_hd, txt_hd)


class HDPrototypeMemory(nn.Module):
    """原型记忆：用 register_buffer 替代 nn.Parameter，消除 non-leaf 警告"""

    def __init__(self, vocab_size, max_seq_len, hd_dim=5000):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.hd_dim = hd_dim
        # ★ 改为 register_buffer：非学习参数，但参与 state_dict 存取，优化器不可见
        self.register_buffer(
            'prototype',
            torch.zeros(self.max_seq_len, self.vocab_size, self.hd_dim, dtype=torch.float16) + 1e-8
        )

    def update(self, seq_pos, token_ids, combined_hd):
        # ★ OOM 修复：detach 双保险，确保无计算图
        combined_hd = combined_hd.detach().to(torch.float16)
        valid_mask = (seq_pos > 0) & (seq_pos < self.max_seq_len) & (token_ids >= 0) & (token_ids < self.vocab_size)
        if valid_mask.sum() == 0:
            return
        idx_pos = seq_pos[valid_mask]
        idx_tok = token_ids[valid_mask]
        self.prototype[idx_pos, idx_tok] += combined_hd[valid_mask]
        self.prototype[idx_pos, idx_tok] = torch.clamp(self.prototype[idx_pos, idx_tok], -100, 100)

    def binarize(self):
        self.prototype.data = torch.sign(self.prototype.data + 1e-8).to(torch.float16)

    def compute_logits(self, seq_pos, query_hd):
        seq_pos = torch.clamp(seq_pos, 1, self.max_seq_len - 1)
        pos_prototypes = self.prototype[seq_pos, :, :].to(torch.float16)
        dot_product = torch.bmm(
            pos_prototypes, query_hd.unsqueeze(2)
        ).squeeze(2).to(torch.float32)
        return torch.clamp(100 - torch.abs(self.hd_dim - dot_product) / self.hd_dim * 100, 0, 100)