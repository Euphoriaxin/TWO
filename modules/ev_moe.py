# modules/ev_moe.py
"""
EV-MoE: Enhanced View Mixture-of-Experts for Multi-view Medical Report Generation

修复说明（相对于初版）：
1. [DMFM] 不再把空间特征压成全局向量再 repeat 平铺，改为在 patch 级别做 MoE 融合，
   输出保持 (B, Np, d) 空间结构，decoder 的 cross-attn 仍能做空间对齐。
2. [VFEM] 对所有 patch 做跨视角交互（不再只取第0个位置当 CLS）；
   修复残差连接：原实现把非 CLS 位置加了两次自身，现改为标准 Pre-LN 残差。
3. [DMFM] MoE 路由在 patch 级别进行（每个位置独立选专家），专家能看到差异化空间特征，
   而非只能看到一个被平均糊掉的全局向量。
4. [MAL] 修复对比损失的正负样本构造：同图像的不同视角互为正样本，
   其他图像的视角为负样本，二者不再重叠。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


'''标准的 Transformer 多头注意力机制，支持灵活的 2D / 3D 掩码（Mask）广播，可以正确处理医疗图像中不同视角 Patch 的有效性遮罩'''


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        query, key, value = [l(x) for l, x in zip(self.linears, (query, key, value))]
        query, key, value = [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for x in [query, key, value]]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # mask 统一扩成 (B, 1, 1, Lk)，与 scores (B, h, Lq, Lk) 正确广播
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)  # (B, Lk)   -> (B, 1, 1, Lk)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B, Lq, Lk) -> (B, 1, Lq, Lk)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        x = torch.matmul(p_attn, value).transpose(1, 2).contiguous()
        x = x.view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


'''经典的前馈神经网络（FFN），由两层线性变换和一个 ReLU 激活函数、Dropout 组成'''


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# ============ VFEM: 视图特定特征增强（修复版）============
class VFEM(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = MultiHeadedAttention(num_heads, d_model, dropout)
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_self, x_other, mask_self, mask_other):
        # x_self / x_other: (B, Np, d)；mask: (B, Np) 1=有效
        # ★ 修复1：对所有 patch 做跨视角交互，而非只取第0个位置当 CLS
        kv = torch.cat([x_self, x_other], dim=1)  # (B, 2Np, d)
        kv_mask = torch.cat([mask_self, mask_other], dim=1)  # (B, 2Np)
        enhanced = self.cross_attn(x_self, kv, kv, kv_mask)  # (B, Np, d)

        # ★ 修复2：标准 Pre-LN 残差（原实现非 CLS 位置被加两次自身，已修正）
        return self.norm(x_self + self.dropout(enhanced))


# ============ DMFM: 动态多视角融合 (MoE)（修复版）============
class DMFM(nn.Module):
    def __init__(self, d_model, num_experts=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            PositionwiseFeedForward(d_model, d_ff, dropout) for _ in range(num_experts)
        ])
        # ★ 修复3：router 对每个 patch 位置独立路由（输入 (B, Np, d) -> gate (B, Np, K)）
        self.router = nn.Sequential(
            nn.Linear(d_model, num_experts),
            nn.Softmax(dim=-1)
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, view_feats, view_masks):
        # view_feats: list of (B, Np, d)；view_masks: list of (B, Np)
        valid_feats = []
        pooled = []
        for f, m in zip(view_feats, view_masks):
            if m.sum() == 0:
                continue  # 跳过无效视角（如 MIMIC 单图复制的占位）
            valid_feats.append(f)
            mf = m.unsqueeze(-1).float()
            pooled.append((f * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1e-6))  # (B, d)
        if len(valid_feats) == 0:  # 极端兜底
            f, m = view_feats[0], view_masks[0]
            valid_feats.append(f)
            mf = m.unsqueeze(-1).float()
            pooled.append((f * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1e-6))

        # ★ 修复1：在 patch 级别融合（保持空间结构），不再压成全局向量再 repeat
        avg_seq = torch.stack(valid_feats, dim=0).mean(dim=0)  # (B, Np, d)

        # ★ 修复3：动态 MoE，对每个 patch 位置独立路由
        B, Np, d = avg_seq.shape
        gate = self.router(avg_seq)  # (B, Np, K)
        expert_out = torch.stack([e(avg_seq) for e in self.experts], dim=2)  # (B, Np, K, d)
        gate = gate.unsqueeze(-1)  # (B, Np, K, 1)
        moed_seq = (gate * expert_out).sum(dim=2)  # (B, Np, d)

        # 残差：保留原始平均融合信息（alpha 控制保留比例）
        fused_seq = moed_seq + self.alpha * avg_seq  # (B, Np, d)

        # 全局向量（供 MAL 使用），仍是各视角池化后平均
        fused_global = torch.stack(pooled, dim=0).mean(dim=0)  # (B, d)

        return fused_seq, fused_global


# ============ EV-MoE 容器（仅做 VFEM 编排）============
'''把送进来的 N 个视角，两两配对，轮流交给 VFEM 做“本视角看别人”的增强，最后把增强完的 N 份特征原样返回。'''


class EV_MoEModule(nn.Module):
    def __init__(self, d_model, num_heads, num_experts=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.vfem_list = nn.ModuleList([VFEM(d_model, num_heads, dropout) for _ in range(2)])

    def forward(self, view_feats, view_masks):
        n = len(view_feats)
        enhanced = []
        for i in range(n):
            others = [view_feats[j] for j in range(n) if j != i]
            other_masks = [view_masks[j] for j in range(n) if j != i]
            other_feat = torch.cat(others, dim=1)
            other_mask = torch.cat(other_masks, dim=1)
            idx = min(i, len(self.vfem_list) - 1)
            enhanced.append(self.vfem_list[idx](
                view_feats[i], other_feat, view_masks[i], other_mask))
        return enhanced


# ============ MAL: 多视角对齐损失（修复版）============
class MultiViewAlignmentLoss(nn.Module):
    def __init__(self, alpha=0.5, tau=0.1):
        super().__init__()
        self.alpha = alpha  # 重建损失权重
        self.tau = tau  # 温度系数（让相似度更尖锐）

    def forward(self, fused_global, view_pooled):
        # fused_global: (B, d) 融合后的全局表征
        # view_pooled: list of (B, d) 各视角的全局池化（已过滤无效视角）

        '''如果连一个有效视角都没有（极端脏数据），直接返回 0 损失，但保留梯度通路，避免训练崩。'''
        if len(view_pooled) == 0:
            return torch.tensor(0.0, device=fused_global.device, requires_grad=True)

        B = fused_global.size(0)
        d = fused_global.size(1)

        # 所有视角表征（含融合表征），L2 归一化
        '''views = [融合, 正位, 侧位] → V = 3'''
        views = [F.normalize(fused_global, dim=-1)] + \
                [F.normalize(v, dim=-1) for v in view_pooled]
        V = len(views)
        stacked = torch.stack(views, dim=0)  # (V, B, d)

        # 同图像掩码（对角线为 True）
        '''same_img[b1, b2] = True 表示 b1 和 b2 是同一个病人'''
        same_img = torch.eye(B, dtype=torch.bool, device=fused_global.device)

        # 对比损失主循环
        total_loss = 0.0
        '''让所有视角互相监督，而不是只逼融合向量向别人靠。
            i 遍历每一个“视角角色”：
                i=0：拿融合向量当 anchor
                i=1：拿正位当 anchor
                i=2：拿侧位当 anchor'''
        for i in range(V):
            anchor = stacked[i]  # (B, d)：第 i 个视角的所有图像
            # 相似度：anchor(b) 与所有 (视图, 图像) 的相似度
            sim = torch.matmul(anchor, stacked.reshape(V * B, d).t()) / self.tau  # (B, V*B)
            sim = sim.view(B, V, B)  # [anchor_image, view_idx, target_image]
            '''第1维：anchor 是 batch 中哪个病人
               第2维：去和哪个视角比
               第3维：去和 batch 中哪个病人比'''

            # 正样本：同图像（对角线）且非自身视角 i
            pos_mask = same_img.unsqueeze(1).expand(B, V, B).clone()
            pos_mask[:, i, :] = False  # 把 anchor 自己这个视角全部屏蔽掉
            '''如果 anchor 是“正位”，那 sim[病人b, 正位, 病人b] 就是“正位和自己比”，相似度必然是 1，属于白给的正样本。把它去掉，模型才真正去学“正位 → 侧位”、“正位 → 融合”的跨视角关系。'''

            # 负样本：不同图像的所有视角
            neg_mask = (~same_img).unsqueeze(1).expand(B, V, B)

            pos_sim = sim[pos_mask].view(B, -1)  # (B, V-1)
            neg_sim = sim[neg_mask].view(B, -1)  # (B, V*(B-1))

            logits = torch.cat([pos_sim, neg_sim], dim=1)  # (B, V-1 + V*(B-1))
            labels = torch.zeros(B, dtype=torch.long, device=fused_global.device)
            total_loss += F.cross_entropy(logits, labels)

        total_loss /= V

        # 重建损失：融合表征应保留各视角信息
        recon = sum(F.mse_loss(fused_global, vp) for vp in view_pooled) / len(view_pooled)
        return (1 - self.alpha) * total_loss + self.alpha * recon