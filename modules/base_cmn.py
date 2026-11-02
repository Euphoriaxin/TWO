# modules/base_cmn.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .att_model import pack_wrapper, AttModel
from .hd_computing import HyperdimensionalComputing, HDPrototypeMemory
from .ev_moe import EV_MoEModule, DMFM, MultiViewAlignmentLoss


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

'''生成一个下三角为 True、上三角为 False 的掩码。
解码器做自回归时，当前位置只能看自己及左边的词，不能偷看右边的答案。'''
def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, hdc, hd_proj, hdc_scale=0.3):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.hdc = hdc
        self.hd_proj = hd_proj
        self.hdc_scale = hdc_scale

    def forward(self, src, tgt, src_mask, tgt_mask, hdc_memory=None):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask, hdc_memory=hdc_memory)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask, past=None, hdc_memory=None):
        embeddings = self.tgt_embed(tgt)
        txt_hd_seq = self.hdc.compute_txt_hd_seq(embeddings)
        responses = self.hd_proj(txt_hd_seq.float())
        embeddings = embeddings + self.hdc_scale * responses
        return self.decoder(embeddings, memory, src_mask, tgt_mask, past=past)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        _x = sublayer(self.norm(x))
        if type(_x) is tuple:
            return x + self.dropout(_x[0]), _x[1]
        return x + self.dropout(_x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask, past=None):
        if past is not None:
            present = [[], []]
            x = x[:, -1:]
            tgt_mask = tgt_mask[:, -1:] if tgt_mask is not None else None
            past = list(zip(past[0].split(2, dim=0), past[1].split(2, dim=0)))
        else:
            past = [None] * len(self.layers)
        for i, (layer, layer_past) in enumerate(zip(self.layers, past)):
            x = layer(x, memory, src_mask, tgt_mask, layer_past)
            if layer_past is not None:
                present[0].append(x[1][0])
                present[1].append(x[1][1])
                x = x[0]
        if past[0] is None:
            return self.norm(x)
        else:
            return self.norm(x), [torch.cat(present[0], 0), torch.cat(present[1], 0)]


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask, layer_past=None):
        m = memory
        if layer_past is None:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
            return self.sublayer[2](x, self.feed_forward)
        else:
            present = [None, None]
            x, present[0] = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, layer_past[0]))
            x, present[1] = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask, layer_past[1]))
            return self.sublayer[2](x, self.feed_forward), present


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, layer_past=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        if layer_past is not None and layer_past.shape[2] == key.shape[1] > 1:
            query = self.linears[0](query)
            key, value = layer_past[0], layer_past[1]
            present = torch.stack([key, value])
        else:
            query, key, value = [l(x) for l, x in zip(self.linears, (query, key, value))]
        if layer_past is not None and not (layer_past.shape[2] == key.shape[1] > 1):
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            present = torch.stack([key, value])
        query, key, value = [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for x in [query, key, value]]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        if layer_past is not None:
            return self.linears[-1](x), present
        else:
            return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class BaseCMN(AttModel):

    def make_model(self, tgt_vocab, hdc, hd_proj):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        model = Transformer(
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),
            Decoder(DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout), self.num_layers),
            nn.Sequential(c(position)),
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)),
            hdc,
            hd_proj,
            hdc_scale=self.hdc_scale
        )
        for p in model.parameters():
            if p.requires_grad and p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, args, tokenizer):
        super(BaseCMN, self).__init__(args, tokenizer)
        self.args = args
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.topk = args.topk

        tgt_vocab = self.vocab_size + 1

        self.hdc = HyperdimensionalComputing(
            img_feat_dim=args.d_model, txt_feat_dim=args.d_model,
            hd_dim=args.hd_dim, num_patches=args.num_patches
        )
        self.hd_memory = HDPrototypeMemory(
            vocab_size=tgt_vocab, max_seq_len=args.max_seq_length, hd_dim=args.hd_dim
        )
        self.hd_proj = nn.Linear(args.hd_dim, args.d_model)
        nn.init.normal_(self.hd_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.hd_proj.bias)
        '''随机初始化后固定不动的线性变换'''
        self.hd_proj.requires_grad_(False)

        self.hdc_scale = getattr(args, 'hdc_scale', 0.3)
        self.use_hd_memory = getattr(args, 'use_hd_memory', True)
        self.update_hd_memory = getattr(args, 'update_hd_memory', True)
        self.hd_memory_weight = getattr(args, 'hd_memory_weight', 0.2)
        self.hd_memory_temperature = getattr(args, 'hd_memory_temperature', 0.1)
        if self.hd_memory_weight < 0 or self.hd_memory_temperature <= 0:
            raise ValueError('HDC memory requires nonnegative weight and positive temperature')

        # ===== EV-MoE 开关与模块 =====
        self.use_evmoe = getattr(args, 'use_evmoe', True)
        self.use_mal = getattr(args, 'use_mal', True)
        if self.use_evmoe:
            self.ev_moe_vfem = EV_MoEModule(
                d_model=args.d_model, num_heads=args.num_heads, dropout=args.dropout
            )
            self.ev_moe_dmfm = DMFM(
                d_model=args.d_model,
                num_experts=getattr(args, 'num_experts', 4),
                d_ff=args.d_ff, dropout=args.dropout
            )
        if self.use_mal:
            self.mal = MultiViewAlignmentLoss(
                alpha=getattr(args, 'mal_alpha', 0.5),
                tau=getattr(args, 'mal_tau', 0.1)
            )
        self._last_mal_loss = None
        # =============================

        self.model = self.make_model(tgt_vocab, self.hdc, self.hd_proj)
        self.logit = nn.Linear(args.d_model, tgt_vocab)

        self.top_k = getattr(args, 'top_k_sampling', 0)
        self.top_p = getattr(args, 'top_p_sampling', 0.0)
        _lp = getattr(args, 'length_penalty', 0.0)
        self.length_penalty = f'wu_{_lp}' if _lp and _lp > 0 else ''
        self.sample_method = getattr(args, 'sample_method', 'beam_search')

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, seq, att_masks, seq_mask, img_hd = self._prepare_feature_forward(att_feats, att_masks)
        memory = self.model.encode(att_feats, att_masks)
        if fc_feats.dim() == 3:
            fc_feats = fc_feats.mean(dim=1)
        # Carry the image query through the existing beam/sample batch expansion.
        return img_hd, att_feats, memory, att_masks

    def _prepare_feature_forward(self, att_list, view_masks, seq=None):
        # ===== 1) 各视角分别投影到 d_model =====
        proj_list, mask_list = [], []
        for att in att_list:
            att = self.clip_att(att, None)[0]
            att_p = pack_wrapper(self.att_embed, att, None)
            proj_list.append(att_p)
            mask_list.append(torch.ones(att_p.size(0), att_p.size(1), device=att_p.device))

        # ===== 2) 分支：EV-MoE 或 基线 concat =====
        if self.use_evmoe:
            enhanced = self.ev_moe_vfem(proj_list, mask_list)
            fused_seq, fused_global = self.ev_moe_dmfm(enhanced, mask_list)
            # 方案A：融合特征后做 HDC
            img_hd_seq = self.hdc.compute_img_hd_seq(fused_seq.detach())
            responses = self.hd_proj(img_hd_seq.float())
            fused_seq = fused_seq + self.hdc_scale * responses
            # MAL 损失
            if self.use_mal:
                valid_enh = [enhanced[i] for i in range(len(enhanced)) if view_masks[i].sum() > 0]
                view_pooled = [v.mean(dim=1) for v in valid_enh]
                self._last_mal_loss = self.mal(fused_global, view_pooled)
            else:
                self._last_mal_loss = None
            att_masks = mask_list[0].unsqueeze(-2)   # (B,1,Np)
        else:
            # 基线：双图 concat（还原原仓库行为）
            fused_seq = torch.cat(proj_list, dim=1)  # (B, V*Np, d)
            img_hd_seq = self.hdc.compute_img_hd_seq(fused_seq.detach())
            responses = self.hd_proj(img_hd_seq.float())
            fused_seq = fused_seq + self.hdc_scale * responses
            self._last_mal_loss = None
            cat_mask = torch.cat(mask_list, dim=1)    # (B, V*Np)
            att_masks = cat_mask.unsqueeze(-2)        # (B,1,V*Np)

        # ===== 3) 后续 seq / mask 处理（与原逻辑一致）=====
        if seq is not None:
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True
            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None
        return fused_seq, seq, att_masks, seq_mask, self.hdc.bundling(img_hd_seq.detach())

    @torch.no_grad()
    def _memory_queries(self, img_hd, input_tokens):
        # No dropout or future tokens: same query in teacher forcing and sampling.
        text_hd = self.hdc.compute_txt_hd_seq(self.model.tgt_embed[0](input_tokens))
        return img_hd.unsqueeze(1) * text_hd

    def _memory_bias(self, positions, queries):
        return (self.hd_memory.compute_logits(positions, queries)
                * (self.hd_memory_weight / self.hd_memory_temperature))

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        targets = seq[:, 1:]
        att_feats, inputs, att_masks, seq_mask, img_hd = self._prepare_feature_forward(
            att_feats, att_masks, seq)
        out = self.model(att_feats, inputs, att_masks, seq_mask)
        logits = self.logit(out)
        if self.use_hd_memory:
            positions = torch.arange(inputs.size(1), device=seq.device).expand_as(inputs)
            queries = self._memory_queries(img_hd, inputs)
            if self.hd_memory_weight:
                logits = logits + self._memory_bias(positions, queries)
            # Include the first prediction and EOS, exclude everything after EOS.
            valid = torch.cat([torch.ones_like(inputs[:, :1], dtype=torch.bool),
                               inputs[:, 1:].ne(self.eos_idx).cumprod(1).bool()], dim=1)
            # Read before write: the current batch must not supply its own answer.
            if self.training and self.update_hd_memory:
                self.hd_memory.update(positions, targets, queries, valid)
        return F.log_softmax(logits, dim=-1), self._last_mal_loss

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks,
                          state, output_logsoftmax=1):
        output, state = self.core(it, fc_feats, att_feats, p_att_feats, state, att_masks)
        logits = self.logit(output)
        if self.use_hd_memory and self.hd_memory_weight:
            queries = self._memory_queries(fc_feats, it.unsqueeze(1)).squeeze(1)
            positions = torch.full_like(it, state[0].size(1) - 1)
            logits = logits + self._memory_bias(positions, queries)
        return (F.log_softmax(logits, dim=-1) if output_logsoftmax else logits), state

    def _save_attns(self, start=False):
        pass

    # def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
    #     if len(state) == 0:
    #         ys = it.unsqueeze(1)
    #     else:
    #         ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
    #     out = self.model.decode(
    #         memory, mask, ys,
    #         subsequent_mask(ys.size(1)).to(memory.device),
    #         past=None
    #     )
    #     return out[:, -1], [ys.unsqueeze(0)]

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0], it.unsqueeze(1)], dim=1)   # ← 改动：state[0][0] → state[0]
        out = self.model.decode(
            memory, mask, ys,
            subsequent_mask(ys.size(1)).to(memory.device),
            past=None
        )
        return out[:, -1], [ys]                                  # ← 改动：[ys.unsqueeze(0)] → [ys]