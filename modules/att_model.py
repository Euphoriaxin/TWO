# modules/att_model.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

import modules.utils as utils
from modules.caption_model import CaptionModel


def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths.cpu(), batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0, len(indices)).type_as(inv_ix)
    return tmp, inv_ix


def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp


def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)


class AttModel(CaptionModel):
    def __init__(self, args, tokenizer):
        super(AttModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.idx2token)
        self.input_encoding_size = args.d_model
        self.rnn_size = args.d_ff
        self.num_layers = args.num_layers
        self.drop_prob_lm = args.drop_prob_lm
        self.max_seq_length = args.max_seq_length
        self.att_feat_size = args.d_vf
        self.att_hid_size = args.d_model

        self.bos_idx = args.bos_idx
        self.eos_idx = args.eos_idx
        self.pad_idx = args.pad_idx

        self.use_bn = args.use_bn

        self.embed = lambda x: x
        self.fc_embed = lambda x: x
        self.att_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                (nn.Linear(self.att_feat_size, self.input_encoding_size),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(self.input_encoding_size),) if self.use_bn == 2 else ())))

    def clip_att(self, att_feats, att_masks):
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        p_att_feats = self.ctx2att(att_feats)

        return fc_feats, att_feats, p_att_feats, att_masks

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, state, output_logsoftmax=1):
        xt = self.embed(it)

        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks)
        if output_logsoftmax:
            logprobs = F.log_softmax(self.logit(output), dim=1)
        else:
            logprobs = self.logit(output)

        return logprobs, state

    # def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt=None):
    #     # ★ 修复：可变默认参数改为 None
    #     if opt is None:
    #         opt = {}
    #     beam_size = opt.get('beam_size', 10)
    #     group_size = opt.get('group_size', 1)
    #     sample_n = opt.get('sample_n', 10)
    #     assert sample_n == 1 or sample_n == beam_size // group_size, 'when beam search, sample_n == 1 or beam search'
    #     batch_size = fc_feats.size(0)

    #     p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

    #     assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
    #     seq = fc_feats.new_full((batch_size * sample_n, self.max_seq_length), self.pad_idx, dtype=torch.long)
    #     seqLogprobs = fc_feats.new_zeros(batch_size * sample_n, self.max_seq_length, self.vocab_size + 1)

    #     self.done_beams = [[] for _ in range(batch_size)]

    #     state = self.init_hidden(batch_size)

    #     it = fc_feats.new_full([batch_size], self.bos_idx, dtype=torch.long)
    #     logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)

    #     p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = utils.repeat_tensors(beam_size,
    #                                                                               [p_fc_feats, p_att_feats,
    #                                                                                pp_att_feats, p_att_masks]
    #                                                                               )
    #     self.done_beams = self.beam_search(state, logprobs, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, opt=opt)
    #     for k in range(batch_size):
    #         if sample_n == beam_size:
    #             for _n in range(sample_n):
    #                 seq_len = self.done_beams[k][_n]['seq'].shape[0]
    #                 seq[k * sample_n + _n, :seq_len] = self.done_beams[k][_n]['seq']
    #                 seqLogprobs[k * sample_n + _n, :seq_len] = self.done_beams[k][_n]['logps']
    #         else:
    #             seq_len = self.done_beams[k][0]['seq'].shape[0]
    #             seq[k, :seq_len] = self.done_beams[k][0]['seq']
    #             seqLogprobs[k, :seq_len] = self.done_beams[k][0]['logps']
    #     return seq, seqLogprobs

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt=None):
        # ★ 修复：可变默认参数改为 None
        if opt is None:
            opt = {}
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        sample_n = opt.get('sample_n', 10)
        assert sample_n == 1 or sample_n == beam_size // group_size, 'when beam search, sample_n == 1 or beam search'
        batch_size = fc_feats.size(0)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = fc_feats.new_full((batch_size * sample_n, self.max_seq_length), self.pad_idx, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size * sample_n, self.max_seq_length, self.vocab_size + 1)

        self.done_beams = [[] for _ in range(batch_size)]

        state = self.init_hidden(batch_size)

        it = fc_feats.new_full([batch_size], self.bos_idx, dtype=torch.long)
        logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = utils.repeat_tensors(beam_size,
                                                                                  [p_fc_feats, p_att_feats,
                                                                                   pp_att_feats, p_att_masks]
                                                                                  )
        # ★ 新增：将 state 也按 beam_size 在第 0 维扩展，使其与特征的 B*beam 维度对齐
        state = utils.repeat_tensors(beam_size, state)

        self.done_beams = self.beam_search(state, logprobs, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, opt=opt)
        for k in range(batch_size):
            if sample_n == beam_size:
                for _n in range(sample_n):
                    seq_len = self.done_beams[k][_n]['seq'].shape[0]
                    seq[k * sample_n + _n, :seq_len] = self.done_beams[k][_n]['seq']
                    seqLogprobs[k * sample_n + _n, :seq_len] = self.done_beams[k][_n]['logps']
            else:
                seq_len = self.done_beams[k][0]['seq'].shape[0]
                seq[k, :seq_len] = self.done_beams[k][0]['seq']
                seqLogprobs[k, :seq_len] = self.done_beams[k][0]['logps']
        return seq, seqLogprobs

    def _sample(self, fc_feats, att_feats, att_masks=None, update_opts=None):
        # ★ 修复：可变默认参数改为 None
        if update_opts is None:
            update_opts = {}
        # ★ 修复：不要直接修改 self.args.__dict__，使用副本避免污染全局配置
        opt = {**self.args.__dict__, **update_opts}

        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        sample_n = int(opt.get('sample_n', 1))
        group_size = opt.get('group_size', 1)
        output_logsoftmax = opt.get('output_logsoftmax', 1)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)

        # ★ 修复：显式从 opt 读取 top_k / top_p（优先 update_opts，其次实例属性兜底）
        top_k = opt.get('top_k', getattr(self, 'top_k', 0))
        top_p = opt.get('top_p', getattr(self, 'top_p', 0.0))

        if beam_size > 1 and sample_method in ['greedy', 'beam_search']:
            return self._sample_beam(fc_feats, att_feats, att_masks, opt)
        if group_size > 1:
            return self._diverse_sample(fc_feats, att_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size * sample_n)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        if sample_n > 1:
            p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = utils.repeat_tensors(sample_n,
                                                                                      [p_fc_feats, p_att_feats,
                                                                                       pp_att_feats, p_att_masks]
                                                                                      )

        trigrams = []

        seq = fc_feats.new_full((batch_size * sample_n, self.max_seq_length), self.pad_idx, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size * sample_n, self.max_seq_length, self.vocab_size + 1)
        for t in range(self.max_seq_length + 1):
            if t == 0:
                it = fc_feats.new_full([batch_size * sample_n], self.bos_idx, dtype=torch.long)

            logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state,
                                                      output_logsoftmax=output_logsoftmax)

            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:, t - 1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            if block_trigrams and t >= 3:
                prev_two_batch = seq[:, t - 3:t - 1]
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    current = seq[i][t - 1]
                    if t == 3:
                        trigrams.append({prev_two: [current]})
                    elif t > 3:
                        if prev_two in trigrams[i]:
                            trigrams[i][prev_two].append(current)
                        else:
                            trigrams[i][prev_two] = [current]
                prev_two_batch = seq[:, t - 2:t]
                # ★ 修复：使用 logprobs.device 替代硬编码 .cuda()
                mask = torch.zeros(logprobs.size(), requires_grad=False, device=logprobs.device)
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    if prev_two in trigrams[i]:
                        for j in trigrams[i][prev_two]:
                            mask[i, j] += 1
                alpha = 2.0
                logprobs = logprobs + (mask * -0.693 * alpha)

            if t == self.max_seq_length:
                break
            # ★ 修复：显式传入 top_k / top_p
            it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, temperature, top_k=top_k, top_p=top_p)

            if t == 0:
                unfinished = it != self.eos_idx
            else:
                it[~unfinished] = self.pad_idx
                logprobs = logprobs * unfinished.unsqueeze(1).float()
                unfinished = unfinished * (it != self.eos_idx)
            seq[:, t] = it
            seqLogprobs[:, t] = logprobs
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs

    def _diverse_sample(self, fc_feats, att_feats, att_masks=None, opt=None):
        # ★ 修复：可变默认参数改为 None
        if opt is None:
            opt = {}

        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        group_size = opt.get('group_size', 1)
        diversity_lambda = opt.get('diversity_lambda', 0.5)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        trigrams_table = [[] for _ in range(group_size)]

        seq_table = [fc_feats.new_full((batch_size, self.max_seq_length), self.pad_idx, dtype=torch.long) for _ in
                     range(group_size)]
        seqLogprobs_table = [fc_feats.new_zeros(batch_size, self.max_seq_length) for _ in range(group_size)]
        state_table = [self.init_hidden(batch_size) for _ in range(group_size)]

        for tt in range(self.max_seq_length + group_size):
            for divm in range(group_size):
                t = tt - divm
                seq = seq_table[divm]
                seqLogprobs = seqLogprobs_table[divm]
                trigrams = trigrams_table[divm]
                if t >= 0 and t <= self.max_seq_length - 1:
                    if t == 0:
                        it = fc_feats.new_full([batch_size], self.bos_idx, dtype=torch.long)
                    else:
                        it = seq[:, t - 1]

                    logprobs, state_table[divm] = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats,
                                                                          p_att_masks, state_table[divm])
                    logprobs = F.log_softmax(logprobs / temperature, dim=-1)

                    if divm > 0:
                        unaug_logprobs = logprobs.clone()
                        for prev_choice in range(divm):
                            prev_decisions = seq_table[prev_choice][:, t]
                            logprobs[:, prev_decisions] = logprobs[:, prev_decisions] - diversity_lambda

                    if decoding_constraint and t > 0:
                        tmp = logprobs.new_zeros(logprobs.size())
                        tmp.scatter_(1, seq[:, t - 1].data.unsqueeze(1), float('-inf'))
                        logprobs = logprobs + tmp

                    if block_trigrams and t >= 3:
                        prev_two_batch = seq[:, t - 3:t - 1]
                        for i in range(batch_size):
                            prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                            current = seq[i][t - 1]
                            if t == 3:
                                trigrams.append({prev_two: [current]})
                            elif t > 3:
                                if prev_two in trigrams[i]:
                                    trigrams[i][prev_two].append(current)
                                else:
                                    trigrams[i][prev_two] = [current]
                        prev_two_batch = seq[:, t - 2:t]
                        # ★ 修复：使用 logprobs.device 替代硬编码 .cuda()
                        mask = torch.zeros(logprobs.size(), requires_grad=False, device=logprobs.device)
                        for i in range(batch_size):
                            prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                            if prev_two in trigrams[i]:
                                for j in trigrams[i][prev_two]:
                                    mask[i, j] += 1
                        alpha = 2.0
                        logprobs = logprobs + (mask * -0.693 * alpha)

                    it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, 1)

                    if t == 0:
                        unfinished = it != self.eos_idx
                    else:
                        unfinished = seq[:, t - 1] != self.pad_idx & seq[:, t - 1] != self.eos_idx
                        it[~unfinished] = self.pad_idx
                        unfinished = unfinished & (it != self.eos_idx)
                    seq[:, t] = it
                    seqLogprobs[:, t] = sampleLogprobs.view(-1)

        return torch.stack(seq_table, 1).reshape(batch_size * group_size, -1), torch.stack(seqLogprobs_table,
                                                                                           1).reshape(
            batch_size * group_size, -1)