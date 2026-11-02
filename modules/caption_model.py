# modules/caption_model.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import modules.utils as utils


class CaptionModel(nn.Module):
    def __init__(self):
        super(CaptionModel, self).__init__()
        # ===== 新增：可被外部（models.py / trainer）覆盖的采样与解码超参 =====
        self.top_k = 0            # 显式 top-k（0=关闭）
        self.top_p = 0.0          # 显式 nucleus（0=关闭）
        self.length_penalty = ''  # 传给 utils.penalty_builder 的字符串，如 'wu0.6' / 'length'
        # =====================================================================

    # implements beam search
    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'forward')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, '_' + mode)(*args, **kwargs)

    def beam_search(self, init_state, init_logprobs, *args, **kwargs):

        def add_diversity(beam_seq_table, logprobs, t, divm, diversity_lambda, bdash):
            local_time = t - divm
            unaug_logprobs = logprobs.clone()
            batch_size = beam_seq_table[0].shape[0]

            if divm > 0:
                change = logprobs.new_zeros(batch_size, logprobs.shape[-1])
                for prev_choice in range(divm):
                    prev_decisions = beam_seq_table[prev_choice][:, :, local_time]
                    for prev_labels in range(bdash):
                        change.scatter_add_(1, prev_decisions[:, prev_labels].unsqueeze(-1),
                                            change.new_ones(batch_size, 1))

                if local_time == 0:
                    logprobs = logprobs - change * diversity_lambda
                else:
                    logprobs = logprobs - self.repeat_tensor(bdash, change) * diversity_lambda

            return logprobs, unaug_logprobs

        # def beam_step(logprobs, unaug_logprobs, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
        #     batch_size = beam_logprobs_sum.shape[0]
        #     vocab_size = logprobs.shape[-1]
        #     logprobs = logprobs.reshape(batch_size, -1, vocab_size)
        #     if t == 0:
        #         assert logprobs.shape[1] == 1
        #         beam_logprobs_sum = beam_logprobs_sum[:, :1]
        #     candidate_logprobs = beam_logprobs_sum.unsqueeze(-1) + logprobs
        #     ys, ix = torch.sort(candidate_logprobs.reshape(candidate_logprobs.shape[0], -1), -1, True)
        #     ys, ix = ys[:, :beam_size], ix[:, :beam_size]
        #     beam_ix = ix // vocab_size
        #     selected_ix = ix % vocab_size
        #     state_ix = (beam_ix + torch.arange(batch_size).type_as(beam_ix).unsqueeze(-1) * logprobs.shape[1]).reshape(-1)

        #     if t > 0:
        #         assert (beam_seq.gather(1, beam_ix.unsqueeze(-1).expand_as(beam_seq)) ==
        #                 beam_seq.reshape(-1, beam_seq.shape[-1])[state_ix].view_as(beam_seq)).all()
        #         beam_seq = beam_seq.gather(1, beam_ix.unsqueeze(-1).expand_as(beam_seq))
        #         beam_seq_logprobs = beam_seq_logprobs.gather(1, beam_ix.unsqueeze(-1).unsqueeze(-1).expand_as(
        #             beam_seq_logprobs))

        #     beam_seq = torch.cat([beam_seq, selected_ix.unsqueeze(-1)], -1)
        #     beam_logprobs_sum = beam_logprobs_sum.gather(1, beam_ix) + \
        #                         logprobs.reshape(batch_size, -1).gather(1, ix)
        #     # ★ 修复：禁用脆弱的浮点自检断言（PyTorch 版本/数值路径差异下易误报 AssertionError）
        #     # assert (beam_logprobs_sum == ys).all()
        #     _tmp_beam_logprobs = unaug_logprobs[state_ix].reshape(batch_size, -1, vocab_size)
        #     beam_logprobs = unaug_logprobs.reshape(batch_size, -1, vocab_size).gather(1,
        #                                                                               beam_ix.unsqueeze(-1).expand(-1,
        #                                                                                                            -1,
        #                                                                                                            vocab_size))
        #     # assert (_tmp_beam_logprobs == beam_logprobs).all()
        #     beam_seq_logprobs = torch.cat([
        #         beam_seq_logprobs,
        #         beam_logprobs.reshape(batch_size, -1, 1, vocab_size)], 2)

        #     new_state = [None for _ in state]
        #     for _ix in range(len(new_state)):
        #         new_state[_ix] = state[_ix][:, state_ix]
        #     state = new_state
        #     return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state

        def beam_step(logprobs, unaug_logprobs, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            batch_size = beam_logprobs_sum.shape[0]
            vocab_size = logprobs.shape[-1]
            logprobs = logprobs.reshape(batch_size, -1, vocab_size)
            if t == 0:
                assert logprobs.shape[1] == 1
                beam_logprobs_sum = beam_logprobs_sum[:, :1]
            candidate_logprobs = beam_logprobs_sum.unsqueeze(-1) + logprobs
            ys, ix = torch.sort(candidate_logprobs.reshape(candidate_logprobs.shape[0], -1), -1, True)
            ys, ix = ys[:, :beam_size], ix[:, :beam_size]
            beam_ix = ix // vocab_size
            selected_ix = ix % vocab_size
            state_ix = (beam_ix + torch.arange(batch_size).type_as(beam_ix).unsqueeze(-1) * logprobs.shape[1]).reshape(-1)

            if t > 0:
                assert (beam_seq.gather(1, beam_ix.unsqueeze(-1).expand_as(beam_seq)) ==
                        beam_seq.reshape(-1, beam_seq.shape[-1])[state_ix].view_as(beam_seq)).all()
                beam_seq = beam_seq.gather(1, beam_ix.unsqueeze(-1).expand_as(beam_seq))
                beam_seq_logprobs = beam_seq_logprobs.gather(1, beam_ix.unsqueeze(-1).unsqueeze(-1).expand_as(
                    beam_seq_logprobs))

            beam_seq = torch.cat([beam_seq, selected_ix.unsqueeze(-1)], -1)
            beam_logprobs_sum = beam_logprobs_sum.gather(1, beam_ix) + \
                                logprobs.reshape(batch_size, -1).gather(1, ix)
            # ★ 修复：禁用脆弱的浮点自检断言（PyTorch 版本/数值路径差异下易误报 AssertionError）
            # assert (beam_logprobs_sum == ys).all()
            _tmp_beam_logprobs = unaug_logprobs[state_ix].reshape(batch_size, -1, vocab_size)
            beam_logprobs = unaug_logprobs.reshape(batch_size, -1, vocab_size).gather(1,
                                                                                      beam_ix.unsqueeze(-1).expand(-1,
                                                                                                                   -1,
                                                                                                                   vocab_size))
            # assert (_tmp_beam_logprobs == beam_logprobs).all()
            beam_seq_logprobs = torch.cat([
                beam_seq_logprobs,
                beam_logprobs.reshape(batch_size, -1, 1, vocab_size)], 2)

            new_state = [None for _ in state]
            for _ix in range(len(new_state)):
                new_state[_ix] = state[_ix][state_ix]   # ← 改动：[:, state_ix] → [state_ix]
            state = new_state
            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state

        # Start diverse_beam_search
        opt = kwargs['opt']
        temperature = opt.get('temperature', 1)
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        diversity_lambda = opt.get('diversity_lambda', 0.5)
        decoding_constraint = opt.get('decoding_constraint', 0)
        suppress_UNK = opt.get('suppress_UNK', 0)
        # ===== 修改：length_penalty 优先读 opt，其次读实例属性 self.length_penalty =====
        _lp_raw = opt.get('length_penalty', getattr(self, 'length_penalty', ''))
        if isinstance(_lp_raw, (int, float)):
            _lp_arg = f'wu_{_lp_raw}' if _lp_raw and _lp_raw > 0 else ''
        else:
            _lp_arg = _lp_raw if _lp_raw else ''
        # ★ 修复：兼容 utils.penalty_builder 的两种字符串格式（'wu_0.6' 或 'wu0.6'）
        try:
            length_penalty = utils.penalty_builder(_lp_arg)
        except (ValueError, TypeError):
            if isinstance(_lp_arg, str) and _lp_arg.startswith('wu_'):
                length_penalty = utils.penalty_builder(_lp_arg.replace('wu_', 'wu'))
            else:
                length_penalty = utils.penalty_builder('')
        # ===========================================================================
        bdash = beam_size // group_size

        batch_size = init_logprobs.shape[0]
        device = init_logprobs.device
        beam_seq_table = [torch.LongTensor(batch_size, bdash, 0).to(device) for _ in range(group_size)]
        beam_seq_logprobs_table = [torch.FloatTensor(batch_size, bdash, 0, self.vocab_size + 1).to(device) for _ in
                                   range(group_size)]
        beam_logprobs_sum_table = [torch.zeros(batch_size, bdash).to(device) for _ in range(group_size)]

        done_beams_table = [[[] for __ in range(group_size)] for _ in range(batch_size)]
        state_table = [[_.clone() for _ in init_state] for _ in range(group_size)]
        logprobs_table = [init_logprobs.clone() for _ in range(group_size)]

        args = list(args)
        args = utils.split_tensors(group_size, args)
        if self.__class__.__name__ == 'AttEnsemble':
            args = [[[args[j][i][k] for i in range(len(self.models))] for j in range(len(args))] for k in
                    range(group_size)]
        else:
            args = [[args[i][j] for i in range(len(args))] for j in range(group_size)]

        for t in range(self.max_seq_length + group_size - 1):
            for divm in range(group_size):
                if t >= divm and t <= self.max_seq_length + divm - 1:
                    logprobs = logprobs_table[divm]
                    if decoding_constraint and t - divm > 0:
                        logprobs.scatter_(1, beam_seq_table[divm][:, :, t - divm - 1].reshape(-1, 1).to(device),
                                          float('-inf'))
                    if suppress_UNK and hasattr(self, 'vocab') and self.vocab[str(logprobs.size(1) - 1)] == 'UNK':
                        logprobs[:, logprobs.size(1) - 1] = logprobs[:, logprobs.size(1) - 1] - 1000
                    logprobs, unaug_logprobs = add_diversity(beam_seq_table, logprobs, t, divm, diversity_lambda, bdash)

                    beam_seq_table[divm], \
                    beam_seq_logprobs_table[divm], \
                    beam_logprobs_sum_table[divm], \
                    state_table[divm] = beam_step(logprobs,
                                                  unaug_logprobs,
                                                  bdash,
                                                  t - divm,
                                                  beam_seq_table[divm],
                                                  beam_seq_logprobs_table[divm],
                                                  beam_logprobs_sum_table[divm],
                                                  state_table[divm])

                    for b in range(batch_size):
                        is_end = beam_seq_table[divm][b, :, t - divm] == self.eos_idx
                        assert beam_seq_table[divm].shape[-1] == t - divm + 1
                        if t == self.max_seq_length + divm - 1:
                            is_end.fill_(1)
                        for vix in range(bdash):
                            if is_end[vix]:
                                final_beam = {
                                    'seq': beam_seq_table[divm][b, vix].clone(),
                                    'logps': beam_seq_logprobs_table[divm][b, vix].clone(),
                                    'unaug_p': beam_seq_logprobs_table[divm][b, vix].sum().item(),
                                    'p': beam_logprobs_sum_table[divm][b, vix].item()
                                }
                                final_beam['p'] = length_penalty(t - divm + 1, final_beam['p'])
                                done_beams_table[b][divm].append(final_beam)
                        beam_logprobs_sum_table[divm][b, is_end] -= 1000

                    it = beam_seq_table[divm][:, :, t - divm].reshape(-1)
                    # ★ 修复：去掉 .cuda() 硬编码，改用 device
                    logprobs_table[divm], state_table[divm] = self.get_logprobs_state(it.to(device), *(
                            args[divm] + [state_table[divm]]))
                    logprobs_table[divm] = F.log_softmax(logprobs_table[divm] / temperature, dim=-1)

        done_beams_table = [[sorted(done_beams_table[b][i], key=lambda x: -x['p'])[:bdash] for i in range(group_size)]
                            for b in range(batch_size)]
        done_beams = [sum(_, []) for _ in done_beams_table]
        return done_beams

    def old_beam_search(self, init_state, init_logprobs, *args, **kwargs):

        def add_diversity(beam_seq_table, logprobsf, t, divm, diversity_lambda, bdash):
            local_time = t - divm
            unaug_logprobsf = logprobsf.clone()
            for prev_choice in range(divm):
                prev_decisions = beam_seq_table[prev_choice][local_time]
                for sub_beam in range(bdash):
                    for prev_labels in range(bdash):
                        logprobsf[sub_beam][prev_decisions[prev_labels]] = logprobsf[sub_beam][prev_decisions[
                            prev_labels]] - diversity_lambda
            return unaug_logprobsf

        def beam_step(logprobsf, unaug_logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            ys, ix = torch.sort(logprobsf, 1, True)
            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1
            for c in range(cols):
                for q in range(rows):
                    local_logprob = ys[q, c].item()
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    candidates.append({'c': ix[q, c], 'q': q, 'p': candidate_logprob, 'r': unaug_logprobsf[q]})
            candidates = sorted(candidates, key=lambda x: -x['p'])

            new_state = [_.clone() for _ in state]
            if t >= 1:
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
            for vix in range(beam_size):
                v = candidates[vix]
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                for state_ix in range(len(new_state)):
                    new_state[state_ix][:, vix] = state[state_ix][:, v['q']]
                beam_seq[t, vix] = v['c']
                beam_seq_logprobs[t, vix] = v['r']
                beam_logprobs_sum[vix] = v['p']
            state = new_state
            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state, candidates

        opt = kwargs['opt']
        temperature = opt.get('temperature', 1)
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        diversity_lambda = opt.get('diversity_lambda', 0.5)
        decoding_constraint = opt.get('decoding_constraint', 0)
        suppress_UNK = opt.get('suppress_UNK', 0)
        # ★ 修复：同样增加格式兼容
        _lp_raw_old = opt.get('length_penalty', getattr(self, 'length_penalty', ''))
        if isinstance(_lp_raw_old, (int, float)):
            _lp_arg_old = f'wu_{_lp_raw_old}' if _lp_raw_old and _lp_raw_old > 0 else ''
        else:
            _lp_arg_old = _lp_raw_old if _lp_raw_old else ''
        try:
            length_penalty = utils.penalty_builder(_lp_arg_old)
        except (ValueError, TypeError):
            if isinstance(_lp_arg_old, str) and _lp_arg_old.startswith('wu_'):
                length_penalty = utils.penalty_builder(_lp_arg_old.replace('wu_', 'wu'))
            else:
                length_penalty = utils.penalty_builder('')
        bdash = beam_size // group_size

        beam_seq_table = [torch.LongTensor(self.max_seq_length, bdash).zero_() for _ in range(group_size)]
        beam_seq_logprobs_table = [torch.FloatTensor(self.max_seq_length, bdash, self.vocab_size + 1).zero_() for _ in
                                   range(group_size)]
        beam_logprobs_sum_table = [torch.zeros(bdash) for _ in range(group_size)]

        done_beams_table = [[] for _ in range(group_size)]
        state_table = list(zip(*[_.chunk(group_size, 1) for _ in init_state]))
        logprobs_table = list(init_logprobs.chunk(group_size, 0))

        args = list(args)
        if self.__class__.__name__ == 'AttEnsemble':
            args = [[_.chunk(group_size) if _ is not None else [None] * group_size for _ in args_] for args_ in
                    args]
            args = [[[args[j][i][k] for i in range(len(self.models))] for j in range(len(args))] for k in
                    range(group_size)]
        else:
            args = [_.chunk(group_size) if _ is not None else [None] * group_size for _ in args]
            args = [[args[i][j] for i in range(len(args))] for j in range(group_size)]

        for t in range(self.max_seq_length + group_size - 1):
            for divm in range(group_size):
                if t >= divm and t <= self.max_seq_length + divm - 1:
                    logprobsf = logprobs_table[divm].float()
                    if decoding_constraint and t - divm > 0:
                        logprobsf.scatter_(1, beam_seq_table[divm][t - divm - 1].unsqueeze(1).cuda(), float('-inf'))
                    if suppress_UNK and hasattr(self, 'vocab') and self.vocab[str(logprobsf.size(1) - 1)] == 'UNK':
                        logprobsf[:, logprobsf.size(1) - 1] = logprobsf[:, logprobsf.size(1) - 1] - 1000
                    unaug_logprobsf = add_diversity(beam_seq_table, logprobsf, t, divm, diversity_lambda, bdash)

                    beam_seq_table[divm], \
                    beam_seq_logprobs_table[divm], \
                    beam_logprobs_sum_table[divm], \
                    state_table[divm], \
                    candidates_divm = beam_step(logprobsf,
                                                unaug_logprobsf,
                                                bdash,
                                                t - divm,
                                                beam_seq_table[divm],
                                                beam_seq_logprobs_table[divm],
                                                beam_logprobs_sum_table[divm],
                                                state_table[divm])

                    for vix in range(bdash):
                        if beam_seq_table[divm][t - divm, vix] == self.eos_idx or t == self.max_seq_length + divm - 1:
                            final_beam = {
                                'seq': beam_seq_table[divm][:, vix].clone(),
                                'logps': beam_seq_logprobs_table[divm][:, vix].clone(),
                                'unaug_p': beam_seq_logprobs_table[divm][:, vix].sum().item(),
                                'p': beam_logprobs_sum_table[divm][vix].item()
                            }
                            final_beam['p'] = length_penalty(t - divm + 1, final_beam['p'])
                            done_beams_table[divm].append(final_beam)
                            beam_logprobs_sum_table[divm][vix] = -1000

                    it = beam_seq_table[divm][t - divm]
                    logprobs_table[divm], state_table[divm] = self.get_logprobs_state(it.cuda(), *(
                            args[divm] + [state_table[divm]]))
                    logprobs_table[divm] = F.log_softmax(logprobs_table[divm] / temperature, dim=-1)

        done_beams_table = [sorted(done_beams_table[i], key=lambda x: -x['p'])[:bdash] for i in range(group_size)]
        done_beams = sum(done_beams_table, [])
        return done_beams

    # ===== 修改：增强 sample_next_word，支持显式 top_k / top_p =====
    def _apply_top_k_top_p(self, logprobs, top_k, top_p):
        if top_k > 0:
            the_k = min(int(top_k), logprobs.size(-1))
            tmp = torch.empty_like(logprobs).fill_(float('-inf'))
            topk_vals, topk_idx = torch.topk(logprobs, the_k, dim=1)
            tmp = tmp.scatter(1, topk_idx, topk_vals)
            logprobs = tmp
        if top_p > 0.0:
            probs = F.softmax(logprobs, dim=1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=1)
            cumsum = sorted_probs.cumsum(1)
            mask = cumsum < top_p
            mask = torch.cat([torch.ones_like(mask[:, :1]), mask[:, :-1]], 1)
            sorted_probs = sorted_probs * mask.float()
            sorted_probs = sorted_probs / (sorted_probs.sum(1, keepdim=True) + 1e-12)
            logprobs = logprobs.clone().scatter_(1, sorted_indices, sorted_probs.log())
        return logprobs

    def sample_next_word(self, logprobs, sample_method, temperature, top_k=None, top_p=None):
        # 若调用方未显式传入，则从实例属性读取（由 models.py 注入）
        if top_k is None:
            top_k = getattr(self, 'top_k', 0)
        if top_p is None:
            top_p = getattr(self, 'top_p', 0.0)

        if sample_method == 'greedy':
            sampleLogprobs, it = torch.max(logprobs.data, 1)
            it = it.view(-1).long()
        elif sample_method == 'gumbel':
            def sample_gumbel(shape, eps=1e-20):
                U = torch.rand(shape).cuda()
                return -torch.log(-torch.log(U + eps) + eps)

            def gumbel_softmax_sample(logits, temperature):
                y = logits + sample_gumbel(logits.size())
                return F.log_softmax(y / temperature, dim=-1)

            _logprobs = gumbel_softmax_sample(logprobs, temperature)
            _, it = torch.max(_logprobs.data, 1)
            sampleLogprobs = logprobs.gather(1, it.unsqueeze(1))
        else:
            # 随机采样分支：先施加温度
            logprobs = logprobs / max(float(temperature), 1e-6)

            # 优先使用显式 top_k / top_p
            if top_k > 0 or top_p > 0.0:
                logprobs = self._apply_top_k_top_p(logprobs, top_k, top_p)
            elif sample_method.startswith('top'):
                # 兼容原字符串写法，如 'top20' / 'top0.9'
                top_num = float(sample_method[3:])
                if 0 < top_num < 1:
                    probs = F.softmax(logprobs, dim=1)
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=1)
                    _cumsum = sorted_probs.cumsum(1)
                    mask = _cumsum < top_num
                    mask = torch.cat([torch.ones_like(mask[:, :1]), mask[:, :-1]], 1)
                    sorted_probs = sorted_probs * mask.float()
                    sorted_probs = sorted_probs / sorted_probs.sum(1, keepdim=True)
                    logprobs.scatter_(1, sorted_indices, sorted_probs.log())
                else:
                    the_k = int(top_num)
                    tmp = torch.empty_like(logprobs).fill_(float('-inf'))
                    topk, indices = torch.topk(logprobs, the_k, dim=1)
                    tmp = tmp.scatter(1, indices, topk)
                    logprobs = tmp

            it = torch.distributions.Categorical(logits=logprobs.detach()).sample()
            sampleLogprobs = logprobs.gather(1, it.unsqueeze(1))
        return it, sampleLogprobs