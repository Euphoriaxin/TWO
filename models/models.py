# models/models.py
import numpy as np
import torch
import torch.nn as nn

from modules.base_cmn import BaseCMN
from modules.visual_extractor import VisualExtractor


class BaseCMNModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(BaseCMNModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)

        if not hasattr(self.visual_extractor, 'num_patches_single'):
            raise AttributeError(
                "VisualExtractor 缺少 num_patches_single 属性，请使用多骨干版 visual_extractor.py"
            )

        # ★ 根据是否使用 EV-MoE 动态设置 num_patches（保证 HDC 位置编码长度正确）
        if getattr(args, 'use_evmoe', True):
            args.num_patches = self.visual_extractor.num_patches_single          # 融合后=单图长度
        else:
            args.num_patches = self.visual_extractor.num_patches_single * 2      # 基线 concat 双图长度

        self.encoder_decoder = BaseCMN(args, tokenizer)

        ed = self.encoder_decoder
        ed.beam_size       = args.beam_size
        ed.length_penalty  = args.length_penalty
        ed.temperature     = args.temperature
        ed.top_k           = args.top_k_sampling
        ed.top_p           = args.top_p_sampling

        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def binarize_hd_memory(self):
        self.encoder_decoder.hd_memory.binarize()

    def _extract_views(self, images, dataset_name):
        B = images.size(0)
        device = images.device
        if dataset_name == 'iu_xray':
            imgs = [images[:, 0], images[:, 1]]
            masks = [torch.ones(B, device=device), torch.ones(B, device=device)]
        else:
            if images.dim() == 5:
                V = images.size(1)
                imgs = [images[:, v] for v in range(V)]
                masks = [torch.ones(B, device=device) for _ in range(V)]
            else:
                imgs = [images, images]
                masks = [torch.ones(B, device=device), torch.zeros(B, device=device)]
        return imgs, masks

    def forward_iu_xray(self, images, targets=None, mode='train', update_opts=None):
        if update_opts is None:
            update_opts = {}
        imgs, view_masks = self._extract_views(images, 'iu_xray')
        att_list, fc_list = [], []
        for img in imgs:
            att, fc = self.visual_extractor(img)
            att_list.append(att)
            fc_list.append(fc)
        fc_feats = torch.cat(fc_list, dim=1)
        if mode == 'train':
            return self.encoder_decoder(fc_feats, att_list, targets, att_masks=view_masks, mode='forward')
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_list, att_masks=view_masks,
                                                        mode='sample', update_opts=update_opts)
            return output, output_probs
        else:
            raise ValueError

    def forward_mimic_cxr(self, images, targets=None, mode='train', update_opts=None):
        if update_opts is None:
            update_opts = {}
        imgs, view_masks = self._extract_views(images, 'mimic_cxr')
        att_list, fc_list = [], []
        for img in imgs:
            att, fc = self.visual_extractor(img)
            att_list.append(att)
            fc_list.append(fc)
        fc_feats = torch.cat(fc_list, dim=1)
        if mode == 'train':
            return self.encoder_decoder(fc_feats, att_list, targets, att_masks=view_masks, mode='forward')
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_list, att_masks=view_masks,
                                                        mode='sample', update_opts=update_opts)
            return output, output_probs
        else:
            raise ValueError