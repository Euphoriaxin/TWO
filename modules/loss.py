# modules/loss.py
import math

import torch
import torch.nn as nn


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output


def compute_loss(output, reports_ids, reports_masks, mal_loss=None, mal_weight=1.0):
    if not math.isfinite(mal_weight) or mal_weight < 0:
        raise ValueError("mal_weight must be finite and nonnegative")
    criterion = LanguageModelCriterion()
    loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
    if mal_loss is not None and mal_weight != 0:
        loss = loss + mal_weight * mal_loss
    return loss
