import logging
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from modules.ev_moe import MultiViewAlignmentLoss
from modules.loss import compute_loss
from modules.trainer import Trainer


def representations(batch=3):
    generator = torch.Generator().manual_seed(38)
    return [torch.randn(batch, 5, generator=generator, dtype=torch.double,
                        requires_grad=True) for _ in range(3)]


def reference_contrastive(views, tau):
    # Independent scalar reference: all same-sample, other-view positives
    # receive equal target mass; the anchor itself is never a candidate.
    normalized = [F.normalize(view, dim=-1) for view in views]
    losses = []
    for i, view in enumerate(normalized):
        for b, anchor in enumerate(view):
            candidates, positives = [], []
            for j, other in enumerate(normalized):
                for c, embedding in enumerate(other):
                    if i == j and b == c:
                        continue
                    candidates.append(anchor.dot(embedding) / tau)
                    if b == c:
                        positives.append(len(candidates) - 1)
            scores = torch.stack(candidates)
            losses.append(torch.logsumexp(scores, dim=0) - scores[positives].mean())
    return torch.stack(losses).mean()


def test_all_positives_match_independent_value_and_gradients():
    views = representations()
    alpha, tau = .1, .2
    loss = MultiViewAlignmentLoss(alpha, tau)(views[0], views[1:])
    expected = ((1 - alpha) * reference_contrastive(views, tau)
                + alpha * sum(F.mse_loss(views[0], v) for v in views[1:]) / 2)
    torch.testing.assert_close(loss, expected)
    actual_grads = torch.autograd.grad(loss, views, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, views)
    for actual, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, expected)
        assert torch.isfinite(actual).all() and actual.abs().sum() > 0


def test_swapping_positive_views_preserves_loss_and_gradients():
    fused, front, side = representations()
    mal = MultiViewAlignmentLoss(alpha=.1, tau=.1)
    first = mal(fused, [front, side])
    swapped = mal(fused, [side, front])
    torch.testing.assert_close(first, swapped)
    first_grads = torch.autograd.grad(first, [fused, front, side], retain_graph=True)
    swapped_grads = torch.autograd.grad(swapped, [fused, front, side])
    for first, swapped in zip(first_grads, swapped_grads):
        torch.testing.assert_close(first, swapped)


def test_single_sample_has_two_positive_floor_and_no_self_match():
    same = torch.tensor([[1., 0.]], requires_grad=True)
    loss = MultiViewAlignmentLoss(alpha=0)(same, [same, same])
    torch.testing.assert_close(loss, torch.tensor(math.log(2)))
    loss.backward()
    assert torch.isfinite(same.grad).all()


def test_no_views_returns_connected_zero():
    fused = torch.randn(2, 5, requires_grad=True)
    loss = MultiViewAlignmentLoss()(fused, [])
    loss.backward()
    assert loss.item() == 0
    torch.testing.assert_close(fused.grad, torch.zeros_like(fused))


@pytest.mark.parametrize('weight', [0., .1, 1.])
def test_total_loss_weight_scales_only_auxiliary_gradients(weight):
    logits = torch.randn(2, 3, 4, requires_grad=True)
    output = logits.log_softmax(-1)
    ids = torch.tensor([[0, 1, 2, 0], [0, 2, 0, 0]])
    masks = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    auxiliary = torch.tensor(3., requires_grad=True)
    baseline = compute_loss(output, ids, masks)
    loss = compute_loss(output, ids, masks, auxiliary, mal_weight=weight)
    torch.testing.assert_close(loss, baseline + weight * auxiliary)
    actual = torch.autograd.grad(loss, [logits, auxiliary], allow_unused=True, retain_graph=True)
    reference = torch.autograd.grad(baseline, logits)[0]
    torch.testing.assert_close(actual[0], reference)
    if weight:
        torch.testing.assert_close(actual[1], torch.tensor(weight))
    else:
        assert actual[1] is None


@pytest.mark.parametrize('weight', [0., .1, 1.])
def test_trainer_applies_weight_and_logs_components(weight):
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.words = torch.nn.Parameter(torch.zeros(3))
            self.alignment = torch.nn.Parameter(torch.tensor(2.))

        def forward(self, images, ids, mode):
            return self.words.log_softmax(-1).expand(1, 2, 3), self.alignment.square()

    trainer = object.__new__(Trainer)
    trainer.model = ToyModel()
    trainer.args = SimpleNamespace(mal_weight=weight, log_period=50)
    trainer.epochs = 1
    trainer.logger = logging.getLogger('mal-test')
    trainer.device = torch.device('cpu')
    trainer.criterion = compute_loss
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=.01)
    trainer.lr_scheduler = None
    trainer.train_dataloader = [(('id',), torch.zeros(1), torch.tensor([[0, 1, 0]]), torch.ones(1, 3))]
    trainer.val_dataloader = trainer.test_dataloader = []
    trainer._evaluate_epoch = lambda *args: {}
    log = trainer._train_epoch(1)
    assert trainer.model.alignment.item() == pytest.approx(2 - .04 * weight)
    assert log['train_report_loss'] == pytest.approx(math.log(3))
    assert log['train_mal_loss'] == pytest.approx(4)
    assert log['train_weighted_mal_loss'] == pytest.approx(4 * weight)
    assert log['train_loss'] == pytest.approx(log['train_report_loss'] + log['train_weighted_mal_loss'])


def test_cli_weight_and_invalid_value(monkeypatch):
    from main_train import parse_agrs
    from main_test import parse_agrs as test_args
    monkeypatch.setattr('sys.argv', ['main_train.py', '--mal_weight', '.1'])
    assert parse_agrs().mal_weight == .1
    monkeypatch.setattr('sys.argv', ['main_test.py', '--load', 'unused', '--mal_weight', '.3'])
    assert test_args().mal_weight == .3
    monkeypatch.setattr('sys.argv', ['main_train.py', '--mal_weight', '-1'])
    with pytest.raises(SystemExit):
        parse_agrs()


@pytest.mark.parametrize('alpha,tau', [(-.1,.1), (1.1,.1), (.1,0), (.1,-1), (.1,float('nan'))])
def test_invalid_alignment_parameters(alpha, tau):
    with pytest.raises(ValueError):
        MultiViewAlignmentLoss(alpha, tau)
