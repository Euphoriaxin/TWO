import copy
from types import SimpleNamespace

import pytest
import torch

from modules.base_cmn import BaseCMN
from modules.hd_computing import HDPrototypeMemory


def make_model():
    torch.manual_seed(17)
    args = SimpleNamespace(
        d_model=8, d_ff=16, num_layers=1, drop_prob_lm=0.,
        max_seq_length=5, d_vf=6, bos_idx=0, eos_idx=0, pad_idx=0,
        use_bn=0, num_heads=2, dropout=0., topk=2, hd_dim=32,
        num_patches=3, hdc_scale=.1, use_evmoe=True, use_mal=False,
        num_experts=2, use_hd_memory=True, update_hd_memory=True,
        hd_memory_weight=.2, hd_memory_temperature=.1,
        sample_method='greedy', beam_size=1, sample_n=1,
        group_size=1, block_trigrams=0)
    return BaseCMN(args, SimpleNamespace(idx2token={1:'a', 2:'b', 3:'c'}))


def features():
    return torch.randn(2, 12), [torch.randn(2, 3, 6), torch.randn(2, 3, 6)], [torch.ones(2), torch.ones(2)]


def test_duplicate_updates_padding_and_empty_memory():
    mem = HDPrototypeMemory(4, 5, 4)
    positions = torch.tensor([[0, 1, 2], [0, 1, 2]])
    tokens = torch.tensor([[1, 0, 0], [1, 2, 0]])
    queries = torch.ones(2, 3, 4)
    valid = torch.tensor([[True, True, False], [True, True, True]])
    assert not mem.compute_logits(positions, queries).any()
    mem.update(positions, tokens, queries, valid)
    assert mem.counts[0, 1] == 2
    assert torch.equal(mem.prototype[0, 1], torch.full((4,), 2.))
    assert mem.counts[1, 0] == 1  # EOS is a real target
    assert mem.counts[2, 0] == 1  # padding in first sample excluded
    scores = mem.compute_logits(torch.tensor([0]), torch.ones(1, 4))
    assert scores[0, 1] == 1 and scores[0, 2] == 0


def test_projection_frozen_and_initialization_preserved_after_optimizer_step():
    model = make_model().train()
    assert not any(p.requires_grad for p in model.hd_proj.parameters())
    assert set(model.hdc.patch_pos_vectors.unique().tolist()) <= {-1., 1.}
    assert model.hd_proj.weight.std() < .03
    before = model.hd_proj.weight.detach().clone()
    fc, att, masks = features()
    logits, _ = model(fc, att, torch.tensor([[0, 1, 2, 0], [0, 2, 0, 0]]), att_masks=masks)
    loss = -logits[:, 0, 1].mean()
    loss.backward()
    assert model.hd_proj.weight.grad is None
    assert model.logit.weight.grad is not None
    torch.optim.AdamW(model.parameters(), lr=.01).step()
    torch.testing.assert_close(before, model.hd_proj.weight)


def test_update_targets_next_word_and_reads_before_write():
    model = make_model().train()
    fc, att, masks = features()
    seq = torch.tensor([[0, 1, 2, 0, 0], [0, 2, 0, 0, 0]])
    baseline = copy.deepcopy(model)
    baseline.use_hd_memory = False
    torch.manual_seed(42)  # match attention dropout in both training forwards
    expected, _ = baseline(fc, att, seq, att_masks=masks)
    torch.manual_seed(42)
    actual, _ = model(fc, att, seq, att_masks=masks)
    torch.testing.assert_close(expected, actual)  # empty memory did not see answers
    assert model.hd_memory.counts.sum() == 5
    assert model.hd_memory.counts[0, 1] == 1
    assert model.hd_memory.counts[0, 2] == 1
    assert model.hd_memory.counts[1, 2] == 1
    assert model.hd_memory.counts[1, 0] == 1
    assert model.hd_memory.counts[2, 0] == 1
    assert not model.hd_memory.counts[3:].any()


def test_memory_changes_prediction_and_matches_teacher_forcing():
    model = make_model().eval()
    fc, att, masks = features()
    seq = torch.tensor([[0, 1, 2, 0], [0, 2, 1, 0]])
    image_hd, att_feats, encoded, att_masks = model._prepare_feature(fc, att, masks)
    queries = model._memory_queries(image_hd, seq[:, :-1])
    positions = torch.arange(3).expand(2, -1)
    model.hd_memory.update(positions, seq[:, 1:], queries)
    counts = model.hd_memory.counts.clone()
    teacher, _ = model(fc, att, seq, att_masks=masks)
    state = []
    for t in range(3):
        logprobs, state = model.get_logprobs_state(
            seq[:, t], image_hd, att_feats, encoded, att_masks, state)
        torch.testing.assert_close(teacher[:, t], logprobs, atol=1e-5, rtol=1e-5)
    model.hd_memory_weight = 0
    without, _ = model(fc, att, seq, att_masks=masks)
    assert not torch.allclose(without, teacher)
    torch.testing.assert_close(counts, model.hd_memory.counts)


@pytest.mark.parametrize('beam', [1, 3])
def test_sampling_uses_memory_and_does_not_mutate_it(beam):
    model = make_model().eval()
    fc, att, masks = features()
    image_hd, *_ = model._prepare_feature(fc, att, masks)
    query = model._memory_queries(image_hd, torch.zeros(2, 1, dtype=torch.long))
    model.hd_memory.update(torch.zeros(2, 1, dtype=torch.long), torch.ones(2, 1, dtype=torch.long), query)
    before = model.hd_memory.prototype.clone()
    model.hd_memory_weight = 100.
    with torch.no_grad():
        seq, probs = model(fc, att, att_masks=masks, mode='sample',
                          update_opts={'sample_method': 'beam_search' if beam > 1 else 'greedy',
                                       'beam_size': beam, 'length_penalty': 0.})
    assert seq.shape == (2, model.max_seq_length)
    assert (seq[:, 0] == 1).all()
    assert torch.isfinite(probs).all()
    torch.testing.assert_close(before, model.hd_memory.prototype)


def test_checkpoint_roundtrip_and_legacy_reset(tmp_path):
    model = make_model()
    model.hd_memory.prototype[0, 1].fill_(3.)
    model.hd_memory.counts[0, 1] = 3
    path = tmp_path / 'model.pth'
    torch.save(model.state_dict(), path)
    restored = make_model()
    restored.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(restored.hd_memory.prototype, model.hd_memory.prototype)
    torch.testing.assert_close(restored.hd_memory.counts, model.hd_memory.counts)
    legacy = model.state_dict()
    del legacy['hd_memory.counts']
    with pytest.warns(UserWarning, match='Legacy HDC'):
        restored.load_state_dict(legacy, strict=True)
    assert not restored.hd_memory.prototype.any()
    assert not restored.hd_memory.counts.any()


def test_training_and_test_cli_share_memory_and_model_options(monkeypatch):
    from main_train import parse_agrs as train_args
    from main_test import parse_agrs as test_args
    monkeypatch.setattr('sys.argv', ['main_train.py'])
    training = train_args()
    monkeypatch.setattr('sys.argv', ['main_test.py', '--load', 'unused.pth'])
    testing = test_args()
    for key in ('hd_dim', 'hdc_scale', 'hd_memory_weight', 'hd_memory_temperature',
                'bos_idx', 'eos_idx', 'pad_idx', 'num_experts', 'd_model'):
        assert getattr(training, key) == getattr(testing, key)


def test_real_resnet_iu_forward_backward_and_sample(monkeypatch):
    from main_train import parse_agrs
    from models.models import BaseCMNModel
    monkeypatch.setattr('sys.argv', ['main_train.py'])
    args = parse_agrs()
    args.visual_extractor_pretrained = False  # no network/data needed
    args.d_model, args.d_ff, args.num_heads, args.num_layers = 8, 16, 2, 1
    args.hd_dim, args.max_seq_length = 32, 5
    args.use_mal = False
    args.sample_method, args.beam_size = 'greedy', 1
    model = BaseCMNModel(args, SimpleNamespace(idx2token={1:'a', 2:'b', 3:'c'}))
    images = torch.randn(1, 2, 3, 224, 224)
    model.train()
    output, _ = model(images, torch.tensor([[0, 1, 2, 0]]))
    (-output[..., 1].mean()).backward()
    assert model.visual_extractor.model[0].weight.grad is not None
    assert model.encoder_decoder.hd_proj.weight.grad is None
    assert model.encoder_decoder.hd_memory.counts.sum() == 3
    model.eval()
    with torch.no_grad():
        seq, probs = model(images, mode='sample')
    assert seq.shape == (1, 5)
    assert torch.isfinite(probs).all()
