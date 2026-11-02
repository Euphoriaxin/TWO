from pathlib import Path
import shutil

import pytest

from pycocoevalcap.meteor.meteor import Meteor
from modules.metrics import compute_scores
from scripts.setup_meteor import install

DATA = Path(__file__).resolve().parents[1] / 'pycocoevalcap/meteor/data/paraphrase-en.gz'
requires_runtime = pytest.mark.skipif(
    not DATA.is_file() or shutil.which('java') is None,
    reason='Run python scripts/setup_meteor.py and install Java for integration tests')


def test_missing_data_has_setup_instructions(monkeypatch):
    original = Path.is_file
    monkeypatch.setattr(Path, 'is_file', lambda path: False if path.name == 'paraphrase-en.gz' else original(path))
    with pytest.raises(FileNotFoundError, match='python scripts/setup_meteor.py'):
        Meteor()


def test_setup_rejects_modified_wheel(tmp_path):
    wheel = tmp_path / 'bad.whl'
    wheel.write_bytes(b'not the published artifact')
    with pytest.raises(RuntimeError, match='wheel checksum'):
        install(wheel, tmp_path / 'data.gz')
    assert not (tmp_path / 'data.gz').exists()


def test_setup_preserves_existing_unverified_data(tmp_path):
    destination = tmp_path / 'data.gz'
    destination.write_bytes(b'user data')
    with pytest.raises(RuntimeError, match='Checksum mismatch'):
        install(destination=destination)
    assert destination.read_bytes() == b'user data'


@requires_runtime
def test_java_error_is_reported_and_process_is_closed(tmp_path, monkeypatch):
    java = tmp_path / 'java'
    java.write_text('#!/bin/sh\necho "fixture: failed to load paraphrases" >&2\nexit 7\n')
    java.chmod(0o755)
    monkeypatch.setenv('PATH', str(tmp_path))
    with Meteor() as scorer:
        process = scorer.meteor_p
        with pytest.raises(RuntimeError, match='fixture: failed to load paraphrases'):
            scorer.compute_score({0:['normal lungs']}, {0:['normal lungs']})
    assert process.poll() == 7
    assert scorer.meteor_p is None


@requires_runtime
def test_actual_meteor_scores_and_repeated_calls():
    references = {0:['no pleural effusion .'], 1:['heart size is normal .']}
    with Meteor() as scorer:
        process = scorer.meteor_p
        for _ in range(2):
            score, individual = scorer.compute_score(references, references)
            assert score == pytest.approx(1.)
            assert individual == pytest.approx([1., 1.])
        score, _ = scorer.compute_score(references, {0:['banana'], 1:['spaceship']})
        assert score < 1.
    assert process.poll() is not None


@requires_runtime
def test_complete_metric_pipeline():
    references = {0:['no pleural effusion .'], 1:['heart size is normal .']}
    scores = compute_scores(references, references)
    assert set(scores) == {'BLEU_1', 'BLEU_2', 'BLEU_3', 'BLEU_4', 'METEOR', 'ROUGE_L', 'CIDEr'}
    assert scores['BLEU_4'] == pytest.approx(1.)
    assert scores['METEOR'] == pytest.approx(1.)
    assert scores['ROUGE_L'] == pytest.approx(1.)
