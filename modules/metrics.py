from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor import Meteor
from pycocoevalcap.rouge import Rouge
# ====== 1. 导入 CIDEr 模块 ======
from pycocoevalcap.cider.cider import Cider


def validate_meteor():
    """Fail before training if Java or METEOR's English resources are unusable."""
    with Meteor() as scorer:
        scorer.compute_score({0: ['no pleural effusion .']}, {0: ['no pleural effusion .']})


def compute_scores(gts, res):
    """Compute every configured metric; failures are reported, never hidden."""
    with Meteor() as meteor:
        scorers = [
            (Bleu(4), ['BLEU_1', 'BLEU_2', 'BLEU_3', 'BLEU_4']),
            (meteor, 'METEOR'),
            (Rouge(), 'ROUGE_L'),
            (Cider(), 'CIDEr'),
        ]
        eval_res = {}
        for scorer, method in scorers:
            if isinstance(scorer, Bleu):
                score, _ = scorer.compute_score(gts, res, verbose=0)
            else:
                score, _ = scorer.compute_score(gts, res)
            if isinstance(method, list):
                eval_res.update(zip(method, score))
            else:
                eval_res[method] = score
        return eval_res


def compute_mlc(gt, pred, label_set):
    res_mlc = {}
    avg_aucroc = 0
    for i, label in enumerate(label_set):
        res_mlc['AUCROC_' + label] = roc_auc_score(gt[:, i], pred[:, i])
        avg_aucroc += res_mlc['AUCROC_' + label]
    res_mlc['AVG_AUCROC'] = avg_aucroc / len(label_set)

    res_mlc['F1_MACRO'] = f1_score(gt, pred, average="macro")
    res_mlc['F1_MICRO'] = f1_score(gt, pred, average="micro")
    res_mlc['RECALL_MACRO'] = recall_score(gt, pred, average="macro")
    res_mlc['RECALL_MICRO'] = recall_score(gt, pred, average="micro")
    res_mlc['PRECISION_MACRO'] = precision_score(gt, pred, average="macro")
    res_mlc['PRECISION_MICRO'] = precision_score(gt, pred, average="micro")

    return res_mlc


class MetricWrapper(object):
    def __init__(self, label_set):
        self.label_set = label_set

    def __call__(self, gts, res, gts_mlc, res_mlc):
        eval_res = compute_scores(gts, res)
        eval_res_mlc = compute_mlc(gts_mlc, res_mlc, self.label_set)

        eval_res.update(**eval_res_mlc)
        return eval_res
