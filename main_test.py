"""Evaluate a checkpoint with the same model/tokenizer options as training."""
import argparse

import numpy as np
import torch

from main_train import parse_agrs as parse_train_args
from models.models import BaseCMNModel
from modules.dataloaders import R2DataLoader
from modules.loss import compute_loss
from modules.metrics import compute_scores
from modules.tokenizers import Tokenizer
from modules.tester import Tester


def parse_agrs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', required=True, help='Checkpoint to evaluate (strict state loading).')
    parser.add_argument('--eval_split', choices=['val', 'test'], default='test')
    return parse_train_args(parser)


def main():
    args = parse_agrs()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tokenizer = Tokenizer(args)
    dataloader = R2DataLoader(args, tokenizer, split=args.eval_split, shuffle=False)
    model = BaseCMNModel(args, tokenizer)
    tester = Tester(model, compute_loss, compute_scores, args, dataloader)
    tester.test()


if __name__ == '__main__':
    main()
