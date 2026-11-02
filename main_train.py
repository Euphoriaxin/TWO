# main_train.py
import argparse

import numpy as np
import torch

from models.models import BaseCMNModel
from modules.dataloaders import R2DataLoader
from modules.loss import compute_loss
from modules.metrics import compute_scores, validate_meteor
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.tokenizers import Tokenizer
from modules.trainer import Trainer


def parse_agrs(parser=None):
    parser = parser if parser is not None else argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/',
                        help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation.json',
                        help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'],
                        help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101',
                        choices=['resnet50', 'resnet101', 'densenet121', 'vit_base_patch16', 'biomedclip'],
                        help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')

    # for Cross-modal Memory (kept for compatibility, unused in Scheme B)
    parser.add_argument('--topk', type=int, default=32, help='the number of k.')
    parser.add_argument('--cmm_size', type=int, default=2048, help='the numebr of cmm size.')
    parser.add_argument('--cmm_dim', type=int, default=512, help='the dimension of cmm dimension.')

    # ===== for Hyperdimensional Computing (Scheme B) + EV-MoE =====
    parser.add_argument('--hd_dim', type=int, default=5000, help='HDC hyperdimension.')
    parser.add_argument('--num_patches', type=int, default=16, help='number of patches for HDC binding.')
    parser.add_argument('--hdc_scale', type=float, default=0.3, help='HDC response injection scale')
    parser.add_argument('--use_hd_memory', type=lambda x: str(x).lower() == 'true', default=True)
    parser.add_argument('--update_hd_memory', type=lambda x: str(x).lower() == 'true', default=True)
    parser.add_argument('--hd_memory_weight', type=float, default=0.2,
                        help='Weight of prototype similarity added to next-token logits; 0 disables retrieval.')
    parser.add_argument('--hd_memory_temperature', type=float, default=0.1)
    # --- EV-MoE 超参 ---
    parser.add_argument('--num_experts', type=int, default=4, help='EV-MoE number of experts (DMFM).')
    parser.add_argument('--mal_alpha', type=float, default=0.5, help='MAL balance: (1-a)*contrast + a*mse.')
    parser.add_argument('--mal_tau', type=float, default=0.1, help='MAL contrastive temperature.')
    # --- EV-MoE / MAL 总开关（消融实验用）---
    parser.add_argument('--use_evmoe', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='Enable EV-MoE (VFEM+DMFM). Set False to use original concat baseline.')
    parser.add_argument('--use_mal', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='Enable Multi-view Alignment Loss (MAL).')
    # =====================================================
    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=5, help='the beam size when beam searching (increased from 3).')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--length_penalty', type=float, default=0.0, help='the length penalty for beam search (e.g. 0.6).')
    parser.add_argument('--top_k_sampling', type=int, default=0,
                        help='top-k sampling for inference (0=off, e.g. 20). Used when sample_method=top.')
    parser.add_argument('--top_p_sampling', type=float, default=0.0,
                        help='nucleus (top-p) sampling threshold (0=off, e.g. 0.9). Used when sample_method=top.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray_resnet101/', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments.')
    parser.add_argument('--log_period', type=int, default=50, help='the logging interval (in batches).')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period (in epochs).')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=20, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='AdamW', help='the type of the optimizer (Adam / AdamW).')
    parser.add_argument('--lr_ve', type=float, default=1e-4, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=5e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--adam_betas', type=tuple, default=(0.9, 0.98), help='the adam betas.')
    parser.add_argument('--adam_eps', type=float, default=1e-9, help='the adam eps.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')
    parser.add_argument('--noamopt_warmup', type=int, default=5000, help='.')
    parser.add_argument('--noamopt_factor', type=int, default=1, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='CosineWithWarmup',
                        choices=['StepLR', 'CosineAnnealingLR', 'CosineWithWarmup'],
                        help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=10, help='the step size of StepLR.')
    parser.add_argument('--gamma', type=float, default=0.8, help='the gamma of StepLR.')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='number of epochs for linear warmup (CosineWithWarmup).')
    parser.add_argument('--cosine_t_max', type=int, default=0,
                        help='T_max for CosineAnnealingLR (0=auto use args.epochs).')
    parser.add_argument('--min_lr', type=float, default=1e-7,
                        help='minimum lr for cosine schedulers.')

    # Others
    parser.add_argument('--seed', type=int, default=7580, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')

    args = parser.parse_args()
    return args


def main():
    args = parse_agrs()
    validate_meteor()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    tokenizer = Tokenizer(args)

    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    model = BaseCMNModel(args, tokenizer)

    criterion = compute_loss
    metrics = compute_scores

    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer, steps_per_epoch=len(train_dataloader))

    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader)
    trainer.train()


if __name__ == '__main__':
    main()