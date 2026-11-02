import argparse
import sys
import os
import numpy as np
import torch
import logging

# 将项目根目录添加到系统路径，防止云服务器上跨目录导入报错
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.models import BaseCMNModel
from modules.dataloaders import R2DataLoader
from modules.loss import compute_loss
from modules.metrics import compute_scores
from modules.tokenizers import Tokenizer
from modules.tester import Tester


def parse_agrs():
    parser = argparse.ArgumentParser()

    # 安全的布尔值解析器，防止 argparse 将字符串 "False" 误判为 True
    str2bool = lambda x: (str(x).lower() == 'true')

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
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=str2bool, default=True,
                        help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')

    # 词汇表特殊标记索引
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=1, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=3, help='the index of <pad>.')
    parser.add_argument('--unk_idx', type=int, default=2, help='the index of <unk>.')
    parser.add_argument('--vocab_size', type=int, default=4730, help='default vocab size')

    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')

    # for Cross-modal Memory
    parser.add_argument('--topk', type=int, default=32, help='the number of k.')
    parser.add_argument('--cmm_size', type=int, default=2048, help='the numebr of cmm size.')
    parser.add_argument('--cmm_dim', type=int, default=512, help='the dimension of cmm dimension.')

    # HD 记忆模块核心参数
    parser.add_argument('--hd_dim', type=int, default=256, help='HD embedding dimension')
    parser.add_argument('--use_hd_memory', type=str2bool, default=True, help='whether to use HD memory')
    parser.add_argument('--hd_memory_weight', type=float, default=0.8, help='weight for HD memory fusion')
    parser.add_argument('--update_hd_memory', type=str2bool, default=True, help='whether to update HD memory')
    parser.add_argument('--hd_topk', type=int, default=16, help='topk for HD memory retrieval')

    # CRFT 图像对齐模块参数
    parser.add_argument('--use_crft', type=str2bool, default=True, help='whether to use CRFT image alignment')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search',
                        help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/',
                        help='the patch to save the results of experiments.')
    parser.add_argument('--log_period', type=int, default=1000, help='the logging interval (in batches).')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period (in epochs).')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'],
                        help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=7e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--adam_betas', type=str, default='(0.9, 0.98)', help='the weight decay.')
    parser.add_argument('--adam_eps', type=float, default=1e-9, help='the weight decay.')
    parser.add_argument('--amsgrad', type=str2bool, default=True, help='.')
    parser.add_argument('--noamopt_warmup', type=int, default=5000, help='.')
    parser.add_argument('--noamopt_factor', type=int, default=1, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')
    parser.add_argument('--load', type=str, help='whether to load the pre-trained model.')

    args = parser.parse_args()

    # 规范化优化器 betas 参数
    if hasattr(args, 'adam_betas') and isinstance(args.adam_betas, str):
        try:
            betas_str = args.adam_betas.strip('()').replace(' ', '')
            betas = tuple(map(float, betas_str.split(',')))
            args.adam_betas = betas
        except Exception:
            args.adam_betas = (0.9, 0.999)

    return args


def main():
    # 1. 解析命令行与默认参数
    args = parse_agrs()

    # ✅ 创建 logger 并添加到 args
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    args.logger = logger

    # 严格固定随机种子，确保测试集指标可复现
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # 2. 初始化分词器
    tokenizer = Tokenizer(args)
    args.vocab_size = tokenizer.get_vocab_size()

    # 3. ==================== 【解绑补丁】 ====================
    # 彻底移除会发生误判的硬编码，完全信任命令行传入的 use_crft 和 use_hd_memory
    print(f"✨ [Test] 遵循 sh 脚本参数配置：use_crft={args.use_crft} | use_hd_memory={args.use_hd_memory}")
    # ========================================================

    # 4. 初始化测试集 R2DataLoader
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    # 5. 构建正式的模型网络
    model = BaseCMNModel(args, tokenizer)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')  # ✅ 确保模型在正确的设备上

    # ✅✅✅ 关键修复：加载预训练模型权重（过滤不匹配的键） ✅✅✅
    if hasattr(args, 'load') and args.load:
        checkpoint_path = args.load
        if os.path.exists(checkpoint_path):
            print(f"✅ 正在加载预训练模型: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # 处理不同的保存格式
            if 'model_state' in checkpoint:
                state_dict = checkpoint['model_state']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # ✅ 过滤掉模型中不存在的键
            model_state_dict = model.state_dict()
            filtered_state_dict = {}

            for k, v in state_dict.items():
                if k in model_state_dict:
                    if model_state_dict[k].shape == v.shape:
                        filtered_state_dict[k] = v
                    else:
                        print(f"⚠️ 形状不匹配，跳过: {k}, 模型形状: {model_state_dict[k].shape}, 权重形状: {v.shape}")
                else:
                    print(f"⚠️ 模型中不存在，跳过: {k}")

            # 加载过滤后的权重
            missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)

            if missing_keys:
                print(f"⚠️ 缺失的键: {missing_keys}")
            if unexpected_keys:
                print(f"⚠️ 意外的键: {unexpected_keys}")

            print(f"✅ 模型加载成功！加载了 {len(filtered_state_dict)}/{len(state_dict)} 个权重")
        else:
            print(f"❌ 错误：找不到模型文件 {checkpoint_path}")
            print("请确保训练已完成，且 --load 参数指向正确的模型路径。")
            return
    else:
        print("❌ 警告：未指定 --load 参数，将使用随机初始化的模型进行测试！")
        print("这会导致极低的指标。请检查你的 test_mimic_cxr.sh 脚本是否包含 --load 参数。")
        return

    criterion = compute_loss
    metrics = compute_scores

    # 6. 将规整好的模型与参数托付给 Tester，正式刷分评估
    tester = Tester(model, criterion, metrics, tokenizer, args, test_dataloader)  # ✅ 添加 tokenizer 参数
    tester.test()


if __name__ == '__main__':
    main()