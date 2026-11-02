# modules/trainer.py
import logging
import os
import time
from datetime import datetime
from abc import abstractmethod

import torch
from numpy import inf
from tqdm import tqdm


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler):
        self.args = args

        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)

        self._setup_file_logging()

        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        self.best_model_info = {
            'epoch': 0,
            'metric_name': self.mnt_metric,
            'metric_value': self.mnt_best,
            'path': '',
            'timestamp': ''
        }

        self._init_best_recorder()

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.start_time = time.time()

    def _setup_file_logging(self):
        if not hasattr(self.args, 'record_dir'):
            self.args.record_dir = 'records/'
        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(self.args.record_dir, f'training_{timestamp}.log')
        if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '') == os.path.abspath(log_file)
                   for h in self.logger.handlers):
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                datefmt='%m/%d/%Y %H:%M:%S'
            ))
            self.logger.addHandler(file_handler)
        self.log_file = log_file
        self.logger.info(f"日志文件保存到: {log_file}")

    def _init_best_recorder(self):
        val_metrics = {f'val_{k}': -inf if self.mnt_mode == 'max' else inf
                       for k in ['BLEU_1', 'BLEU_2', 'BLEU_3', 'BLEU_4', 'ROUGE_L', 'METEOR', 'CIDEr']}
        test_metrics = {f'test_{k}': -inf if self.mnt_mode == 'max' else inf
                        for k in ['BLEU_1', 'BLEU_2', 'BLEU_3', 'BLEU_4', 'ROUGE_L', 'METEOR', 'CIDEr']}
        self.best_recorder = {'val': val_metrics, 'test': test_metrics}
        self.best_recorder['val'][self.mnt_metric] = self.mnt_best
        self.best_recorder['test'][self.mnt_metric_test] = self.mnt_best

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _get_current_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        return 0.0

    def _binarize_memory(self):
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.binarize_hd_memory()
        else:
            self.model.binarize_hd_memory()

    def train(self):
        not_improved_count = 0
        self._print_training_config()
        for epoch in range(self.start_epoch, self.epochs + 1):
            epoch_start_time = time.time()
            result = self._train_epoch(epoch)
            epoch_time = time.time() - epoch_start_time
            log = {'epoch': epoch, 'epoch_time': epoch_time}
            log.update(result)
            self._record_best(log)
            self._print_epoch_info(epoch, log, epoch_time)
            best = False
            if self.mnt_mode != 'off':
                try:
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning(f"监控指标 '{self.mnt_metric}' 未找到，停止监控")
                    self.mnt_mode = 'off'
                    improved = False
                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                    self.best_model_info = {
                        'epoch': epoch,
                        'metric_name': self.mnt_metric.replace('val_', ''),
                        'metric_value': self.mnt_best,
                        'path': os.path.join(self.checkpoint_dir, 'model_best.pth'),
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    self._binarize_memory()
                    self._save_checkpoint(epoch, save_best=True)
                    self._print_best_model_info()
                else:
                    not_improved_count += 1
                if not_improved_count > self.early_stop:
                    total_time = time.time() - self.start_time
                    self.logger.info(f"🚨 训练早停！在 {self.early_stop} 个epoch内验证指标未提升")
                    self.logger.info(f"⏱️  总训练时间: {total_time/3600:.2f} 小时")
                    break
            if epoch % self.save_period == 0 and not best:
                self._save_checkpoint(epoch, save_best=False)
            self._log_to_file(epoch, log, best)
        total_time = time.time() - self.start_time
        self._binarize_memory()
        self._save_checkpoint(self.epochs, save_best=False, prefix='final_')
        self._print_training_summary(total_time)
        self._print_best_recorder()

    # ============================================================
    # ★ 重写：完整训练配置打印（含 EV-MoE / MAL / 优化器等）
    # ============================================================
    def _print_training_config(self):
        A = self.args
        g = lambda k, d='N/A': getattr(A, k, d)

        self.logger.info("=" * 80)
        self.logger.info("🚀 开始 CMN + HDC (方案B) + EV-MoE 训练流程")
        self.logger.info("=" * 80)

        # ---- 基础配置 ----
        self.logger.info("📋 基础训练配置:")
        self.logger.info(f"  📁 数据集: {g('dataset_name')}")
        self.logger.info(f"  🖼️  视觉骨干: {g('visual_extractor')}")
        self.logger.info(f"  📈 批次大小: {g('batch_size')}")
        self.logger.info(f"  📊 训练轮数: {g('epochs')}")
        self.logger.info(f"  🎯 监控指标: {g('monitor_metric')} ({self.mnt_mode})")
        self.logger.info(f"  ⏰ 早停轮数: {self.early_stop}")
        self.logger.info(f"  🌱 随机种子: {g('seed')}")
        self.logger.info(f"  💾 保存目录: {self.checkpoint_dir}")
        self.logger.info(f"  📝 日志文件: {self.log_file}")

        # ---- Transformer 配置 ----
        self.logger.info("-" * 80)
        self.logger.info("🧠 Transformer 配置:")
        self.logger.info(f"  - 层数 (num_layers): {g('num_layers')}")
        self.logger.info(f"  - 注意力头数 (num_heads): {g('num_heads')}")
        self.logger.info(f"  - 模型维度 (d_model): {g('d_model')}")
        self.logger.info(f"  - FFN维度 (d_ff): {g('d_ff')}")
        self.logger.info(f"  - Dropout: {g('dropout')}")
        self.logger.info(f"  - 最大序列长度: {g('max_seq_length')}")
        self.logger.info(f"  - 词频阈值 (threshold): {g('threshold')}")

        # ---- 优化器 / 学习率调度 ----
        self.logger.info("-" * 80)
        self.logger.info("⚙️  优化器配置:")
        self.logger.info(f"  - 优化器: {g('optim')}")
        self.logger.info(f"  - 视觉骨干学习率 (lr_ve): {g('lr_ve')}")
        self.logger.info(f"  - 编解码器学习率 (lr_ed): {g('lr_ed')}")
        self.logger.info(f"  - 权重衰减: {g('weight_decay')}")
        self.logger.info(f"  - 学习率调度器: {g('lr_scheduler')}")
        if g('lr_scheduler') == 'CosineWithWarmup':
            self.logger.info(f"  - Warmup轮数: {g('warmup_epochs')}")
        self.logger.info(f"  - 最小学习率 (min_lr): {g('min_lr')}")

        # ---- HDC 配置 ----
        self.logger.info("-" * 80)
        if hasattr(A, 'hd_dim'):
            self.logger.info("⚡ HDC 记忆配置 (方案B):")
            self.logger.info(f"  - 视觉特征维度 (d_vf): {g('d_vf')}")
            self.logger.info(f"  - HD维度: {g('hd_dim')}")
            self.logger.info(f"  - Patch数量: {g('num_patches')}")
            self.logger.info(f"  - HDC注入比例 (hdc_scale): {g('hdc_scale')}")
            self.logger.info(f"  - 替换原CMN记忆矩阵: 是 (共享memory_matrix已移除)")

        # ---- EV-MoE 配置 ----
        self.logger.info("-" * 80)
        use_evmoe = g('use_evmoe', True)
        self.logger.info(f"🔥 EV-MoE 配置: {'✅ 启用' if use_evmoe else '❌ 禁用 (基线 concat)'}")
        if use_evmoe:
            self.logger.info(f"  - 专家数量 (num_experts): {g('num_experts')}")
            self.logger.info(f"  - VFEM: 视图特定特征增强 (跨视角交叉注意力)")
            self.logger.info(f"  - DMFM: 动态多视角融合 (MoE)")

        # ---- MAL 配置 ----
        use_mal = g('use_mal', True)
        self.logger.info(f"🎯 MAL (多视角对齐损失): {'✅ 启用' if use_mal else '❌ 禁用'}")
        if use_mal:
            self.logger.info(f"  - 平衡系数 (mal_alpha): {g('mal_alpha')}")
            self.logger.info(f"  - 温度系数 (mal_tau): {g('mal_tau')}")

        # ---- 评测 / 采样 ----
        self.logger.info("-" * 80)
        self.logger.info("📊 评测 / 采样配置:")
        self.logger.info(f"  - 实际评测方法: {g('sample_method')} (由 --sample_method 控制)")
        self.logger.info(f"  - Beam Size: {g('beam_size')}")
        self.logger.info(f"  - Length Penalty: {g('length_penalty')}")

        self.logger.info("=" * 80)

    def _print_epoch_info(self, epoch, log, epoch_time):
        train_loss = log.get('train_loss', 0)
        train_mal_loss = log.get('train_mal_loss', None)
        val_bleu1 = log.get('val_BLEU_1', 0)
        val_bleu2 = log.get('val_BLEU_2', 0)
        val_bleu3 = log.get('val_BLEU_3', 0)
        val_bleu4 = log.get('val_BLEU_4', 0)
        val_rouge = log.get('val_ROUGE_L', 0)
        val_meteor = log.get('val_METEOR', 0)
        val_cider = log.get('val_CIDEr', 0)
        test_bleu1 = log.get('test_BLEU_1', 0)
        test_bleu2 = log.get('test_BLEU_2', 0)
        test_bleu3 = log.get('test_BLEU_3', 0)
        test_bleu4 = log.get('test_BLEU_4', 0)
        test_rouge = log.get('test_ROUGE_L', 0)
        test_meteor = log.get('test_METEOR', 0)
        test_cider = log.get('test_CIDEr', 0)
        progress = f"[{epoch:3d}/{self.epochs:3d}]"
        progress_bar = self._create_progress_bar(epoch, self.epochs, width=30)
        self.logger.info("")
        self.logger.info("▬" * 60)
        self.logger.info(f"训练轮数 {progress}: {progress_bar} {100*epoch/self.epochs:.1f}%")
        self.logger.info(f"第{epoch:3d}轮训练平均损失 (总): {train_loss:.4f}")
        # ★ 新增：打印 MAL 分量（若有）
        if train_mal_loss is not None:
            self.logger.info(f"第{epoch:3d}轮 MAL 平均损失: {train_mal_loss:.4f}")
        self.logger.info(f"训练耗时: {epoch_time:.1f}s")
        self.logger.info(f"当前学习率: {self._get_current_lr():.6e}")
        self.logger.info("")
        self.logger.info("▬" * 60)
        self.logger.info("📊 验证集指标:")
        self.logger.info("  " + "─" * 52)
        self.logger.info(f"  │ {'指标':<10} │ {'BLEU-1':<8} │ {'BLEU-2':<8} │ {'BLEU-3':<8} │ {'BLEU-4':<8} │")
        self.logger.info("  " + "─" * 52)
        self.logger.info(f"  │ {'分数':<10} │ {val_bleu1:8.4f} │ {val_bleu2:8.4f} │ {val_bleu3:8.4f} │ {val_bleu4:8.4f} │")
        self.logger.info("  " + "─" * 52)
        self.logger.info(f"  │ {'其他指标':<10} │ {'ROUGE-L':<8} │ {'METEOR':<8} │ {'CIDEr':<8} │ {'':<8} │")
        self.logger.info("  " + "─" * 52)
        self.logger.info(f"  │ {'分数':<10} │ {val_rouge:8.4f} │ {val_meteor:8.4f} │ {val_cider:8.4f} │ {'':<8} │")
        self.logger.info("  " + "─" * 52)
        self.logger.info("")
        self.logger.info("📈 测试集指标:")
        self.logger.info("  " + "─" * 52)
        self.logger.info(f"  │ {'指标':<10} │ {'BLEU-1':<8} │ {'BLEU-2':<8} │ {'BLEU-3':<8} │ {'BLEU-4':<8} │")
        self.logger.info("  " + "─" * 52)
        self.logger.info(f"  │ {'分数':<10} │ {test_bleu1:8.4f} │ {test_bleu2:8.4f} │ {test_bleu3:8.4f} │ {test_bleu4:8.4f} │")
        self.logger.info("  " + "─" * 52)
        self.logger.info(f"  │ {'其他指标':<10} │ {'ROUGE-L':<8} │ {'METEOR':<8} │ {'CIDEr':<8} │ {'':<8} │")
        self.logger.info("  " + "─" * 52)
        self.logger.info(f"  │ {'分数':<10} │ {test_rouge:8.4f} │ {test_meteor:8.4f} │ {test_cider:8.4f} │ {'':<8} │")
        self.logger.info("  " + "─" * 52)
        self.logger.info("")
        self.logger.info("🏆 当前最佳 vs 历史最佳:")
        self.logger.info("  [验证集]")
        self.logger.info(f"    BLEU-4: 当前 {val_bleu4:.4f} | 最佳 {self.best_recorder['val'].get('val_BLEU_4', 0):.4f}")
        self.logger.info(f"    ROUGE-L: 当前 {val_rouge:.4f} | 最佳 {self.best_recorder['val'].get('val_ROUGE_L', 0):.4f}")
        self.logger.info("  [测试集]")
        self.logger.info(f"    BLEU-4: 当前 {test_bleu4:.4f} | 最佳 {self.best_recorder['test'].get('test_BLEU_4', 0):.4f}")
        self.logger.info(f"    ROUGE-L: 当前 {test_rouge:.4f} | 最佳 {self.best_recorder['test'].get('test_ROUGE_L', 0):.4f}")
        self.logger.info("-" * 60)

    def _print_best_model_info(self):
        info = self.best_model_info
        self.logger.info("🌟" + "=" * 60)
        self.logger.info("🌟 发现新的最佳模型！")
        self.logger.info(f"🌟 最佳模型保存: {info['path']}")
        self.logger.info(f"🌟 最佳epoch: {info['epoch']}, {info['metric_name']}: {info['metric_value']:.4f}")
        self.logger.info(f"🌟 保存时间: {info['timestamp']}")
        self.logger.info("🌟" + "=" * 60)

    def _print_training_summary(self, total_time):
        self.logger.info("")
        self.logger.info("🎉" + "=" * 60)
        self.logger.info("🎉 训练完成！")
        self.logger.info(f"🎉 总训练时间: {total_time/3600:.2f} 小时")
        self.logger.info(f"🎉 最终保存目录: {self.checkpoint_dir}")
        self.logger.info("🎉" + "=" * 60)

    def _print_best_recorder(self):
        self.logger.info("")
        self.logger.info("🏆 最佳结果记录:")
        self.logger.info("  [验证集]")
        for display_name, metric_key in [('BLEU-1','val_BLEU_1'),('BLEU-2','val_BLEU_2'),('BLEU-3','val_BLEU_3'),
                                          ('BLEU-4','val_BLEU_4'),('ROUGE-L','val_ROUGE_L'),('METEOR','val_METEOR'),('CIDEr','val_CIDEr')]:
            if metric_key in self.best_recorder['val']:
                value = self.best_recorder['val'][metric_key]
                if value not in [inf, -inf]:
                    self.logger.info(f"    {display_name:10s}: {value:.4f}")
        self.logger.info("  [测试集]")
        for display_name, metric_key in [('BLEU-1','test_BLEU_1'),('BLEU-2','test_BLEU_2'),('BLEU-3','test_BLEU_3'),
                                          ('BLEU-4','test_BLEU_4'),('ROUGE-L','test_ROUGE_L'),('METEOR','test_METEOR'),('CIDEr','test_CIDEr')]:
            if metric_key in self.best_recorder['test']:
                value = self.best_recorder['test'][metric_key]
                if value not in [inf, -inf]:
                    self.logger.info(f"    {display_name:10s}: {value:.4f}")

    def _create_progress_bar(self, current, total, width=30):
        progress = int(width * current / total)
        return "█" * progress + "░" * (width - progress)

    def _record_best(self, log):
        for key, value in log.items():
            if key.startswith('val_') and key in self.best_recorder['val']:
                if (self.mnt_mode == 'min' and value <= self.best_recorder['val'][key]) or \
                   (self.mnt_mode == 'max' and value >= self.best_recorder['val'][key]):
                    self.best_recorder['val'][key] = value
            if key.startswith('test_') and key in self.best_recorder['test']:
                if (self.mnt_mode == 'min' and value <= self.best_recorder['test'][key]) or \
                   (self.mnt_mode == 'max' and value >= self.best_recorder['test'][key]):
                    self.best_recorder['test'][key] = value

    def _log_to_file(self, epoch, log, is_best=False):
        with open(self.log_file, 'a') as f:
            f.write(f"\nEpoch {epoch}:\n")
            f.write(f"  Train Loss: {log.get('train_loss', 0):.4f}\n")
            if log.get('train_mal_loss') is not None:
                f.write(f"  Train MAL Loss: {log.get('train_mal_loss'):.4f}\n")
            f.write("  Validation Metrics:\n")
            for key, value in log.items():
                if key.startswith('val_'):
                    f.write(f"    {key}: {value:.4f}\n")
            f.write("  Test Metrics:\n")
            for key, value in log.items():
                if key.startswith('test_'):
                    f.write(f"    {key}: {value:.4f}\n")
            if is_best:
                f.write(f"  ✅ Best Model Saved: {self.best_model_info['metric_name']}={self.best_model_info['metric_value']:.4f}\n")
            f.write("-" * 50 + "\n")

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("警告: 无可用GPU，将使用CPU训练")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(f"警告: 配置使用 {n_gpu_use} 个GPU，但只有 {n_gpu} 个可用")
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False, prefix=''):
        try:
            state = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'monitor_best': self.mnt_best,
                'best_model_info': self.best_model_info,
                'best_recorder': self.best_recorder
            }
            if self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'state_dict'):
                try:
                    state['lr_scheduler'] = self.lr_scheduler.state_dict()
                except Exception as e:
                    self.logger.warning(f"保存调度器状态失败（忽略）: {e}")
            if save_best:
                filename = os.path.join(self.checkpoint_dir, 'model_best.pth')
                temp_filename = filename + '.tmp'
                try:
                    torch.save(state, temp_filename, _use_new_zipfile_serialization=True)
                    os.rename(temp_filename, filename)
                    self.logger.info(f"💾 已保存最佳模型: {filename}")
                except Exception as e:
                    self.logger.error(f"❌ 保存最佳模型失败: {e}")
                    try:
                        simple_state = {'epoch': epoch, 'state_dict': self.model.state_dict()}
                        torch.save(simple_state, filename + '.simple.pth', _use_new_zipfile_serialization=True)
                        self.logger.info(f"✅ 已保存简化版最佳模型: {filename}.simple.pth")
                    except Exception as e2:
                        self.logger.error(f"❌ 简化版保存也失败: {e2}")
            else:
                if prefix:
                    filename = os.path.join(self.checkpoint_dir, f'{prefix}checkpoint_epoch{epoch}.pth')
                else:
                    filename = os.path.join(self.checkpoint_dir, f'checkpoint_epoch{epoch}.pth')
                if epoch % 5 == 0 or epoch == self.epochs:
                    temp_filename = filename + '.tmp'
                    try:
                        torch.save(state, temp_filename, _use_new_zipfile_serialization=True)
                        os.rename(temp_filename, filename)
                        self.logger.info(f"💾 已保存检查点: {filename}")
                    except Exception as e:
                        self.logger.error(f"❌ 保存检查点失败: {e}")
        except Exception as e:
            self.logger.error(f"❌ 保存检查点时发生未知错误: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        self.logger.info(f"从检查点恢复: {resume_path}")
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'best_model_info' in checkpoint:
            self.best_model_info = checkpoint['best_model_info']
        if 'best_recorder' in checkpoint:
            self.best_recorder = checkpoint['best_recorder']
            self.best_recorder['val'][self.mnt_metric] = self.mnt_best
            self.best_recorder['test'][self.mnt_metric_test] = self.mnt_best
        if 'lr_scheduler' in checkpoint and self.lr_scheduler is not None:
            try:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                self.logger.info(f"已恢复学习率调度器状态，当前 LR={self._get_current_lr():.6e}")
            except Exception as e:
                self.logger.warning(f"恢复调度器状态失败（将从头计数）: {e}")
        self.logger.info(f"从 epoch {self.start_epoch} 恢复训练")
        self.logger.info(f"之前的最佳指标: {self.mnt_best}")


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader,
                 val_dataloader, test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args, lr_scheduler)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.train_size = len(train_dataloader.dataset)
        self.val_size = len(val_dataloader.dataset)
        self.test_size = len(test_dataloader.dataset)
        self.logger.info(f"📊 数据集大小: 训练集={self.train_size}, 验证集={self.val_size}, 测试集={self.test_size}")

    def _build_sample_opts(self):
        # ★ 修改：从 args 读取真实采样配置，支持 beam_search / greedy 切换
        # 注意：beam_search 能否正常工作取决于 models/ 里 sample 函数的实现。
        # 若历史上 beam_search 出现过 state 维度 bug（空序列、指标归零），
        # 仅改此处无法修复根因，需排查模型 beam 路径。
        sample_method = getattr(self.args, 'sample_method', 'greedy')
        beam_size = getattr(self.args, 'beam_size', 1)
        length_penalty = getattr(self.args, 'length_penalty', '')
        # greedy 模式下强制 beam_size=1、length_penalty 清空，避免下游歧义
        if sample_method == 'greedy':
            beam_size = 1
            length_penalty = ''
        return {
            'sample_method': sample_method,
            'beam_size': beam_size,
            'temperature': 1.0,
            'length_penalty': length_penalty,
            'top_k': getattr(self.args, 'top_k_sampling', 0),
            'top_p': getattr(self.args, 'top_p_sampling', 0.0),
            'group_size': 1,
        }

    def _train_epoch(self, epoch):
        self.logger.info(f"[{epoch}/{self.epochs}] 开始训练...")

        # 说明：lr_scheduler.step() 不再放在每个 batch 里，
        # 而是统一在 epoch 末尾调用一次（见本函数最后），符合 PyTorch 官方推荐的
        # "optimizer.step() 之后才能 scheduler.step()" 的顺序。

        train_loss = 0
        mal_loss_sum = 0.0      # ★ 新增：累加 MAL 损失
        mal_batches = 0         # ★ 新增：统计有 MAL 的 batch 数
        self.model.train()

        pbar = tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
            desc=f"训练 [{epoch}/{self.epochs}]",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
            ncols=100
        )

        for batch_idx, (images_id, images, reports_ids, reports_masks) in pbar:
            images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), \
                                                 reports_masks.to(self.device)

            # ===== 修改：解包 EV-MoE 返回的 (model_output, mal_loss) 元组 =====
            output = self.model(images, reports_ids, mode='train')
            if isinstance(output, tuple):
                model_output, mal_loss = output
            else:
                model_output, mal_loss = output, None
            loss = self.criterion(model_output, reports_ids, reports_masks, mal_loss=mal_loss)
            # =================================================================

            train_loss += loss.item()
            # ★ 新增：追踪 MAL 分量
            if mal_loss is not None:
                mal_loss_sum += mal_loss.item()
                mal_batches += 1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            avg_loss = train_loss / (batch_idx + 1)
            postfix = {'loss': f'{avg_loss:.4f}', 'batch': f'{batch_idx+1}/{len(self.train_dataloader)}'}
            if mal_loss is not None:
                postfix['mal'] = f'{mal_loss.item():.4f}'
            pbar.set_postfix(postfix)

            if (batch_idx + 1) % self.args.log_period == 0:
                self.logger.debug(f"Epoch {epoch}, Batch {batch_idx+1}/{len(self.train_dataloader)}, Loss: {avg_loss:.4f}")

        pbar.close()

        avg_train_loss = train_loss / len(self.train_dataloader)
        log = {'train_loss': avg_train_loss}
        # ★ 新增：写入 MAL 平均损失（仅当启用 MAL 时）
        if mal_batches > 0:
            log['train_mal_loss'] = mal_loss_sum / mal_batches

        val_metrics = self._evaluate_epoch(epoch, 'validation', self.val_dataloader)
        log.update(val_metrics)

        test_metrics = self._evaluate_epoch(epoch, 'test', self.test_dataloader)
        log.update(test_metrics)

        # 每个 epoch 结束后更新一次学习率（PyTorch 官方推荐顺序：optimizer.step() 之后）
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            self.logger.info(f"📉 Epoch {epoch} 结束学习率: {self._get_current_lr():.6e}")

        return log

    def _evaluate_epoch(self, epoch, mode, dataloader):
        if mode == 'validation':
            desc = f"验证 [{epoch}/{self.epochs}]"
            prefix = 'val_'
            dataset_size = self.val_size
        else:
            desc = f"测试 [{epoch}/{self.epochs}]"
            prefix = 'test_'
            dataset_size = self.test_size

        self.model.eval()
        with torch.no_grad():
            gts, res = [], []
            pbar = tqdm(
                enumerate(dataloader),
                total=len(dataloader),
                desc=desc,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
                ncols=100
            )
            sample_opts = self._build_sample_opts()
            for batch_idx, (images_id, images, reports_ids, reports_masks) in pbar:
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output, _ = self.model(images, mode='sample', update_opts=sample_opts)
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                # ★ 防御性检测：beam_search 可能产出空序列（历史 state 维度 bug 的表现）
                empty_count = sum(1 for r in reports if not r or not str(r).strip())
                if empty_count > 0:
                    self.logger.warning(
                        f"⚠️ [{mode}] 检测到 {empty_count}/{len(reports)} 条空生成结果。"
                        f"若比例较高，说明 beam_search 路径仍有 bug，建议回退 greedy 或修复 models/ 的 sample 实现。"
                    )
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                res.extend(reports)
                gts.extend(ground_truths)
                processed = len(gts)
                pbar.set_postfix({'已处理': f'{processed}/{dataset_size}'})
            pbar.close()

            raw_metrics = self.metric_ftns({i: [gt] for i, gt in enumerate(gts)},
                                           {i: [re] for i, re in enumerate(res)})
            metrics = {k.upper() if not k.startswith(('BLEU', 'ROUGE', 'METEOR', 'CIDEr')) else k: v
                       for k, v in raw_metrics.items()}
            if 'CIDEr' not in metrics:
                metrics['CIDEr'] = 0.0
            prefixed_metrics = {f'{prefix}{k}': v for k, v in metrics.items()}
            return prefixed_metrics