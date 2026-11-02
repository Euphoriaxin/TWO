# modules/dataloaders.py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .datasets import IuxrayMultiImageDataset, MimiccxrSingleImageDataset


class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

        # ===== 根据视觉骨干选择归一化参数（新增）=====
        extractor = getattr(args, 'visual_extractor', 'resnet101')
        if extractor == 'biomedclip':
            # BiomedCLIP 专用预处理统计量
            norm_mean = [0.48145466, 0.4578275, 0.40821073]
            norm_std = [0.26862954, 0.26130258, 0.27577711]
        else:
            # ImageNet 统计量（ResNet/DenseNet/timm-ViT 均适用）
            norm_mean = [0.485, 0.456, 0.406]
            norm_std = [0.229, 0.224, 0.225]
        # =============================================

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)])

        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        else:
            self.dataset = MimiccxrSingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        image_id_batch, image_batch, report_ids_batch, report_masks_batch, seq_lengths_batch = zip(*data)
        image_batch = torch.stack(image_batch, 0)
        max_seq_length = max(seq_lengths_batch)

        target_batch = np.zeros((len(report_ids_batch), max_seq_length), dtype=int)
        target_masks_batch = np.zeros((len(report_ids_batch), max_seq_length), dtype=int)

        for i, report_ids in enumerate(report_ids_batch):
            target_batch[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(report_masks_batch):
            target_masks_batch[i, :len(report_masks)] = report_masks

        return image_id_batch, image_batch, torch.LongTensor(target_batch), torch.FloatTensor(target_masks_batch)