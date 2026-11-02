# modules/visual_extractor.py
import torch
import torch.nn as nn
import torchvision.models as models

# 延迟导入，避免未安装 timm/biomedclip 时报错
try:
    import timm
except ImportError:
    timm = None

try:
    import biomedclip
except ImportError:
    biomedclip = None


class VisualExtractor(nn.Module):
    """
    通用视觉特征提取器，支持多种骨干网络。
    自动将 d_vf（特征维度）和 num_patches_single（单图 patch 数）同步到 args，
    供后续 att_embed 层和 HDC 模块使用。
    """

    # 骨干配置：(输出特征维度 d_vf, 特征图空间边长 feat_map, 类型 cnn/vit)
    BACKBONE_CONFIGS = {
        'resnet50': {'d_vf': 2048, 'feat_map': 7, 'type': 'cnn'},
        'resnet101': {'d_vf': 2048, 'feat_map': 7, 'type': 'cnn'},
        'densenet121': {'d_vf': 1024, 'feat_map': 7, 'type': 'cnn'},
        'vit_base_patch16': {'d_vf': 768, 'feat_map': 14, 'type': 'vit'},
        'biomedclip': {'d_vf': 512, 'feat_map': 14, 'type': 'vit'},
    }

    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained

        if self.visual_extractor not in self.BACKBONE_CONFIGS:
            raise ValueError(
                f"Unsupported visual extractor: {self.visual_extractor}. "
                f"Choose from {list(self.BACKBONE_CONFIGS.keys())}"
            )

        config = self.BACKBONE_CONFIGS[self.visual_extractor]
        self.d_vf = config['d_vf']
        self.feat_map = config['feat_map']
        self.num_patches_single = self.feat_map ** 2  # 单图 patch 数 (49 或 196)
        self.extractor_type = config['type']

        # ★ 自动同步 d_vf 到 args（att_embed 层的输入维度依赖此值）
        args.d_vf = self.d_vf

        self._build_model()

        # 全局池化层（仅 CNN 需要；ViT 用 cls token 作全局特征）
        if self.extractor_type == 'cnn':
            self.avg_fnt = nn.AdaptiveAvgPool2d(1)
        else:
            self.avg_fnt = None

    def _build_model(self):
        name = self.visual_extractor
        if name in ['resnet50', 'resnet101']:
            model = getattr(models, name)(pretrained=self.pretrained)
            modules = list(model.children())[:-2]
            self.model = nn.Sequential(*modules)

        elif name == 'densenet121':
            # model = models.densenet121(pretrained=False)

            # 改法 A：用 ImageNet 预训练（最简单）
            model = models.densenet121(pretrained=True)  # torchvision 老版本
            # 或新版本：
            # model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

            # 注：torchvision 的 densenet121 默认在 ImageNet 预训练。
            # 如需 CheXNet (ChestX-ray14) 权重，请自行下载并加载：
            #   state = torch.load('chexnet_densenet121.pth')
            #   self.model.load_state_dict(state)
            modules = list(model.children())[:-1]  # 取到 feature layer (B, 1024, 7, 7)
            self.model = nn.Sequential(*modules)

        elif name == 'vit_base_patch16':
            if timm is None:
                raise ImportError("请先安装 timm: pip install timm")
            # num_classes=0 直接输出 patch embeddings，不含分类头
            self.model = timm.create_model(
                'vit_base_patch16_224', pretrained=self.pretrained, num_classes=0
            )

        elif name == 'biomedclip':
            if biomedclip is None:
                raise ImportError("请先安装 biomedclip: pip install biomedclip")
            # BiomedCLIP 的视觉塔（在 MIMIC-CXR 上图文对比预训练）
            pretrained_model, _ = biomedclip.create_model_and_transforms(
                'biomedclip_local', pretrained=self.pretrained
            )
            self.model = pretrained_model.visual

    def forward(self, images):
        if self.extractor_type == 'cnn':
            # images: (B, 3, 224, 224) -> patch_feats: (B, d_vf, H, W)
            patch_feats = self.model(images)
            avg_feats = self.avg_fnt(patch_feats).squeeze(-1).squeeze(-1)  # (B, d_vf)
            batch_size, feat_size, _, _ = patch_feats.shape
            # 展平空间维度 -> (B, S, d_vf)
            patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
            return patch_feats, avg_feats

        else:  # ViT / BiomedCLIP
            out = self.model(images)
            # 兼容不同返回格式：元组 (global, patches) 或 张量 (B, N+1, d_vf)
            if isinstance(out, (tuple, list)):
                cls_token = out[0]  # (B, d_vf)
                patch_tokens = out[-1]  # (B, N, d_vf)
            else:
                cls_token = out[:, 0, :]  # (B, d_vf)
                patch_tokens = out[:, 1:, :]  # (B, N, d_vf)
            return patch_tokens, cls_token