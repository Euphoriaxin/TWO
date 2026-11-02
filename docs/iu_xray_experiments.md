# IU X-Ray：HDC 修复与参数实验

本次修复不是已验证的 BLEU 提升。需要在相同划分、词表、参考文本处理方式下重新训练和评估。

## 记忆如何参与预测

- 视觉融合特征经 HDC 得到图像查询；与当前已输入词的 HDC 向量绑定。
- 位置 t 的原型按 **下一个词** 分类累计。查询不包含待预测词或未来词。
- `logits = neural_logits + hd_memory_weight * similarity / hd_memory_temperature`。
- similarity 是查询与二值原型的归一化内积，范围 [-1, 1]；未观察过的原型贡献 0。
- teacher forcing、greedy、beam 使用同一逻辑；记忆无梯度，仅训练时在读取后更新。
- 包含首词和 EOS，忽略 padding；重复索引会累加。保存时保留累计值，不破坏性二值化。
- `use_hd_memory=false` 完全关闭读写；`hd_memory_weight=0` 关闭读取，但仍可在训练时积累原型。
- `update_hd_memory=false` 禁止写入。评估模式始终禁止写入。
- `hdc_scale` 控制图像/文本特征增强，与词表 logits 的 `hd_memory_weight` 是两个独立参数。
- `hd_proj.requires_grad_(False)` 冻结权重与偏置；统一 Xavier 初始化跳过冻结的 HDC 参数。

旧 checkpoint 缺少 `hd_memory.counts` 时会警告并清空旧原型（旧原型关联当前词、包含 padding，不能直接沿用）。
其余权重仍可加载；建议本轮从头训练以获得一致的初始化和冻结行为。直接加载旧权重评估不会自动获得新记忆。
`main_test.py` 与训练共用参数定义，严格加载权重；不再悄悄过滤形状不匹配的层。必须保持模型尺寸、词表阈值、HDC 等设置与训练一致。

## 推荐顺序

先跑修复后的原参数基线（结果使用新目录，不覆盖旧实验）：

```bash
bash train_iu_xray.sh
```

随后只修改学习率与 warmup，其他参数保持相同：

```bash
bash train_iu_xray.sh --lr_ve 1e-5 --lr_ed 1e-4 --warmup_epochs 5 \
  --save_dir results/iu_xray_hdc_low_lr
```

这是候选设置，不是已确定的最佳值。保持 seed=7580、batch=16、max_seq_length=60、threshold=3。
训练脚本末尾支持覆盖参数。可在选定学习率后比较 `--hdc_scale 0.1` 与 `0.3`，分别保存到不同目录。
`mal_alpha` 只调节 MAL 内部对比/重建比例，不是 MAL 总权重。本次没有修改 MAL。
当前 CosineWithWarmup 实现未使用 `min_lr`，改变它不会调整该调度器的最低学习率。
脚本注释中的梯度裁剪原来没有对应实现；本次已移除误导注释，没有额外修改训练梯度策略。

## 验证集比较解码和记忆权重（无需重训）

以下示例针对修复后的原参数基线。若训练时修改了模型或 HDC 设置，测试时传入相同设置。

```bash
python main_test.py \
  --load results/iu_xray_hdc_fixed/model_best.pth \
  --eval_split val --sample_method greedy --beam_size 1 \
  --hd_memory_weight 0.2 --save_dir results/iu_xray_eval_val_greedy

python main_test.py \
  --load results/iu_xray_hdc_fixed/model_best.pth \
  --eval_split val --sample_method beam_search --beam_size 3 --length_penalty 0.6 \
  --hd_memory_weight 0.2 --save_dir results/iu_xray_eval_val_beam3
```

同一 checkpoint 可比较 `hd_memory_weight = 0 / 0.1 / 0.2`，固定 temperature=0.1。
先比较 greedy 与 beam=3，再按结果决定是否测 beam=5；beam 较大不保证更好。
用验证集选模型、解码与记忆权重；确定配置后用 `--eval_split test` 评估一次。
每次使用不同 save_dir 保留生成报告。训练末尾的“最佳结果记录”是各指标跨 epoch 独立最大值，
不是同一个 checkpoint 的整组指标；请报告选定 model_best.pth 的完整 BLEU-1～4。

## 回归验证

```bash
OMP_NUM_THREADS=1 python -m pytest -q tests/test_hd_memory.py
```

测试覆盖重复原型累计、EOS/padding、读取先于更新、冻结参数、teacher forcing 与逐词查询一致、
greedy/beam 读取记忆、评估不写记忆、checkpoint 往返、旧原型重置和真实 ResNet101 双图前向/反向。
测试使用随机图像和小维度模型，不代表实际 IU X-Ray BLEU 或 GPU 训练吞吐。
