# IU X-Ray：HDC、MAL 修复与参数实验

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

先跑当前候选配置：多正样本 MAL + `mal_weight=0.1`，结果保存到
`results/iu_xray_hdc_mal_fixed_w01`；其他主要训练超参数沿用原脚本：

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
`mal_alpha` 调节 MAL 内部对比/重建比例；`mal_weight` 独立控制 MAL 总权重，详见下一节。
当前 CosineWithWarmup 实现未使用 `min_lr`，改变它不会调整该调度器的最低学习率。
脚本注释中的梯度裁剪原来没有对应实现；本次已移除误导注释，没有额外修改训练梯度策略。

## 验证集比较解码和记忆权重（无需重训）

以下示例针对当前脚本的候选配置。若训练时修改了模型或 HDC 设置，测试时传入相同设置。

```bash
python main_test.py \
  --load results/iu_xray_hdc_mal_fixed_w01/model_best.pth \
  --eval_split val --sample_method greedy --beam_size 1 \
  --hd_memory_weight 0.2 --save_dir results/iu_xray_eval_val_greedy

python main_test.py \
  --load results/iu_xray_hdc_mal_fixed_w01/model_best.pth \
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

## METEOR 在第一个 epoch 验证时出现 BrokenPipeError

仓库原先缺少 `pycocoevalcap/meteor/data/paraphrase-en.gz`。Java 会先报告 file not found 并退出，
旧 Python 包装器未展示 stderr，随后只显示 BrokenPipeError。另一个 verbose 参数 TypeError
来自给不支持该参数的 scorer 传参，旧代码会捕获重试；现在只向 BLEU 传 verbose。

更新代码后，IU 训练脚本会先执行数据准备。也可以单独执行：

```bash
python scripts/setup_meteor.py
java -version
```

安装脚本通过当前 Python 的 pip 下载固定的 `pycocoevalcap==1.2` wheel（约 104 MB），
校验 wheel 和数据文件的 SHA-256，仅提取约 62 MB 的英文资源，不安装或替换 Python 包。
以后运行只校验本地文件。若服务器不能访问包索引，可以在另一台机器下载 wheel 后上传：

```bash
python -m pip download --no-deps --only-binary=:all: pycocoevalcap==1.2
python scripts/setup_meteor.py --wheel /path/to/pycocoevalcap-1.2-py3-none-any.whl
```

需要可执行的 Java 运行时。训练/测试入口现在会在加载数据和训练前实际运行一次 METEOR 检查。
若 Java 失败，保留其 stderr；METEOR 不会被跳过或记成零分。

```bash
OMP_NUM_THREADS=1 python -m pytest -q tests/test_hd_memory.py tests/test_meteor.py tests/test_mal.py
```

真实 Java 集成测试在缺少 Java 或数据时明确标记 skipped；准备好依赖后应全部执行。

## 多正样本 MAL 与独立总权重

对比损失改为平均所有正样本的负对数概率。原来的 class-0 交叉熵只奖励第一个正样本，
其他正样本仍参与竞争。现在同一样本的其他视角全部得到相同的目标权重，自身配对被排除。

```text
MAL = (1 - mal_alpha) * contrastive + mal_alpha * reconstruction
loss = report_loss + mal_weight * MAL
```

Python 参数默认 `mal_weight=1.0`，保留未显式传参时的辅助项系数。
`train_iu_xray.sh` 显式设置候选 `mal_weight=0.1`，且保持 `mal_alpha=0.1`、`mal_tau=0.1`。
这使总损失中的对比项系数为 0.09、重建项系数为 0.01。
`mal_weight=0` 不引入 MAL 梯度，但仍计算它；`--use_mal false` 则关闭 MAL 计算。
启动日志打印该权重，逐批和 epoch 日志分别记录报告损失、原始 MAL、加权 MAL。
进度条 loss 为运行均值，report/mal/w_mal 为当前批次；epoch 分量采用批次均值。

运行与对照（每次使用独立目录；建议从头训练，同一随机种子比较）：

```bash
# 当前候选：0.1
bash train_iu_xray.sh

# 同样的修正实现，保持原来的辅助项系数 1.0
bash train_iu_xray.sh --mal_weight 1.0 \
  --save_dir results/iu_xray_hdc_mal_fixed_w1

# 关闭 MAL 的消融
bash train_iu_xray.sh --use_mal false \
  --save_dir results/iu_xray_hdc_no_mal
```

本次没有再调整学习率或 HDC 设置。MAL 无新增可训练参数，因此模型权重格式不变，
但旧模型没有接受修正后的目标训练，直接推理不会获得这项训练改进。
多正样本损失的绝对数值可能比旧版高：两个正样本时，纯对比项的下界为 log(2)。
因此不应跨版本直接比较原始 MAL 大小；用验证集选定 checkpoint 后比较完整 BLEU 指标。

测试包含与独立标量定义的数值/梯度对照、视角顺序不变性、单样本和无视角边界、
辅助权重的梯度缩放，以及实际 Trainer 更新和真实 ResNet101 双图模型的联合反向传播。
