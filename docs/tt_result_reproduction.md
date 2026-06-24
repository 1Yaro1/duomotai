# TT 实验结果保留与复现说明

这个文件是为了记录：清理 `result/label/TT` 目录后，当前保留了哪些有效结果，以及以后如何重新跑出两类 TT 实验。

两类实验分别是：

1. **包含文本模态的实验**：`TT_text.csv` 里使用从日志提取出来的文本摘要。
2. **空文本/仅时间序列实验**：`TT_text.csv` 里每个时间步都是占位符 `.`，相当于不提供有效文本信息，只保留时间序列输入。

注意：清理结果目录前，`result/label/TT` 里只有一组真正有效的 TT 结果，也就是“包含日志文本模态”的结果。之前其它 batch_size=1 文件都是单行错误记录或者 NaN 报告，不是有效实验结果。因此目前只保留了这一组有效结果；空文本实验需要以后按下面命令重新跑。

## 当前保留的有效结果

保留的有效结果文件是：

- `result/label/TT/test_report.1782114323.CUDA12-1-T640-GPU1.2327714.csv`
- `result/label/TT/MindTS.1782114323.CUDA12-1-T640-GPU1.2327714.csv.tar.gz`

这组结果对应的是：**TT 数据集 + 日志文本模态 + batch_size=1**。

报告里的关键指标是：

```text
affiliation_f = 0.6055847572114321
```

两个保留文件的校验值如下，用来确认文件没有被意外改动：

```text
2a55b301d3556bdb9f95bf1a1b3d39392b6f52d9bdb7a16a898a606cd87096c9  test_report.1782114323.CUDA12-1-T640-GPU1.2327714.csv
b7b30befed662d5befd3979071b8eb4c0902cae17b127155916be31d29a78fda  MindTS.1782114323.CUDA12-1-T640-GPU1.2327714.csv.tar.gz
```

## 文件含义

`test_report...csv` 是最终汇总报告，只保留了报告指标，例如 `affiliation_f`。

`MindTS...csv.tar.gz` 是压缩后的原始评估结果，里面包含每个 anomaly ratio 下的详细指标。这个文件比 `test_report` 更完整。

`TT_text.csv` 是多模态实验里的文本输入文件。这个文件的内容决定实验是“有文本模态”还是“空文本模态”。

## 复现：包含日志文本模态

如果要恢复“包含日志文本模态”的 TT 文本输入，运行：

```bash
git show 88249f0e952e8b223ef0305ae63c6715805b7892:dataset/anomaly_detect/data/TT_text.csv > dataset/anomaly_detect/data/TT_text.csv
```

然后运行 batch_size=1 的 TT 实验：

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONUNBUFFERED=1 python ./scripts/run_benchmark.py \
  --config-path "unfixed_detect_label_multi_config.json" \
  --data-name-list "TT.csv" \
  --model-name "MindTS.MindTS" \
  --model-hyper-params '{"batch_size": 1, "d_ff": 8, "d_model": 256, "e_layers": 1, "horizon": 0, "norm": true, "num_epochs": 1, "seq_len": 24, "patch_size": 6, "stride": 6, "mask_ratio": 0.4, "r": 0.5, "enc_in_time": 189, "parallel_strategy": "DP"}' \
  --gpus 0 \
  --num-workers 0 \
  --timeout 60000 \
  --save-path "label/TT" \
  --text-name-list "TT_text.csv"
```

这就是当前保留结果对应的实验设置。

## 复现：空文本/仅时间序列

如果要恢复“空文本模态”的 TT 文本输入，运行：

```bash
git show 5b2d5aff587c0cf570551a46ac0e68f74d07956e:dataset/anomaly_detect/data/TT_text.csv > dataset/anomaly_detect/data/TT_text.csv
```

这个版本的 `TT_text.csv` 基本都是 `.`，也就是不提供真实日志文本。然后运行和上面一样的 batch_size=1 命令即可。

空文本版本 `TT_text.csv` 的校验值是：

```text
ce20f1c2089716c10e7440b5227f4539e07d68f0f6b4a867c1164708566b35a3  TT_text.csv from 5b2d5af
```

跑完空文本实验后，如果想切回“包含日志文本模态”，再执行上一节的 `git show 88249f0... > TT_text.csv` 命令即可。

## 重要提醒

当前 `scripts/multivariate_detection/detect_label/TT_script/MindTS.sh` 被改成了 `batch_size=8`，这个配置在当前 GPU 状态下容易 OOM。

如果目的是复现当前保留的有效结果，请使用本文档里的 `batch_size=1` 命令，而不是直接运行当前的 `MindTS.sh`。

GPU 深度学习实验可能受到 CUDA 算子非确定性的影响。代码里设置了 seed，但如果环境、显卡状态、依赖版本或显存占用变化，重新运行时结果可能有很小差异。
