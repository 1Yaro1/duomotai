# GAIA DB 单模态与简单多模态基线：训练前合同与代码审计

## 状态

- 分支：`codex/db-unimodal-baselines`
- 基线起点：`6e87436ea6893cd2ac4e7a59256353cf55830d8b`
- 训练合同：`config/autoresearch/baseline_training_contract.json`
- 训练合同 SHA256：`ee5803930613b67f978f226ea591e75939adcef678546e433e48a6be4ab2d9bf`
- 冻结评分合同 SHA256：`ca2fbf7257705597488be0933e8d7bef548d94d606b3a01a8a7253badf07b3e6`
- `training_started=false`
- `test_used=false`

本阶段只完成合同、实现和只读 preflight。没有创建 GAIA 训练输出目录，没有执行真实 checkpoint/GAIA 模型前向，也没有产生实验 optimizer update；恢复测试中的微型 toy network 会执行内存内 optimizer step，仅验证 checkpoint 恢复机制，不接触实验数据。

## 固定训练协议

| 项目 | 冻结值 |
|---|---|
| 服务 | GAIA_dbservice1、GAIA_dbservice2；缺一则整个模型实验失败 |
| gradient train | `[0,4515)` |
| early stop | `[4515,5644)` |
| proposal validation | `[5644,7056)` |
| final test | 禁止 |
| 第一轮 seed | 2021 |
| batch / epoch | 8 / 3 |
| window / patch / stride | 24 / 6 / 6 |
| optimizer | Adam，lr=1e-4，betas=(0.9,0.999)，eps=1e-8，weight_decay=0 |
| scheduler | `adjust_learning_rate(type1)`；各训练 epoch 的 lr 为 1e-4、1e-4、5e-5 |
| early stop | 固定 seed 的 `[4515,5644)` reconstruction MSE，越低越好 |
| checkpoint | step 1、每 50 step、每 epoch 的 last；严格改善时 best；评分前 evaluation_start |
| 最大更新数 | 每服务 1686（4492 个窗口，562 update/epoch × 3） |
| 数据顺序 | 每 epoch 使用私有 CPU Generator、seed+epoch 的完整 `randperm`；不丢尾批 |
| loss（B0–B4） | reconstruction MSE + 固定 IB；无 intra/inter contrastive loss |
| score | 冻结的 `deterministic_overlap_v1`，61 个阈值，no point adjustment |

数值输入只允许来自已经核验的 7056 行 prefix-only cache；runner 不包含原始 CSV 路径，也不会打开原始 CSV。日志缓存仅覆盖 `[0,7056)`，manifest 明确记录 `test_features_created=false`。

## 模型实现

| ID | 模型 | 指标统计文本 | 日志输入 | 关键控制 |
|---|---|---:|---|---|
| B0 | MetricOnlyPure | 否 | 不读取；零占位且绕过日志融合 | 纯指标重建基线 |
| B1 | MetricOnlyStats | 是 | 不读取；零占位且绕过日志融合 | 相对 B0 测统计文本 |
| B2 | MetricOnlyNullLog | 是 | 全零 features、present=0、learnable missing token | 与 B3/B4 容量匹配 |
| B3 | MetricLogShifted | 是 | 每个 split 内固定循环 `+240` 分钟 | 与 B4 同结构，破坏时间对应关系 |
| B4 | MetricLogSimple | 是 | 正确对齐日志 v2 | 简单真实多模态基线 |
| B5 | FullDualReference | 历史实现 | 历史正确对齐日志 | 只引用已有 checkpoint/确定性结果，禁止重训 |

B2、B3、B4 使用同一个 `MINDTSModel` 前向分支、MinuteLogEncoder、prompt-log fusion、IB 和 reconstruction head。它们在代码中的唯一实验变量是 `BaselineLogView`：`null`、`shifted` 或 `aligned`。

固定错位定义为：

```text
source_log_index = split_start
                 + ((metric_minute - split_start + 240) mod split_length)
```

训练、早停和验证分别循环，不能跨越 4515、5644 或 7056 边界，也不搜索 shift。

## 冻结编码器状态

- 新基线 B0–B4 中的 Qwen 参数设置为 `requires_grad=false`，不进入 optimizer，并在统计 prompt 编码时使用 eval mode。
- BERT 不在训练时运行；日志 v2 直接读取已冻结的 BERT 语义缓存。缓存记录 BERT `model.safetensors` SHA256 为 `68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3`。
- B0、B1、B2 不构造真实 `LogV2Store`。B2 的日志维度由合同固定为 semantic=768、count=12。

## 评分与结果输出

每个新 checkpoint 使用同一个 `score_threshold_and_prediction()` 同时生成训练阈值分数和验证预测；固定 six-mask bank、masked-only squared error、overlap-add mean、soft gate、step=1、no tail padding 和 channel mean。

runner 会进行两次完整评分，并在改变全局 RNG 后逐数组比较；不一致立即失败。结果包括：

- AP、VUS-PR；
- Precision、Recall、Point-F1、AUC、Affiliation-F、VUS-ROC；
- Event Precision、Event Recall、Event IoU；
- 距离真实事件超过 10 分钟的 FP 点数；
- 独立误报段数和每千正常点误报段数；
- FP 的 recurring top-error channels。

## 判定规则

“正确对齐真实日志有增量”要求 B4 的 AP 在 dbservice1 和 dbservice2 上都分别高于 B2 和 B3，同时两个服务的 VUS-PR 均不得低于相应控制组。只有 seed=2021 全部成功并满足该门槛，才允许追加 seed=2022、2023，并且最终必须报告全部 seed，不能挑最好 seed。

B5 是历史 checkpoint。即便 `AP(B5) > AP(B4)`，也只能作为描述性参考，不能单独证明 dual objective 的因果增益；严格因果结论仍需要同一代码、初始化和训练协议下只改变 objective 的 matched ablation。

## 审计证据

真实 prefix-only cache 和 log-v2 manifest 的只读 preflight 输出：

```json
{"baseline_preflight":"ok","models":["B0","B1","B2","B3","B4","B5"],"seed":2021,"training_started":false,"test_used":false,"contract_sha256":"ee5803930613b67f978f226ea591e75939adcef678546e433e48a6be4ab2d9bf"}
```

CPU 单元测试：41 项通过，1 项因 CPU-only 条件正常跳过。覆盖原 deterministic scoring、prompt median、contrastive、checkpoint/resume 测试，以及新增的 null/no-log、split-safe +240 shift、split crossing 拒绝、step=1 数据集、B2/B3/B4 参数拓扑一致、合同哈希和 B5 禁止因果声明测试。

关键实现文件 SHA256：

| 文件 | SHA256 |
|---|---|
| `scripts/autoresearch/gaia_db_baseline_runner.py` | `1f7ae9c907f3b56c7ead738a1c9addad03a2510a72b7c9167afa0b1ed2d0d3aa` |
| `ts_benchmark/baselines/MindTSDBBaselines.py` | `f31d7b56c413e95fc50292f21009ca92869eca12498710c899158255b30e1c47` |
| `ts_benchmark/baselines/MindTS/cmc_log_data.py` | `35cdc25a5bddde2897e893bc32c49f3d594522ef1bef94f723b54af72f8c7a21` |
| `ts_benchmark/baselines/MindTS/models/MindTS_model.py` | `3c6158539f2990ce319cd78b1515db9e97f986dd82b570811a677a32f1910fd2` |
| `ts_benchmark/baselines/MindTSWeb.py` | `5d44043cb6609b7705c85b085fb360ab747902d6fa409c9ae79afa22826ad45c` |
| `tests/test_db_baseline_contract.py` | `57d3bf187e012f300918103eff8a3c28451f75cab9f69a583561fea8bf790fd3` |

注意：上表为训练前审计时的内容哈希；任何后续代码改动都会导致 runner 的 clean-worktree/identity guard 拒绝启动。

## 训练启动门槛

只有在以下条件同时满足后才应运行 `--run-training`：

1. 当前代码已提交，Git 工作区干净；
2. 合同、评分合同、prefix cache、log manifest 哈希全部通过；
3. 输出目录位于 Git 仓库外且此前不存在；
4. 一次只运行一个模型，两个服务顺序执行；
5. 服务器只暴露一张目标 GPU；
6. 用户明确授权开始训练。

本审计完成时尚未满足第 6 条，因此没有启动训练。
