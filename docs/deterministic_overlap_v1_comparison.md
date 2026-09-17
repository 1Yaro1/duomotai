# Historical 与 deterministic_overlap_v1 评分协议对比

本文记录 `codex/db-deterministic-overlap-v1` 相对历史 MindTS 评分路径的
修改内容，并汇总同一组既有 DB checkpoint 在两套协议下的验证结果。

## 重要解释边界

- 对比对象是**评分协议**，不是两个重新训练的模型版本。
- 两套结果使用相同的既有 `best.pt` 权重和同一验证区间 `[5644,7056)`。
- 新协议没有执行训练更新，最终测试区间 `[7056,10080)` 未使用。
- 因此下表差值只能称为“确定性重评分后的指标差异”，不能称为
  “修改模型后提升”或“重新训练后提升”。
- Historical 结果保持不变，新协议使用独立名称
  `deterministic_overlap_v1`，不得覆盖历史结果。
- `Affiliation-F` 不是 ordinary point-F1；`PR` 表示 Precision，`AP` 表示
  Average Precision。

## 1. 评分计算口径修改

| 环节 | Historical | deterministic_overlap_v1 |
| --- | --- | --- |
| 窗口步长 | 阈值路径存在强制大步长的历史行为 | 训练分数与验证分数统一 `step=1` |
| 绝对位置 | 未形成完整的绝对时间点 overlap-add 序列 | Dataset 返回窗口绝对起点并回填到时间点 |
| 遮挡 | 推理时随机遮挡 | 固定枚举 `C(4,2)=6` 种 patch 对 |
| 遮挡覆盖 | 依赖随机状态和访问顺序 | 每个窗口点恰好被遮挡 3 次 |
| 重建误差 | 历史窗口聚合路径 | 只累计被遮挡位置的平方误差 |
| 窗口内归约 | 历史实现 | 先对 3 次 masked observation 取均值 |
| 跨窗口归约 | 历史拼接/截断路径 | overlap-add 后按 coverage 取均值 |
| 通道归约 | 通道均值 | 保持通道均值，并额外保存逐通道误差 |
| 门控 | hard Gumbel gate，依赖随机数 | deterministic soft gate |
| 尾部 | 最后 20 个验证点补零 | 禁止 `np.pad`，输出严格完整长度 |
| 分数长度 | 历史路径存在尾部覆盖问题 | 每个服务严格 1412 点 |
| threshold 与 prediction | 历史路径可能采用不同遍历方式 | 两者调用同一 `score_series()` |
| 确定性验证 | 固定 seed 不能保证重复前向使用同一随机状态 | 改变全局 RNG 后完整运行两次并逐字节比较 |
| 错误处理 | 历史运行路径 | fail fast；禁止异常后返回默认 0 分 |

保持不变的阈值规则：

- 仍使用固定的 61 个 anomaly-ratio candidates；
- 仍在 proposal validation 上选择 ordinary point-F1 最大值；
- 相同 F1 时保留冻结顺序中的第一个候选；
- 不使用 point adjustment；
- 两个服务等权宏平均；
- AP 和 VUS-PR 作为后续模型比较的主要指标。

## 2. 代码修改位置

### 新增文件

- `ts_benchmark/baselines/MindTS/deterministic_scoring.py`
  - 固定六 mask bank；
  - 绝对位置窗口 Dataset；
  - masked-only squared error；
  - overlap-add 与 coverage；
  - 61 个冻结阈值候选；
  - threshold 和 prediction 共用评分函数。
- `scripts/autoresearch/score_db_overlap_v1.py`
  - 核验 checkpoint、Scaler、模型资源、prefix cache 和运行环境；
  - 默认只做 CPU 身份检查；
  - 只有显式传入 `--execute-score` 才执行真实评分；
  - 完整评分两次并逐字节比较；
  - 禁止训练、原始 CSV 重开和自动参数调整。
- `scripts/autoresearch/extract_db_prefix_once.py`
  - 一次性生成 `[0,7056)` prefix-only cache；
  - 测试后缀数值不解析、不保存、不使用。
- `tests/test_deterministic_overlap_v1.py`
  - 覆盖确定性、输出长度、coverage、无尾部补零、统一评分函数、
    masked-only error、固定 mask bank、split 边界和配置恢复。
- `config/evaluation_contract_db_overlap_v1.json`
  - 冻结机器可读评测合同。
- `docs/deterministic_overlap_v1.md`
  - 可读协议说明、运行方式和限制。

### 修改文件

- `ts_benchmark/baselines/MindTS/models/MindTS_model.py`
  - 在显式的评测专用参数下接受 `fixed_patch_mask`；
  - 在显式的评测专用参数下使用 deterministic soft gate；
  - 默认训练和 historical 路径仍使用原随机 mask 与 hard Gumbel gate；
  - 评测专用路径若在 training mode、缺少固定 mask 或混用 contrast
    输出时会直接报错。
- `README.md`
  - 增加新协议入口、合同链接和解释警告。

分支还继承了 D1 batch8 实验的冻结 LLM 显存优化代码；该优化不是本次
评分协议的指标归因对象。评分协议相关结论只针对上述确定性评测路径。

## 3. 每个服务的完整指标

### GAIA_dbservice1

| 协议 | threshold | Affiliation-F | PR | RC | Point-F1 | AUC | AP | VUS-ROC | VUS-PR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical | 0.554506 | 0.9207 | 0.3977 | 0.6364 | 0.4895 | 0.9237 | 0.3572 | 0.9492 | 0.4337 |
| deterministic_overlap_v1 | 1.197731 | 0.9667 | 0.5972 | 0.7818 | 0.6772 | 0.9790 | 0.6116 | 0.9893 | 0.7368 |
| 差值（新−旧） | 不直接比较 | +0.0460 | +0.1995 | +0.1455 | +0.1877 | +0.0553 | +0.2544 | +0.0401 | +0.3031 |

混淆矩阵：

| 协议 | TP | FP | TN | FN |
| --- | ---: | ---: | ---: | ---: |
| Historical | 35 | 53 | 1304 | 20 |
| deterministic_overlap_v1 | 43 | 29 | 1328 | 12 |

### GAIA_dbservice2

| 协议 | threshold | Affiliation-F | PR | RC | Point-F1 | AUC | AP | VUS-ROC | VUS-PR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical | 0.468707 | 0.8805 | 0.3582 | 0.7273 | 0.4800 | 0.9666 | 0.3190 | 0.9817 | 0.4653 |
| deterministic_overlap_v1 | 0.793038 | 0.9320 | 0.5263 | 0.9091 | 0.6667 | 0.9881 | 0.5481 | 0.9914 | 0.6360 |
| 差值（新−旧） | 不直接比较 | +0.0515 | +0.1681 | +0.1818 | +0.1867 | +0.0215 | +0.2291 | +0.0097 | +0.1707 |

混淆矩阵：

| 协议 | TP | FP | TN | FN |
| --- | ---: | ---: | ---: | ---: |
| Historical | 24 | 43 | 1336 | 9 |
| deterministic_overlap_v1 | 30 | 27 | 1352 | 3 |

## 4. 两个服务等权平均

| 协议 | Affiliation-F | PR | RC | Point-F1 | AUC | AP | VUS-ROC | VUS-PR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical | 0.9006 | 0.3780 | 0.6818 | 0.4848 | 0.9452 | 0.3381 | 0.9654 | 0.4495 |
| deterministic_overlap_v1 | 0.9493 | 0.5618 | 0.8455 | 0.6719 | 0.9835 | 0.5798 | 0.9903 | 0.6864 |
| 差值（新−旧） | +0.0487 | +0.1838 | +0.1637 | +0.1872 | +0.0383 | +0.2417 | +0.0249 | +0.2369 |

注意：两套分数的数值尺度发生变化，因此 absolute threshold 本身不能跨协议
直接比较。可比较的是各协议按冻结规则选出的指标及连续分数的排序表现。

## 5. 结果解释

已验证事实：

- 两个服务的 deterministic 完整评分均连续执行两次，所有保存数组逐字节
  一致；
- 每个验证序列均为 1412 点，每个点 coverage 大于 0；
- 没有尾部补零；
- 本阶段训练更新为 0，测试集未使用；
- 两个服务在新协议下的 FP 与 FN 均少于 historical 路径。

不能据此得出的结论：

- 不能断言 dual contrastive loss 优于 legacy objective；
- 不能断言日志 v2、统计文本或模型结构带来了上表提升；
- 不能把上表差值写成重新训练后的性能增益；
- 不能把 Affiliation-F 当作 ordinary point-F1。

后续模型、损失或模态消融必须全部使用冻结的
`evaluation_contract_db_overlap_v1.json`，并优先比较 AP 与 VUS-PR。
