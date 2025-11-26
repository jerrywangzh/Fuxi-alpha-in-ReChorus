# FuXi-alpha x ReChorus 参数手册

本文档汇总了使用 `python ReChorus/src/main.py` 训练 FuXi 模型时会用到的全部命令行参数、默认值以及作用。参数按解析顺序从外到内分组，覆盖模型选择、全局控制、数据读取、训练调度以及 FuXi 专属超参。

> **使用方法**：所有参数都可以通过 CLI 传入，例如  
> `python ReChorus/src/main.py --model_name FuXi --dataset Beauty --history_max 50 --fuxi_blocks 3 ...`


## 1. 模型选择（`init_parser`）
| 参数 | 类型/默认值 | 说明 |
| --- | --- | --- |
| `--model_name` | `str`, 默认 `SASRec` | 指定要运行的模型类名。复现 FuXi 时请设为 `FuXi`。 |
| `--model_mode` | `str`, 默认 `''` | 为部分模型附加模式后缀：`'CTR'` 用于点击率任务，`'Impression'` 适用于曝光重排，空字符串表示常规 Top-K 推荐。FuXi 复现保持默认。 |

## 2. 全局训练控制（`parse_global_args`）
| 参数 | 类型/默认值 | 说明 |
| --- | --- | --- |
| `--gpu` | `str`, 默认 `'0'` | 指定 `CUDA_VISIBLE_DEVICES`，空字符串表示仅用 CPU。 |
| `--verbose` | `int`, 默认 `logging.INFO` (`20`) | Python logging 等级。 |
| `--log_file` | `str`, 默认 `''` | 训练日志完整路径。留空则自动写入 `../log/<model>/<auto_name>.txt`。 |
| `--random_seed` | `int`, 默认 `0` | 同时用于 NumPy / PyTorch 随机种子。 |
| `--load` | `int`, 默认 `0` | `>0` 时在训练前加载 `--model_path` 指向的 checkpoint。 |
| `--train` | `int`, 默认 `1` | 为 `0` 时跳过训练，仅执行评估/推理。 |
| `--save_final_results` | `int`, 默认 `1` | 是否在训练结束后导出 dev/test 的 Top-K 结果 CSV。 |
| `--regenerate` | `int`, 默认 `0` | 若置为 `1`，忽略缓存的语料 `.pkl`，强制用 Reader 重新构建。 |

## 3. 数据读取参数（`BaseReader.parse_data_args` + `SeqReader`）
| 参数 | 类型/默认值 | 说明 |
| --- | --- | --- |
| `--path` | `str`, 默认 `data/` | 数据目录，Reader 会在该路径下寻找各数据集子文件夹。 |
| `--dataset` | `str`, 默认 `Grocery_and_Gourmet_Food` | 选择数据集名称，对应 `path/dataset/{train,dev,test}.csv`。 |
| `--sep` | `str`, 默认 `'\t'` | CSV 文件分隔符。 |

`SeqReader` 无额外 CLI 参数，它在内部根据交互时间构造 `history_items/history_times/position` 供序列模型使用。

## 4. Runner / 训练调度参数（`BaseRunner.parse_runner_args`）
| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--epoch` (`int`) | `200` | 训练总轮数。 |
| `--check_epoch` (`int`) | `1` | 每隔多少轮输出 `model.check_list` 中的张量。 |
| `--test_epoch` (`int`) | `-1` | 每隔多少轮打印 test 指标；`-1` 表示训练过程中不打印。 |
| `--early_stop` (`int`) | `10` | dev 主指标连续下降多少轮后提前停止。 |
| `--lr` (`float`) | `1e-3` | 学习率。 |
| `--l2` (`float`) | `0` | `weight_decay` 权重衰减。 |
| `--batch_size` (`int`) | `256` | 训练批大小。 |
| `--eval_batch_size` (`int`) | `256` | 验证/测试批大小。 |
| `--optimizer` (`str`) | `Adam` | 备选：`SGD`, `Adam`, `Adagrad`, `Adadelta` 等。 |
| `--num_workers` (`int`) | `5` | PyTorch `DataLoader` 中并行进程数。 |
| `--pin_memory` (`int`) | `0` | `DataLoader` 的 `pin_memory` 标志。 |
| `--topk` (`str`) | `'5,10,20,50'` | 需要评估的 Top-K 列表，逗号分隔。 |
| `--metric` (`str`) | `'NDCG,HR'` | 指标集合，Runner 内会自动转为大写。 |
| `--main_metric` (`str`) | `''` | 若留空，则使用 `metric` 中的第一个指标与 `topk` 中第一个 K 组成的键（如 `NDCG@5`）做早停与保存判据。 |

## 5. 基础模型参数（`BaseModel` → `GeneralModel` → `SequentialModel`）
| 参数 | 类型/默认值 | 说明 |
| --- | --- | --- |
| `--model_path` | `str`, 默认 `''` | checkpoint 保存路径，留空时自动写入 `../model/<model>/<auto_name>.pt`。 |
| `--buffer` | `int`, 默认 `1` | 非训练阶段是否缓存 Dataset 中的样本字典，以减少多次遍历的重复构造。 |
| `--num_neg` | `int`, 默认 `1` | 训练时每个正样本随机采样的负样本数量。 |
| `--dropout` | `float`, 默认 `0` | 应用于各深层模块的 dropout 比例（FuXi 中用于位置编码等）。 |
| `--test_all` | `int`, 默认 `0` | 为 `1` 时在评估阶段对所有物品打分而不是使用预采负样本。 |
| `--history_max` | `int`, 默认 `20` | 截断用户可见历史的最大长度；`0` 表示不截断。 |

## 6. FuXi 专属参数（`models/sequential/fuxi.py`）
| 参数 | 类型/默认值 | 说明 |
| --- | --- | --- |
| `--emb_size` | `int`, 默认 `240` | 物品嵌入 / FuXi 隐层宽度，同步应用于查询与候选。 |
| `--fuxi_blocks` | `int`, 默认 `2` | AMS Transformer block 的层数。 |
| `--fuxi_heads` | `int`, 默认 `4` | 多头注意力的头数。 |
| `--fuxi_linear_dim` | `int`, 默认 `64` | 值向量（线性投影）的维度。 |
| `--fuxi_attention_dim` | `int`, 默认 `64` | 每个头的查询/键维度。 |
| `--fuxi_linear_dropout` | `float`, 默认 `0.1` | 线性投影后的 dropout。 |
| `--fuxi_attn_dropout` | `float`, 默认 `0.1` | 注意力权重的 dropout。 |
| `--fuxi_linear_activation` | `str`, 默认 `silu` | AMS 线性层的激活函数（如 `relu`, `gelu`, `silu`）。 |
| `--fuxi_ffn_multiply` | `float`, 默认 `1.0` | Feed-Forward 网络扩张系数，相当于 `FFN_hidden = multiply * emb_size`。 |
| `--fuxi_ffn_single_stage` | `int`, 默认 `0` | 是否使用单阶段 FFN（`1` 为启用）。 |
| `--fuxi_enable_rel_bias` | `int`, 默认 `1` | 控制是否注入可学习的相对时间/位置偏置。 |

## 7. 参数依赖与建议
- **路径设置**：`--path`、`--log_file`、`--model_path` 都是相对 `ReChorus/src/main.py` 运行位置解析的。推荐在仓库根目录执行命令，并使用默认的相对路径结构。
- **加载既有模型**：若要仅评估某个 `.pt` checkpoint，设置 `--load 1 --train 0`，同时把 `--model_path` 指向目标文件。


