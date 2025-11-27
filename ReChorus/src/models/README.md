# FuXi-α 编码块从输入到预测的完整数据流

本文说明 FuXi-α 模型中 **一个 FuXi Block** （AMS + MFFN） 从输入向量到输出预测的完整流程，并给出每一步张量形状、线性 / 非线性运算公式。

全流程拆成两大部分：

* **AMS（Adaptive Multi-channel Self-attention，多通道自注意力）**
* **MFFN（Multi-Stage Feed-Forward Network，多阶段前馈网络）**

> 输入序列通过一个fuxiblock，可以视作一下过程
> 输入序列表示 `x` ──> AMS（多通道注意力）──> MFFN（两阶段前馈）──> 新的序列表示 `x'`
> 多个 FuXi Block 堆叠，再接一个 MLP / 点积层，就是最终的推荐预测。

---

## 0. 统一符号与张量形状

假设：

* `B`：batch size（一次喂入的用户序列条数）
* `N`：每个用户序列在当前 batch 中的最大长度（padding 到统一长度）
* `D`：embedding 维度（`embedding_dim`）
* `H`：多头注意力的 head 数（`num_heads`）
* `D_attn`：注意力中每个 head 的 q/k 维度（`attention_dim`）
* `D_lin`：每个 head 的 v 维度（`linear_dim`）

典型形状：

* 交互序列输入（从 embedding 层来）
  `x` : `(B, N, D)`

* 时间戳：
  `timestamps` : `(B, N)`

在 FuXi 中，为了效率，经常用 **Jagged Tensor**（不等长拼接）形式：

* `x` 也可以被展平成
  `x_flat` : `(sum_i N_i, D)`
  其中 `sum_i N_i` 是当前 batch 中所有序列长度之和。
* `x_offsets` : `(B + 1,)`，记录每个序列在 `x_flat` 里的起止位置。

下面讲 AMS 和 MFFN 时，会交替用 `(B, N, ·)`（dense）与 `(sum_i N_i, ·)`（jagged）两种视角。

---

## 1. AMS：多通道自注意力模块

### 1.1 输入与输出（AMS 这块）

AMS 模块所在的 `FuXiBlockJagged.forward` 签名：

```python
def forward(
    self,
    x: Tensor,              # (sum_i N_i, D)
    x_offsets: Tensor,      # (B + 1,)
    all_timestamps: Tensor, # (B, N)   时间戳
    invalid_attn_mask: Tensor,  # (B, N, N)   合法位置 = 1, 非法位置(未来/填充) = 0
    ...
) -> new_outputs: Tensor    # (sum_i N_i, D)
```

在进入 AMS 计算前，`x` 会先做 RMSNorm 和一个大矩阵投影 `_uvqk`。

---

### 1.2 归一化输入：`_norm_input`

```python
normed_x = F.rms_norm(x, normalized_shape=[D], eps=eps)
# 形状不变： (sum_i N_i, D)
```

每一行向量 `x_i` 做 **RMSNorm**，公式：

```text
r_i = sqrt( mean_j( x_i[j]^2 ) + eps )
normed_x_i[j] = x_i[j] / r_i
```

* 这里 `i` 走的是所有 token（总共 `sum_i N_i` 个），每个 token 向量长度是 `D`。
* RMSNorm 是对每个 token 自己的一维向量做归一化，不会跨样本、跨时间混。

---

### 1.3 一次大矩阵乘：得到 u/v/q/k

```python
batched_mm_output = torch.mm(normed_x, self._uvqk)
# normed_x:       (sumN, D)
# _uvqk(weight):  (D, 4 * H * D_lin + 2 * H * D_attn)
# batched_mm_output: (sumN, 4*H*D_lin + 2*H*D_attn)
```

* 这里 `_uvqk` 是**一个共享的线性变换矩阵**，同时产生：

  * u：AMS 内的门控向量（后面与注意力输出逐点相乘）
  * v：值向量 value
  * q：查询向量 query
  * k：键向量 key

激活：

```python
if self._linear_activation == "silu":
    batched_mm_output = F.silu(batched_mm_output)
```

SiLU 逐元素作用：

```text
silu(z) = z * sigmoid(z)
```

接着按照维度拆开：

```python
u, v, q, k = torch.split(
    batched_mm_output,
    [
        3 * H * D_lin,   # u: 三个通道各 H 个 head，每个 head D_lin 维
        1 * H * D_lin,   # v: H * D_lin
        1 * H * D_attn,  # q: H * D_attn
        1 * H * D_attn,  # k: H * D_attn
    ],
    dim=1,
)
# u: (sumN, 3*H*D_lin)
# v: (sumN,   H*D_lin)
# q: (sumN,   H*D_attn)
# k: (sumN,   H*D_attn)
```

**直观理解：**

* 每个 token 的原始 embedding `x_i` 被映射到长向量
  `[u_i, v_i, q_i, k_i]`
* 其中：

  * `q_i, k_i` 用于**语义通道**的注意力打分
  * `v_i` 用于**三个通道**的值
  * `u_i` 则是对三通道注意力输出的**门控权重**

---

### 1.4 jagged 转 dense：构造 `(B, N, H, D_attn)` 和 `(B, N, H, D_lin)`

注意：`q, k, v` 当前是 `(sumN, ·)`，要变成 batch 结构：

```python
# 先从 jagged -> dense，得到 (B, N, H*D_attn)
padded_q = jagged_to_padded_dense(q, x_offsets, max_len=n)  # (B, N, H*D_attn)
padded_k = jagged_to_padded_dense(k, x_offsets, max_len=n)  # (B, N, H*D_attn)
padded_v = jagged_to_padded_dense(v, x_offsets, max_len=n)  # (B, N, H*D_lin)

# 再 reshape 成多头形式
padded_q = padded_q.view(B, N, H, D_attn)  # (B, N, H, D_attn)
padded_k = padded_k.view(B, N, H, D_attn)
padded_v = padded_v.view(B, N, H, D_lin)   # (B, N, H, D_lin)
```

这样，一个 FuXi Block 的 AMS 部分就有了标准多头注意力常用的 `q, k, v` 形状。

---

### 1.5 语义通道注意力：`qk_attn`

代码（简化后）：

```python
qk_attn = torch.einsum(
    "bnhd,bmhd->bhnm",
    padded_q,  # (B,N,H,D_attn)
    padded_k,  # (B,N,H,D_attn)
)
# qk_attn: (B, H, N, N)

qk_attn = F.silu(qk_attn) / n    # n = N
qk_attn = qk_attn * invalid_attn_mask.unsqueeze(0).unsqueeze(0)
```

逐步解释：

1. `einsum("bnhd,bmhd->bhnm", q, k)` 计算的是对每个 head 的 `Q * K^T`：

   * 固定 `b`（batch）和 `h`（head），对 `n` 和 `m` 做：

     ```text
     qk_attn[b, h, n, m] = sum_{d=0..D_attn-1} padded_q[b, n, h, d] * padded_k[b, m, h, d]
     ```
   * 得到形状 `(B, H, N, N)`：第一个 N 是 query 位置，第二个 N 是 key 位置。

2. `F.silu(qk_attn) / n`：

   * 用 SiLU 代替 Softmax，作为注意力权重的非线性映射。
   * `/ n` 是一个缩放因子，防止序列过长时数值过大。

3. `* invalid_attn_mask`：

   * `invalid_attn_mask` 形状 `(B, N, N)`，值为 1 或 0。
   * `unsqueeze(0).unsqueeze(0)` 之后变为 `(1, 1, B, N, N)`，广播到 `(B, H, N, N)`。
   * 非法位置（未来、padding）被乘成 0，权重清零。

得到的 `qk_attn` 就是**语义通道**的注意力权重。

---

### 1.6 时间 / 位置通道注意力：`pos_attn` 与 `ts_attn`

通过 `SeparatedRelativeBucketedTimeAndPositionBasedBias` 计算：

```python
pos_attn, ts_attn = rel_attn_bias(all_timestamps)
# pos_attn: (1, N, N)  只依赖位置，相同 batch 共享
# ts_attn:  (B, N, N)  依赖时间戳（batch 不同）

# 掩码
pos_attn = pos_attn * invalid_attn_mask.unsqueeze(0)  # (1, N, N)
ts_attn  = ts_attn  * invalid_attn_mask               # (B, N, N)
```

这里的相对偏置逻辑：

* 给定时间戳矩阵 `t[b, i]`，构造时间差 `Δt_ij = t[b, j] - t[b, i]`
* 通过分桶函数 `bucketization_fn(Δt)` 映射到离散桶号 `k`
* 查表 `ts_w[k]` 得到一个标量，再经过 SiLU 变成时间通道权重 `A_time[i,j]`
* 对位置偏置相似，只是差值是 `Δp = j - i`（相对位置）

从形状上看：

* `pos_attn` 可认为是 `(B, N, N)`，只是本质上 batch 维度是复制的（因为所有序列长度一样时位置结构一样）。
* `ts_attn` 每个 batch 都不同，因为时间戳不同。

---

### 1.7 三通道输出：`output_pos` / `output_ts` / `output_latent`

利用 `v`（值向量）和三种权重，做三次加权求和。

1. **位置通道**：

```python
output_pos = torch.einsum(
    "bnm,bmhd->bnhd",
    pos_attn,   # (B,N,N) 广播或复制 pos_attn
    padded_v,   # (B,N,H,D_lin)
)
# output_pos: (B, N, H, D_lin)
```

解释：固定 `b, n, h`，对所有 `m` 求和：

```text
output_pos[b, n, h, d] = sum_{m=0..N-1} pos_attn[b, n, m] * padded_v[b, m, h, d]
```

2. **时间通道**：

```python
output_ts = torch.einsum(
    "bnm,bmhd->bnhd",
    ts_attn,    # (B,N,N)
    padded_v,   # (B,N,H,D_lin)
)
# output_ts: (B, N, H, D_lin)
```

3. **语义通道（latent 通道）**：

```python
output_latent = torch.einsum(
    "bhnm,bmhd->bnhd",
    qk_attn,    # (B,H,N,N)
    padded_v,   # (B,N,H,D_lin)
)
# output_latent: (B, N, H, D_lin)
```

注意这里 `einsum` 的下标稍有不同（`bhnm` vs `bnm`）。

---

### 1.8 三通道拼接 + 回到 Jagged：AMS 总输出

三通道在 head 和通道维上拼接：

```python
combined_output = torch.concat(
    [output_pos, output_ts, output_latent],  # 每个都是 (B,N,H,D_lin)
    dim=-1,                                  # 在 D_lin 这个维度拼接
)
# combined_output: (B, N, H, 3*D_lin)

combined_output = combined_output.reshape(
    B, N, H * 3 * D_lin
)
# 形状: (B, N, 3*H*D_lin)
```

然后从 dense 转回 jagged：

```python
attn_output = dense_to_jagged(
    combined_output, [x_offsets]
)[0]
# attn_output: (sumN, 3*H*D_lin)
```

这个 `attn_output` 就是 AMS 模块的最终输出，后面进入 MFFN 门控和前馈。

---

## 2. MFFN：多阶段前馈网络

MFFN 在代码里对应 `MultistageFeedforwardNeuralNetwork`，在一个 FuXi Block 里被这样调用：

```python
ams_output = u * self._norm_attn_output(attn_output)
# u:           (sumN, 3*H*D_lin)
# attn_output: (sumN, 3*H*D_lin) 先 RMSNorm 再逐元素乘
# ams_output:  (sumN, 3*H*D_lin)  —— 这是 MFFN 的输入 X

new_outputs = self._mffn(ams_output, x)
# x:           (sumN, D) 原始输入
# new_outputs: (sumN, D) 块输出
```

下面分 Stage 1 和 Stage 2 详细分析。

---

### 2.1 Stage 1：线性融合多通道 + 残差

MFFN 的 `forward`：

```python
def forward(self, X, X0):
    X = (
        self.lin0(
            F.dropout(
                X,
                p = dropout_ratio,
                training = self.training,
            )
        ) + X0
    )
    ...
    return X
```

参数与形状：

* `X`：AMS 总输出（已经经过门控）
  形状 `(sumN, 3*H*D_lin)`，称为 `z`。
* `X0`：块的原始输入（也就是 `x`）
  形状 `(sumN, D)`。
* `lin0`：形状 `(3*H*D_lin, D)` 的线性层。

逐步：

1. Dropout：

   ```python
   X_drop = dropout(X)         # (sumN, 3*H*D_lin)
   ```

2. 线性映射：

   ```python
   fused = X_drop @ W1 + b1    # (sumN, D)
   # W1 = lin0.weight^T, 形状 (3*H*D_lin, D)
   ```

3. 加残差：

   ```python
   X = fused + X0              # (sumN, D)
   ```

数学形式可以写成：

```text
h1 = X0 + X_drop @ W1 + b1     # 每个 token 一行
```

这里 `h1` 就是 Stage 1 的输出，也就是代码里的 `X`（被覆盖）。

**作用理解：**

* 把来自三通道的 AMS 输出（3 * H * D_lin 维）压缩回主流的 embedding 维 `D`。
* 同时加上原始输入 `X0` 的残差，保证低阶特征一直保留在信息流中。

---

### 2.2 Stage 2：RMSNorm + SwiGLU + 第二次残差

当 `single_stage == False` 时，还会执行第二阶段：

```python
normed_X = F.rms_norm(X, normalized_shape=[D], eps=eps)  # (sumN, D)
normed_X = F.dropout(normed_X, p=dropout_ratio, training=self.training)

X1 = F.silu(self.lin1(normed_X)) * self.lin3(normed_X)
X  = self.lin2(X1) + X
```

参数形状：

* `lin1`：`(D, hidden_size)`，输出 `(sumN, hidden_size)`
* `lin3`：`(D, hidden_size)`，输出 `(sumN, hidden_size)`
* `lin2`：`(hidden_size, D)`，输出 `(sumN, D)`
* `hidden_size` 一般设为 `ffn_multiply * D`（例如 4D）

逐步展开：

1. 第二次 RMSNorm（对 Stage 1 输出）：

   ```python
   normed_X = RMSNorm(X)       # (sumN, D)
   ```

2. Dropout：

   ```python
   normed_X_drop = dropout(normed_X)   # (sumN, D)
   ```

3. SwiGLU 风格门控（两路线性 + 元素乘）：

   ```python
   a = lin1(normed_X_drop)     # (sumN, hidden_size)
   b = lin3(normed_X_drop)     # (sumN, hidden_size)

   gate = silu(a)              # (sumN, hidden_size)
   X1   = gate * b             # 逐元素乘, (sumN, hidden_size)
   ```

   也可以理解为：

   ```text
   X1 = silu(normed_X_drop @ W3 + b3)  ⊙  (normed_X_drop @ W4 + b4)
   ```

   其中 `⊙` 表示逐元素乘法。

4. 降回 D 维 + 第二条残差：

   ```python
   update = lin2(X1)           # (sumN, D)
   X      = update + X         # (sumN, D)
   ```

数学上等价于：

```text
h2 = h1 + ( silu(h1_norm @ W3 + b3) ⊙ (h1_norm @ W4 + b4) ) @ W5 + b5
```

* `h1`：Stage 1 输出
* `h1_norm`：RMSNorm(h1)
* `(·) @ W5 + b5` 对应 `lin2`

**直观理解：**

* 先把 Stage 1 输出归一化，再映射到一个更高维的中间空间（`hidden_size`）。
* 用 SwiGLU 的门控结构让网络可以学习“哪些特征组合应该被放大”，哪些应被抑制。
* 再降回原来的维度 `D`，并加残差，让梯度可以绕过高维非线性直接流动。

---

### 2.3 MFFN 整体的输入 / 输出总结

* 输入：

  * `X`：AMS 输出的融合向量 `(sumN, 3*H*D_lin)`
  * `X0`：块的原始输入 `(sumN, D)`

* 过程：

  * Stage 1：`X` 线性压缩 + 残差 `X0`
  * Stage 2：RMSNorm + SwiGLU + 残差

* 输出：

  * `new_outputs`：块的最终输出 `(sumN, D)`
    这就是下一层 FuXi Block 的输入，或者最后一层后拿来做预测。

---

## 3. 从多个 FuXi Block 到最终预测

在完整 FuXi-α 模型里：

1. **Embedding 层：**

   ```text
   user 历史序列 -> item_id -> embedding lookup
   得到 x0: (B, N, D)
   ```

2. **FuXiJagged 堆叠多层 FuXiBlockJagged：**

   ```python
   x, _ = self._fuxi(
       x=x0,                # (B,N,D) 或 jagged
       x_offsets=x_offsets,
       all_timestamps=timestamps,
       invalid_attn_mask=1 - causal_mask,
       ...
   )
   # x: (B, N, D)   编码后的序列表示
   ```

3. **取每个用户最后一个位置的表示**作为用户当前状态 `h_user`：

   ```text
   h_user: (B, D)
   ```

4. **与候选 item embedding 做点积**：

   * 候选 item embedding：`E_items`，形状 `(num_items, D)`
   * 得分：

     ```text
     logits = h_user @ E_items^T    # (B, num_items)
     ```

   在 ReChorus 封装的 `FuXiSeq` 里，是对 `feed_dict["item_id"]` 的 embedding 做批量点积，形状 `(B, 1 + num_neg)`。

5. **损失 / 预测：**

   * 训练时：用负采样 softmax 或 BPR loss 等，`Adam` 更新所有参数（embedding、_uvqk、MFFN、偏置等）。
   * 测试时：对 `logits` 取 top-K 就是推荐结果。

---

如果你愿意，我可以在这个 `.md` 基础上，再帮你加上：

* 某个具体配置（比如 `H=4, D=64, D_lin=32`）下的**具体数值例子**；
* 把关键几步（AMS 的三次 `einsum` + MFFN 两阶段）画成**小表格**，专门列出“输入形状 -> 运算 -> 输出形状”，方便你在论文里直接截图 / 改写。
