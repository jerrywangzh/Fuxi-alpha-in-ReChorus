# FuXi-α：从输入向量到输出预测的完整计算流程（Markdown 说明）

> 本文假定你已经有了 FuXi-α 的 PyTorch 代码（`FuXiBlockJagged`, `MultistageFeedforwardNeuralNetwork` 等），目标是**从数学和张量形状**角度，解释一次前向传播从输入到输出预测的全过程，并把一个 FuXi Block 拆成两大部分：
>
> * **AMS（Adaptive Multi-channel Self-Attention，自适应多通道自注意力）**
> * **MFFN（Multi-stage Feed-Forward Network，多阶段前馈网络）**

---

## 0. 记号与张量形状约定

为了方便描述，我们先把论文和代码里的主要维度统一：

* 批大小：\(B\)（batch size）
* 每个序列截断后的最大长度：\(N\)（`max_seq_len`）
* 嵌入维度：\(D\)（`embedding_dim`）
* 注意力头数：\(H\)（`num_heads`）
* 每个注意力头的 Query/Key 维度：\(d_q\)（`attention_dim`）
* 每个注意力头的 Value 维度：\(d_v\)（`linear_hidden_dim`）
* MFFN 隐藏层维度：\(D_{\text{ffn}}\)（代码中 `hidden_size = ffn_multiply * D`）

### 0.1 输入

对第 \(l\) 层 FuXi Block 而言，输入是上一层的输出：

* 稠密表示（概念上）：
  [
  \mathbf{X}^{(l-1)} \in \mathbb{R}^{B \times N \times D}
  \]
  第 \(b\) 个用户序列第 \(n\) 个位置向量记为
  \(\mathbf{x}^{(l-1)}_{b,n} \in \mathbb{R}^{D}\)。

* 代码里为了节省显存和计算，使用 **jagged 表示**：把所有非 padding 的位置在 batch 内拼成一个长向量：

  * `x` 形状：\((\sum_i N_i) \times D\)
  * `x_offsets` 形状：\((B+1,)\)，告诉每个 user 对应在这条长向量中的起止下标。
    这是 FBGEMM 的 jagged tensor 表示法。

后面讲公式时，为了直观，会主要使用稠密形状 \(B \times N \times D\)，再顺带说明对应的代码张量形状。

---

## 1. AMS 模块：自适应多通道自注意力

AMS 的作用：
**把语义、位置、时间三种信息分成三个通道分别建模，然后再融合。**

AMS 的输入与输出（在本层）：

* 输入：

  * 上一层隐状态：\(\mathbf{X}^{(l-1)} \in \mathbb{R}^{B \times N \times D}\)
  * 时间戳矩阵：\(\mathbf{T} \in \mathbb{R}^{B \times N}\)
  * 注意力 mask：\(\mathbf{M} \in {0,1}^{B \times N \times N}\)，其中 \(M_{b,i,j}=0\) 表示不允许关注（例如未来位置、padding）。
* 输出（AMS 的结果传给 MFFN）：

  * 多通道输出：\(\mathbf{Z} \in \mathbb{R}^{B \times N \times 3 H d_v}\)

### 1.1 第一步：对输入做 RMSNorm

代码对应：`normed_x = F.rms_norm(x, normalized_shape=[D], eps=ε)`

对每个位置向量 \(\mathbf{x}^{(l-1)}_{b,n} \in \mathbb{R}^{D}\)，RMSNorm 定义为（逐位置独立）

[
\text{RMSNorm}(\mathbf{x})
= \frac{\mathbf{x}}{\sqrt{\frac{1}{D}\sum_{d=1}^{D} x_d^2 + \varepsilon}}
\in \mathbb{R}^{D}.
]

得到
[
\tilde{\mathbf{X}}^{(l-1)} = \text{RMSNorm}\big(\mathbf{X}^{(l-1)}\big)
\in \mathbb{R}^{B \times N \times D}.
]

代码里这是在 **展平后的形状** \((\sum_i N_i) \times D\) 上做的，但数学上是等价的。

---

### 1.2 第二步：一次大线性变换同时得到 \(u,v,q,k\)

代码：

```python
batched_mm_output = torch.mm(normed_x, self._uvqk)  # (∑N_i, D) @ (D, P)
u, v, q, k = torch.split(
    batched_mm_output,
    [3 * H * d_v, H * d_v, H * d_q, H * d_q],
    dim=1,
)
```

这里

* \(W_{\text{uvqk}} \in \mathbb{R}^{D \times (3 H d_v + H d_v + 2 H d_q)}\) 是一个大矩阵，按列拆成

  * \(W_u \in \mathbb{R}^{D \times 3 H d_v}\)
  * \(W_v \in \mathbb{R}^{D \times H d_v}\)
  * \(W_q \in \mathbb{R}^{D \times H d_q}\)
  * \(W_k \in \mathbb{R}^{D \times H d_q}\)

对每个位置：

[
\begin{aligned}
\mathbf{u}*{b,n} &= \phi\big(\tilde{\mathbf{x}}^{(l-1)}*{b,n} W_u\big)
\in \mathbb{R}^{3 H d_v},\
\mathbf{v}*{b,n} &= \phi\big(\tilde{\mathbf{x}}^{(l-1)}*{b,n} W_v\big)
\in \mathbb{R}^{H d_v},\
\mathbf{q}*{b,n} &= \phi\big(\tilde{\mathbf{x}}^{(l-1)}*{b,n} W_q\big)
\in \mathbb{R}^{H d_q},\
\mathbf{k}*{b,n} &= \phi\big(\tilde{\mathbf{x}}^{(l-1)}*{b,n} W_k\big)
\in \mathbb{R}^{H d_q},
\end{aligned}
]

其中 \(\phi(\cdot)=\text{SiLU}\) 是非线性激活（代码里 `F.silu`），可以省略或开启。

把它们重新 reshape 成多头形式：

* \(U \in \mathbb{R}^{B \times N \times H \times 3d_v}\)
* \(V \in \mathbb{R}^{B \times N \times H \times d_v}\)
* \(Q \in \mathbb{R}^{B \times N \times H \times d_q}\)
* \(K \in \mathbb{R}^{B \times N \times H \times d_q}\)

代码中，`q,k,v` 先是 jagged 形式，再通过 `jagged_to_padded_dense` 转成稠密的 `(B,N,H,d)`。

---

### 1.3 第三步：语义通道注意力（QK 通道）

代码（核心部分）：

```python
qk_attn = torch.einsum(
    "bnhd,bmhd->bhnm",
    padded_q.view(B, n, H, d_q),
    padded_k.view(B, n, H, d_q),
)
qk_attn = F.silu(qk_attn) / n
qk_attn = qk_attn * invalid_attn_mask.unsqueeze(0).unsqueeze(0)
```

数学上：

1. **点积得到注意力打分：**

   [
   S^{(\text{sem})}_{b,h,n,m}
   ==========================

   # \langle \mathbf{q}*{b,n,h,:}, \mathbf{k}*{b,m,h,:}\rangle

   \sum_{d=1}^{d_q} Q_{b,n,h,d} , K_{b,m,h,d}
   \in \mathbb{R}.
   ]

   这里 \(S^{(\text{sem})} \in \mathbb{R}^{B \times H \times N \times N}\)。

2. **SiLU 非线性代替 softmax：**

   [
   A^{(\text{sem})}_{b,h,n,m}
   ==========================

   \frac{\text{SiLU}\big(S^{(\text{sem})}_{b,h,n,m}\big)}{N}.
   ]

   SiLU：\(\text{SiLU}(x) = x \cdot \sigma(x)\)，\(\sigma\) 是 Sigmoid。论文指出，在序列推荐中用 SiLU 替代 softmax 的注意力效果更好。

3. **应用因果 & padding mask：**

   * `invalid_attn_mask[b,n,m]=0` ⇒ 不允许关注（未来位置或 padding）。
   * 乘上 mask：

   [
   \tilde{A}^{(\text{sem})}*{b,h,n,m}
   = A^{(\text{sem})}*{b,h,n,m} \cdot M_{b,n,m}.
   ]

最终得到语义通道权重
\(\tilde{A}^{(\text{sem})} \in \mathbb{R}^{B \times H \times N \times N}\)。

---

### 1.4 第四步：时间 & 位置通道的注意力权重

代码（核心思想）：

```python
pos_attn, ts_attn = rel_attn_bias(all_timestamps)
pos_attn = pos_attn * invalid_attn_mask.unsqueeze(0)  # (B,N,N)
ts_attn  = ts_attn  * invalid_attn_mask.unsqueeze(0)
```

`SeperatedRelativeBucketedTimeAndPositionBasedBias` 模块根据 **相对位置** 和 **时间差** 生成这两个矩阵：

* 位置偏置 \(\mathbf{P} \in \mathbb{R}^{B \times N \times N}\)
* 时间偏置 \(\mathbf{B} \in \mathbb{R}^{B \times N \times N}\)

其中
\(P_{b,n,m} = p_{\Delta p}\)，\(\Delta p = n-m\) 经过分桶后的索引；
\(B_{b,n,m} = b_{\Delta t}\)，\(\Delta t = T_{b,n} - T_{b,m}\) 同样按对数分桶后的索引。

在实现上，它们经过 SiLU 激活后就可以被看作 **时间/位置通道的注意力权重**：

[
\begin{aligned}
A^{(\text{pos})}*{b,n,m} &= \text{SiLU}\big(P*{b,n,m}\big),\
A^{(\text{time})}*{b,n,m} &= \text{SiLU}\big(B*{b,n,m}\big).
\end{aligned}
]

代码中已经包含这个非线性；最终参与后续 `einsum` 时，矩阵大小为 \(B \times N \times N\)，再和 \(V\) 做矩阵乘法时自动广播到对应的 head 维度。

同样乘以 mask 保证因果性和 padding 不被关注。

---

### 1.5 第五步：三通道分别加权求和 Value

代码三次 `einsum`：

```python
output_pos = torch.einsum("bnm,bmhd->bnhd", pos_attn, padded_v)
output_ts  = torch.einsum("bnm,bmhd->bnhd", ts_attn,  padded_v)
output_latent = torch.einsum("bhnm,bmhd->bnhd", qk_attn, padded_v)
```

解释：

1. **位置通道输出：**

   [
   \mathbf{H}^{(\text{pos})}*{b,n,h,:}
   = \sum*{m=1}^{N} A^{(\text{pos})}*{b,n,m} \cdot \mathbf{v}*{b,m,h,:}
   \in \mathbb{R}^{d_v},
   ]
   因为 \(A^{(\text{pos})}\) 无 head 维度，所以对每个 head 共享。

2. **时间通道输出：**

   [
   \mathbf{H}^{(\text{time})}*{b,n,h,:}
   = \sum*{m=1}^{N} A^{(\text{time})}*{b,n,m} \cdot \mathbf{v}*{b,m,h,:}
   \in \mathbb{R}^{d_v}.
   ]

3. **语义通道输出：**

   [
   \mathbf{H}^{(\text{sem})}*{b,n,h,:}
   = \sum*{m=1}^{N} \tilde{A}^{(\text{sem})}*{b,h,n,m} \cdot \mathbf{v}*{b,m,h,:}
   \in \mathbb{R}^{d_v}.
   ]

组合三个输出：

* \(\mathbf{H}^{(\text{pos})}, \mathbf{H}^{(\text{time})}, \mathbf{H}^{(\text{sem})} \in \mathbb{R}^{B \times N \times H \times d_v}\)

---

### 1.6 第六步：三通道拼接 + RMSNorm + 显式二阶交互

代码：

```python
combined_output = torch.concat(
    [output_pos, output_ts, output_latent], dim=-1
)  # (B,N,H,3d_v)

combined_output = combined_output.reshape(B, N, H * 3 * d_v)  # (B,N,3H d_v)
attn_output = dense_to_jagged(combined_output, [x_offsets])[0]  # (∑N_i, 3H d_v)

ams_output = u * self._norm_attn_output(attn_output)
# _norm_attn_output = RMSNorm over last dim (3H d_v)
```

数学上：

1. **拼接：**

   [
   \mathbf{h}^{\text{AMS}}*{b,n,h}
   = [\mathbf{H}^{(\text{pos})}*{b,n,h,:};;
   \mathbf{H}^{(\text{time})}*{b,n,h,:};;
   \mathbf{H}^{(\text{sem})}*{b,n,h,:}]
   \in \mathbb{R}^{3d_v}.
   ]

   拼完再把 head 维和通道维并在一起：

   [
   \mathbf{z}_{b,n}
   \in \mathbb{R}^{3 H d_v}.
   ]

2. **RMSNorm：**

   [
   \hat{\mathbf{z}}*{b,n}
   = \text{RMSNorm}(\mathbf{z}*{b,n})
   \in \mathbb{R}^{3 H d_v}.
   ]

3. **与 \(u\) 做逐元素乘积（显式二阶交互）：**

   * 前面 1.2 中的 \(\mathbf{u}_{b,n} \in \mathbb{R}^{3H d_v}\) 来自输入的线性变换。
   * AMS 最终输出：

   [
   \mathbf{z}'*{b,n}
   = \mathbf{u}*{b,n} \odot \hat{\mathbf{z}}_{b,n}
   \in \mathbb{R}^{3 H d_v}.
   ]

   其中 \(\odot\) 是逐元素乘积，这一步相当于把「来自输入的门控向量 \(\mathbf{u}\)」和「来自 AMS 的多通道注意力结果 \(\hat{\mathbf{z}}\)」做显式二阶交互。

将所有位置展平后，在代码中对应 `ams_output`：

[
\text{AMS 输出 } \mathbf{Z} \in \mathbb{R}^{B \times N \times 3 H d_v}
\quad\leftrightarrow\quad
\text{ams_output} \in \mathbb{R}^{(\sum_i N_i) \times 3 H d_v}.
]

---

## 2. MFFN 模块：多阶段前馈网络

MFFN 接收 AMS 的输出 \(\mathbf{Z}\) 以及残差输入 \(\mathbf{X}^{(l-1)}\)，输出本层最终结果 \(\mathbf{X}^{(l)}\)。

### 2.1 Stage 1：多通道融合 + 残差

代码（`MultistageFeedforwardNeuralNetwork.forward`）：

```python
# X: ams_output,  shape (∑N_i, 3H d_v)
# X0: 原始输入 x, shape (∑N_i, D)

X = self.lin0( dropout(X) ) + X0
# lin0: (3H d_v) -> D
```

数学上，先恢复成 \(B \times N\) 的形状来理解：

1. **线性融合多通道输出：**

   第一阶段的投影矩阵 \(W_1 \in \mathbb{R}^{3 H d_v \times D}\)，偏置 \(\mathbf{b}_1 \in \mathbb{R}^{D}\)。对每个位置：

   [
   \mathbf{g}*{b,n}
   = \mathbf{z}'*{b,n} W_1 + \mathbf{b}_1
   \in \mathbb{R}^{D}.
   ]

2. **加上输入残差：**

   MFFN 第一阶段输出：

   [
   \mathbf{h}^{(1)}*{b,n}
   = \mathbf{x}^{(l-1)}*{b,n} + \mathbf{g}_{b,n}
   \in \mathbb{R}^{D}.
   ]

这一步与论文公式 (5) 完全对应：Stage 1 把多通道 AMS 输出融合回原始嵌入空间，同时保留原始输入。

代码中的 `X` 在这一步之后就表示 \(\mathbf{h}^{(1)}\)。

---

### 2.2 Stage 2：RMSNorm + SwiGLU + 残差

代码：

```python
normed_X = F.rms_norm(X, normalized_shape=[D], eps=self.eps)
normed_X = F.dropout(normed_X, p=self.dropout_ratio, training=self.training)

X1 = F.silu(self.lin1(normed_X)) * self.lin3(normed_X)
X  = self.lin2(X1) + X
```

其中

* `lin1: ℝ^D → ℝ^{D_ffn}`
* `lin3: ℝ^D → ℝ^{D_ffn}`
* `lin2: ℝ^{D_ffn} → ℝ^D`
* \(D_{\text{ffn}} = \text{hidden_size} = \text{ffn_multiply} \cdot D\)，通常取 \(4D\)，类似 LLaMA/Transformer 的 MLP。

数学上：

1. **RMSNorm：**

   [
   \tilde{\mathbf{h}}^{(1)}*{b,n}
   = \text{RMSNorm}\big(\mathbf{h}^{(1)}*{b,n}\big)
   \in \mathbb{R}^{D}.
   ]

2. **两条线性支路（SwiGLU 结构）：**

   设 \(W_2, W_3 \in \mathbb{R}^{D \times D_{\text{ffn}}}\)，\(\mathbf{b}_2,\mathbf{b}*3 \in \mathbb{R}^{D*{\text{ffn}}}\)，则

   [
   \begin{aligned}
   \mathbf{a}*{b,n} &= \tilde{\mathbf{h}}^{(1)}*{b,n} W_2 + \mathbf{b}*2
   \in \mathbb{R}^{D*{\text{ffn}}},\
   \mathbf{b}*{b,n} &= \tilde{\mathbf{h}}^{(1)}*{b,n} W_3 + \mathbf{b}*3
   \in \mathbb{R}^{D*{\text{ffn}}}.
   \end{aligned}
   ]

   然后做 SwiGLU 门控：

   [
   \mathbf{s}*{b,n}
   = \text{SiLU}(\mathbf{a}*{b,n}) \odot \mathbf{b}*{b,n}
   \in \mathbb{R}^{D*{\text{ffn}}}.
   ]

3. **投影回 \(D\) 维 + 残差：**

   设 \(W_4 \in \mathbb{R}^{D_{\text{ffn}} \times D}\)，偏置 \(\mathbf{b}_4\)：

   [
   \begin{aligned}
   \mathbf{h}^{(2)}*{b,n}
   &= \mathbf{s}*{b,n} W_4 + \mathbf{b}*4
   \in \mathbb{R}^{D},[3pt]
   \mathbf{x}^{(l)}*{b,n}
   &= \mathbf{h}^{(1)}*{b,n} + \mathbf{h}^{(2)}*{b,n}
   \in \mathbb{R}^{D}.
   \end{aligned}
   ]

这正是论文公式 (6)、(7) 中的二阶段 MFFN：先显式融合 AMS + 输入（Stage1），再用 SwiGLU 做隐式高阶特征交互（Stage2），最后再残差回原空间。

**最终：**

* 整个 FuXi Block 的输出：
  \(\mathbf{X}^{(l)} \in \mathbb{R}^{B \times N \times D}\)
* 代码中表现为
  `new_outputs` 形状：\((\sum_i N_i) \times D\)，再经过 `jagged_to_padded_dense` 还原到 \(B \times N \times D\)。

---

## 3. 多层 FuXi Block 堆叠与输出预测层

### 3.1 堆叠多层 FuXi Block

论文中，一个完整的 FuXi-α 模型由 \(L\) 层 FuXi Block 堆叠而成：

[
\begin{aligned}
\mathbf{X}^{(0)} &= \text{EmbeddingLayer}(\text{item IDs}),\
\mathbf{X}^{(1)} &= \text{FuXiBlock}^{(1)}(\mathbf{X}^{(0)}),\
&\dots\
\mathbf{X}^{(L)} &= \text{FuXiBlock}^{(L)}(\mathbf{X}^{(L-1)}).
\end{aligned}
]

每一层的输入输出都是 \(B \times N \times D\) 形状，只是语义含义越来越“高阶”：

* 低层：主要捕获局部邻近 item 之间的短期依赖、简单的时间/位置模式。
* 高层：叠加多次显式（AMS）+ 隐式（MFFN）特征交互，能表达任意阶的 item 组合和复杂时序模式。

### 3.2 输出预测层

当得到顶层输出 \(\mathbf{X}^{(L)}\) 后，以序列最后一个非 padding 位置的向量作为“当前用户状态” \(\mathbf{h}_b \in \mathbb{R}^{D}\)：

[
\mathbf{h}*b = \mathbf{X}^{(L)}*{b,,t_b},\quad
t_b = \text{该用户序列最后一个真实交互的位置索引}.
]

设 item embedding 矩阵为 \(E \in \mathbb{R}^{|\mathcal{I}| \times D}\)（训练时与输入 embedding 共享参数，即 **权重共享**），则对所有候选 item 的打分向量为：

[
\mathbf{o}_b
= \mathbf{h}_b E^\top
\in \mathbb{R}^{|\mathcal{I}|}.
]

* 第 \(j\) 维的分数 \(o_{b,j}\) 就是对 item \(j\) 的偏好。
* 再做 softmax 得到概率分布：

[
p(i = j \mid \text{history of user }b)
= \frac{\exp(o_{b,j})}
{\sum_{k=1}^{|\mathcal{I}|} \exp(o_{b,k})}.
]

训练时采用 **sampled softmax**：只对一个正样本和少量负样本计算 softmax，提高效率。

在 ReChorus 这类框架中，如果是点乘召回或Ranking 任务，通常会直接用 \(\mathbf{h}_b\) 与候选 item embedding 做点积，得到评分矩阵 \(B \times K\)。

---

## 4. 为什么 AMS + MFFN 要组成一个 Block，而不是「合在一起」？

从结构和数学上看，这是一个 **Transformer 风格的标准设计**：

1. **AMS = 显式跨位置交互（explicit interaction）**

   * 通过注意力权重 \(A^{(\cdot)}_{n,m}\)，把不同时间步、不同 item 之间的关系显式建模：

     * 语义（QK）通道：基于内容相似度。
     * 时间通道：基于时间差分桶。
     * 位置通道：基于相对位置差。
   * 输出仍然在 token 维度上，一个位置对应一个向量。

2. **MFFN = 每个位置内部的非线性混合（implicit interaction）**

   * Stage 1：把多通道 AMS 输出融合回原始 \(D\) 维空间，并与原始输入做残差 —— 相当于**显式高阶交互的线性组合**。
   * Stage 2：SwiGLU + 线性层，相当于一个强大的位置内 MLP，负责**隐式地组合和筛选**不同维度的特征。

3. **分成两个子层 + 残差的好处：**

   * **功能解耦**：
     AMS 专注于 “token 之间怎么交互”；MFFN 专注于 “同一个 token 内部如何进行复杂非线性变换”。如果把全部操作挤到一个大矩阵里，会让结构变得不可解释，调参困难，也违背 Transformer 的经验设计。
   * **优化与训练稳定性**：
     每个子层都有自己的残差路径和归一化，类似 ResNet/Transformer 的设计，能够显著减轻深层网络的梯度消失/爆炸问题（这是 ResNet 论文提出残差连接的本质动机）。
   * **易于扩展规模**：
     AMS 和 MFFN 在复杂度分析上可以分开度量：AMS 是 \(O(N^2)\)，MFFN 是 \(O(N)\)。把它们作为一个 Block，可以系统地按 “堆层数、扩维度” 的方式去遵循 scaling law。

因此，「FuXi Block = AMS + MFFN」本质上是**“显式跨位置交互 + 隐式位置内交互” 的组合单元**，既保持了 Transformer 的良好工程特性，又在推荐场景下通过时间/位置通道和 SwiGLU 强化了特征交互能力。

---

如果你愿意，我可以在这个 .md 的基础上，再给你画一张 **完整的张量流动图（附上每条边的形状）**，或者帮你把这份说明改写成论文中的 “方法部分 4.x” 草稿。
