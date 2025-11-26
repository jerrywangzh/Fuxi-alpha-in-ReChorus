"""
High-performance FuXi encoder copied from FuXi-alpha with detailed commentary.

This module keeps the original implementation intact while augmenting it with
inline documentation.  Each class, function, and critical variable definition
explains its role inside the FuXi hierarchy so that ReChorus users can study
the algorithm without jumping back and forth between repositories.
"""

import abc
import math
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BaseModel import SequentialModel

# --- 将 yinyong 目录加入 sys.path，以便直接复用 FuXi-alpha 的实现 ---
_YINYONG_ROOT = Path(__file__).resolve().parents[3] / "yinyong"
if _YINYONG_ROOT.exists():
    yinyong_path = str(_YINYONG_ROOT)
    if yinyong_path not in sys.path:
        sys.path.insert(0, yinyong_path)

# --- fbgemm-gpu 依赖检测 ---
try:
    import fbgemm_gpu  # noqa: F401
except ImportError as exc:  # 给出友好提示
    raise ImportError(
        "FuXi 依赖 fbgemm-gpu，请先安装与当前 CUDA 版本匹配的 wheel"
    ) from exc

_REQUIRED_OPS = ["dense_to_jagged", "jagged_to_padded_dense", "asynchronous_complete_cumsum"]
if not hasattr(torch.ops, "fbgemm") or any(
    not hasattr(torch.ops.fbgemm, op) for op in _REQUIRED_OPS
):
    missing_ops = [op for op in _REQUIRED_OPS if not hasattr(getattr(torch.ops.fbgemm, op, None), "__call__")]
    raise ImportError(f"缺少 fbgemm 算子: {missing_ops}. 请重新安装 fbgemm-gpu。")

# --- torch 版本兼容：若无 rms_norm 则退化为 layer_norm ---
if not hasattr(F, "rms_norm"):
    def _fallback_rms_norm(x: torch.Tensor, normalized_shape, eps: float = 1e-5):
        return torch.nn.functional.layer_norm(x, normalized_shape, eps=eps)
    F.rms_norm = _fallback_rms_norm  # type: ignore

from generative_recommenders.modeling.ndp_module import NDPModule

import einops
import logging

from generative_recommenders.modeling.sequential.embedding_modules import (
    EmbeddingModule,
    LocalEmbeddingModule,
)
from generative_recommenders.modeling.sequential.input_features_preprocessors import (
    InputFeaturesPreprocessorModule,
    LearnablePositionalEmbeddingInputFeaturesPreprocessor,
)
from generative_recommenders.modeling.sequential.output_postprocessors import (
    OutputPostprocessorModule,
    L2NormEmbeddingPostprocessor,
)
from generative_recommenders.modeling.sequential.utils import get_current_embeddings
from generative_recommenders.modeling.similarity_module import (
    GeneralizedInteractionModule,
)
from generative_recommenders.modeling.sequential.hstu import RelativeAttentionBiasModule
from generative_recommenders.modeling.similarity_utils import get_similarity_function

TIMESTAMPS_KEY = "timestamps"

class SeperatedRelativeBucketedTimeAndPositionBasedBias(RelativeAttentionBiasModule):
    """
    Learnable relative-bias generator used by FuXi attention.

    For every pair of sequence positions (i, j) it assigns:
        bias_pos(i, j) = w_pos[f(i - j)]
        bias_ts(i, j)  = w_ts[g(t_i - t_j)]
    where f(·) is a clipped positional offset and g(·) is a logarithmic time
    bucketization.  The weights (w_pos, w_ts) are trainable parameters shared
    across layers, implementing the bucket algorithm described in the FuXi
    paper for handling long-range spatial/temporal dependencies.
    """

    def __init__(
        self,
        max_seq_len: int,
        num_buckets: int,
        bucketization_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()

        self._max_seq_len: int = max_seq_len  # maximum context size used for clipping offsets
        self._ts_w = torch.nn.Parameter(
            torch.empty(num_buckets + 1).normal_(mean=0, std=0.02),
        )  # learnable weights for timestamp buckets (extra slot for overflow)
        self._pos_w = torch.nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02),
        )  # positional bias weights for offsets in [-L+1, L-1]
        self._num_buckets: int = num_buckets
        self._bucketization_fn: Callable[[torch.Tensor], torch.Tensor] = (
            bucketization_fn
        )  # callable mapping Δt to discrete bucket id

    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            all_timestamps: (B, N).
        Returns:
            (B, N, N).
        """
        B = all_timestamps.size(0)
        N = self._max_seq_len
        t = F.pad(self._pos_w[: 2 * N - 1], [0, N]).repeat(N)
        t = t[..., :-N].reshape(1, N, 3 * N - 2)
        r = (2 * N - 1) // 2

        # [B, N + 1] to simplify tensor manipulations.
        ext_timestamps = torch.cat(
            [all_timestamps, all_timestamps[:, N - 1 : N]], dim=1
        )
        # causal masking. Otherwise [:, :-1] - [:, 1:] works
        bucketed_timestamps = torch.clamp(
            self._bucketization_fn(
                ext_timestamps[:, 1:].unsqueeze(2) - ext_timestamps[:, :-1].unsqueeze(1)
            ),
            min=0,
            max=self._num_buckets,
        ).detach()
        rel_pos_bias = t[:, :, r:-r]
        rel_ts_bias = torch.index_select(
            self._ts_w, dim=0, index=bucketed_timestamps.view(-1)
        ).view(B, N, N)
        return rel_pos_bias, rel_ts_bias


#  (v, padded_q, padded_k, new_outputs)
FuXiCacheState = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

def _adaptive_multichannel_attention_maybe_from_cache(
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cached_q: Optional[torch.Tensor],
    cached_k: Optional[torch.Tensor],
    delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]],
    x_offsets: torch.Tensor,
    all_timestamps: Optional[torch.Tensor],
    invalid_attn_mask: torch.Tensor,
    rel_attn_bias: RelativeAttentionBiasModule,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Core attention kernel used inside FuXi blocks.

    The function mixes three information channels:
        1) content-content attention (query/key dot product)
        2) relative positional bias
        3) relative temporal bias

    It additionally supports incremental decoding by consuming cached q/k/v
    tensors via `delta_x_offsets`.  The output is the jagged attention response
    together with padded q/k tensors for caching.
    """
    B: int = x_offsets.size(0) - 1
    n: int = invalid_attn_mask.size(-1)
    if delta_x_offsets is not None:
        padded_q, padded_k = cached_q, cached_k
        flattened_offsets = delta_x_offsets[1] + torch.arange(
            start=0,
            end=B * n,
            step=n,
            device=delta_x_offsets[1].device,
            dtype=delta_x_offsets[1].dtype,
        )
        assert isinstance(padded_q, torch.Tensor)
        assert isinstance(padded_k, torch.Tensor)
        padded_q = (
            padded_q.view(B * n, -1)
            .index_copy_(
                dim=0,
                index=flattened_offsets,
                source=q,
            )
            .view(B, n, -1)
        )
        padded_k = (
            padded_k.view(B * n, -1)
            .index_copy_(
                dim=0,
                index=flattened_offsets,
                source=k,
            )
            .view(B, n, -1)
        )
    else:
        padded_q = torch.ops.fbgemm.jagged_to_padded_dense(
            values=q, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
        )
        padded_k = torch.ops.fbgemm.jagged_to_padded_dense(
            values=k, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
        )

    qk_attn = torch.einsum(
        "bnhd,bmhd->bhnm",
        padded_q.view(B, n, num_heads, attention_dim),
        padded_k.view(B, n, num_heads, attention_dim),
    )
    
    qk_attn = F.silu(qk_attn) / n
    
    qk_attn = qk_attn * invalid_attn_mask.unsqueeze(0).unsqueeze(0)
    
    assert (all_timestamps is not None)
    # (B, n, n)
    if rel_attn_bias :
        pos_attn, ts_attn = rel_attn_bias(all_timestamps)
        pos_attn = pos_attn * invalid_attn_mask.unsqueeze(0)
        ts_attn = ts_attn * invalid_attn_mask.unsqueeze(0)
    
    padded_v = torch.ops.fbgemm.jagged_to_padded_dense(v, [x_offsets], [n]).reshape(
        B, n, num_heads, linear_dim
    )
    
    output_pos = torch.einsum(
            "bnm,bmhd->bnhd",
            pos_attn,
            padded_v,
        )
    output_ts = torch.einsum(
            "bnm,bmhd->bnhd",
            ts_attn,
            padded_v,
        )
    output_latent = torch.einsum(
            "bhnm,bmhd->bnhd",
            qk_attn,
            padded_v,
        )
    
    combined_output = torch.concat(
        [output_pos, output_ts, output_latent],
        dim=-1
    ).reshape(B, n, num_heads * linear_dim * 3)
    
    attn_output = torch.ops.fbgemm.dense_to_jagged(
        combined_output, [x_offsets],
    )[0]
    
    return attn_output, padded_q, padded_k

class MultistageFeedforwardNeuralNetwork(torch.nn.Module) :
    """Implements the AMS-weighted feed-forward network (time-space weighting)."""

    def __init__(
        self, 
        ams_output_size, 
        input_size, 
        hidden_size, 
        output_size, 
        dropout_ratio: float,
        bias: bool = False, 
        single_stage: bool = False,
        epsilon: float = 1e-6,
    ) :
        super(MultistageFeedforwardNeuralNetwork, self).__init__()
        self.lin0 = torch.nn.Linear(ams_output_size, input_size)
        self.is_single_stage = single_stage
        self.dropout_ratio = dropout_ratio
        self.input_size = input_size
        self.eps = epsilon
        if not single_stage :
            self.lin1 = torch.nn.Linear(input_size, hidden_size, bias=bias)  # φ(x) branch
            self.lin2 = torch.nn.Linear(hidden_size, output_size, bias=bias)  # projection back to model dim
            self.lin3 = torch.nn.Linear(input_size, hidden_size, bias=bias)  # gating branch ψ(x)
    
    def forward(self, X, X0) :
        X = (
            self.lin0(
                F.dropout(
                    X,
                    p = self.dropout_ratio,
                    training = self.training
                )
            ) + X0
        )
        if not self.is_single_stage :
            normed_X = F.rms_norm(X, normalized_shape=[self.input_size], eps=self.eps)
            normed_X = F.dropout(
                normed_X,
                p = self.dropout_ratio,
                training = self.training
            )
            X1 = F.silu(self.lin1(normed_X)) * self.lin3(normed_X)
            X = self.lin2(X1) + X
        return X
    
    def init(self) :
        torch.nn.init.xavier_uniform_(self.lin0.weight)
        if not self.is_single_stage :
            torch.nn.init.xavier_uniform_(self.lin1.weight)
            torch.nn.init.xavier_uniform_(self.lin2.weight)
            torch.nn.init.xavier_uniform_(self.lin3.weight)
        
class FuXiBlockJagged(torch.nn.Module):
    """Single FuXi block operating on jagged tensors (variable-length sequences)."""

    def __init__(
        self,
        embedding_dim: int,
        linear_hidden_dim: int,
        attention_dim: int,
        dropout_ratio: float,
        attn_dropout_ratio: float,
        num_heads: int,
        linear_activation: str,
        relative_attention_bias_module: Optional[RelativeAttentionBiasModule] = None,
        normalization: str = "rel_bias",
        linear_config: str = "uvqk",
        # concat_ua: bool = False,
        epsilon: float = 1e-6,
        ffn_multiply: float = 1,
        ffn_single_stage: bool = False,
        max_length: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._embedding_dim: int = embedding_dim
        self._linear_dim: int = linear_hidden_dim
        self._attention_dim: int = attention_dim
        self._dropout_ratio: float = dropout_ratio
        self._attn_dropout_ratio: float = attn_dropout_ratio
        self._num_heads: int = num_heads
        self._rel_attn_bias: Optional[RelativeAttentionBiasModule] = (
            relative_attention_bias_module
        )
        self._normalization: str = normalization
        self._linear_config: str = linear_config
        if self._linear_config == "uvqk":
            self._uvqk: torch.nn.Parameter = torch.nn.Parameter(
                torch.empty(
                    (
                        embedding_dim,
                        linear_hidden_dim * num_heads * 4
                        + attention_dim * num_heads * 2,
                    )
                ).normal_(mean=0, std=0.02),
            )
        else:
            raise ValueError(f"Unknown linear_config {self._linear_config}")
        self._linear_activation: str = linear_activation
        
        self._mffn = MultistageFeedforwardNeuralNetwork(
            ams_output_size = linear_hidden_dim * num_heads * 3,
            input_size = embedding_dim,
            hidden_size = int(embedding_dim * ffn_multiply),
            output_size = embedding_dim,
            dropout_ratio = dropout_ratio,
            single_stage = ffn_single_stage,
            epsilon = epsilon
        )
        self._mffn.init()
        
        self._eps: float = epsilon

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, normalized_shape=[self._embedding_dim], eps=self._eps)

    def _norm_attn_output(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(
            x, normalized_shape=[self._linear_dim * self._num_heads * 3], eps=self._eps
        )

    def forward(  # pyre-ignore [3]
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[FuXiCacheState] = None,
        return_cache_states: bool = False,
    ):
        """
        Args:
            x: (\sum_i N_i, D) x float.
            x_offsets: (B + 1) x int32.
            all_timestamps: optional (B, N) x int64.
            invalid_attn_mask: (B, N, N) x float, each element in {0, 1}.
            delta_x_offsets: optional 2-tuple ((B,) x int32, (B,) x int32).
                For the 1st element in the tuple, each element is in [0, x_offsets[-1]). For the
                2nd element in the tuple, each element is in [0, N).
            cache: Optional 4-tuple of (v, padded_q, padded_k, output) from prior runs,
                where all except padded_q, padded_k are jagged.
        Returns:
            x' = f(x), (\sum_i N_i, D) x float.
        """
        n: int = invalid_attn_mask.size(-1)
        cached_q = None
        cached_k = None
        if delta_x_offsets is not None:
            # In this case, for all the following code, x, u, v, q, k become restricted to
            # [delta_x_offsets[0], :].
            assert cache is not None
            x = x[delta_x_offsets[0], :]
            cached_v, cached_q, cached_k, cached_outputs = cache
            
        normed_x = self._norm_input(x)

        if self._linear_config == "uvqk":
            batched_mm_output = torch.mm(normed_x, self._uvqk)
            if self._linear_activation == "silu":
                batched_mm_output = F.silu(batched_mm_output)
            elif self._linear_activation == "none":
                batched_mm_output = batched_mm_output
            u, v, q, k = torch.split(
                batched_mm_output,
                [
                    self._linear_dim * self._num_heads * 3,
                    self._linear_dim * self._num_heads,
                    self._attention_dim * self._num_heads,
                    self._attention_dim * self._num_heads,
                ],
                dim=1,
            )
        else:
            raise ValueError(f"Unknown self._linear_config {self._linear_config}")

        if delta_x_offsets is not None:
            v = cached_v.index_copy_(dim=0, index=delta_x_offsets[0], source=v)

        B: int = x_offsets.size(0) - 1
        if self._normalization == "rel_bias":
            assert self._rel_attn_bias is not None
            attn_output, padded_q, padded_k = _adaptive_multichannel_attention_maybe_from_cache(
                num_heads=self._num_heads,
                attention_dim=self._attention_dim,
                linear_dim=self._linear_dim,
                q=q,
                k=k,
                v=v,
                cached_q=cached_q,
                cached_k=cached_k,
                delta_x_offsets=delta_x_offsets,
                x_offsets=x_offsets,
                all_timestamps=all_timestamps,
                invalid_attn_mask=invalid_attn_mask,
                rel_attn_bias=self._rel_attn_bias,
            )
        else:
            raise ValueError(f"Unknown normalization method {self._normalization}")

        attn_output = (
            attn_output
            if delta_x_offsets is None
            else attn_output[delta_x_offsets[0], :]
        )
        ams_output = u * self._norm_attn_output(attn_output)

        new_outputs = self._mffn(ams_output, x)

        if delta_x_offsets is not None:
            new_outputs = cached_outputs.index_copy_(
                dim=0, index=delta_x_offsets[0], source=new_outputs
            )

        if return_cache_states and delta_x_offsets is None:
            v = v.contiguous()

        return new_outputs, (v, padded_q, padded_k, new_outputs)


class FuXiJagged(torch.nn.Module):
    """Wrapper that stacks multiple `FuXiBlockJagged` modules."""

    def __init__(
        self,
        modules: List[FuXiBlockJagged],
        autocast_dtype: Optional[torch.dtype],
    ) -> None:
        super().__init__()

        self._attention_layers: torch.nn.ModuleList = torch.nn.ModuleList(
            modules=modules
        )
        self._autocast_dtype: Optional[torch.dtype] = autocast_dtype

    def jagged_forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[FuXiCacheState]] = None,
        return_cache_states: bool = False,
    ) -> Tuple[torch.Tensor, List[FuXiCacheState]]:
        """
        Args:
            x: (\sum_i N_i, D) x float
            x_offsets: (B + 1) x int32
            all_timestamps: (B, 1 + N) x int64
            invalid_attn_mask: (B, N, N) x float, each element in {0, 1}
            return_cache_states: bool. True if we should return cache states.

        Returns:
            x' = f(x), (\sum_i N_i, D) x float
        """
        cache_states: List[FuXiCacheState] = []

        with torch.autocast(
            "cuda",
            enabled=self._autocast_dtype is not None,
            dtype=self._autocast_dtype or torch.float16,
        ):
            for i, layer in enumerate(self._attention_layers):
                x, cache_states_i = layer(
                    x=x,
                    x_offsets=x_offsets,
                    all_timestamps=all_timestamps,
                    invalid_attn_mask=invalid_attn_mask,
                    delta_x_offsets=delta_x_offsets,
                    cache=cache[i] if cache is not None else None,
                    return_cache_states=return_cache_states,
                )
                if return_cache_states:
                    cache_states.append(cache_states_i)

        return x, cache_states

    def forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[FuXiCacheState]] = None,
        return_cache_states: bool = False,
    ) -> Tuple[torch.Tensor, List[FuXiCacheState]]:
        """
        Args:
            x: (B, N, D) x float.
            x_offsets: (B + 1) x int32.
            all_timestamps: (B, 1 + N) x int64
            invalid_attn_mask: (B, N, N) x float, each element in {0, 1}.
        Returns:
            x' = f(x), (B, N, D) x float
        """
        if len(x.size()) == 3:
            x = torch.ops.fbgemm.dense_to_jagged(x, [x_offsets])[0]

        jagged_x, cache_states = self.jagged_forward(
            x=x,
            x_offsets=x_offsets,
            all_timestamps=all_timestamps,
            invalid_attn_mask=invalid_attn_mask,
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
        )
        y = torch.ops.fbgemm.jagged_to_padded_dense(
            values=jagged_x,
            offsets=[x_offsets],
            max_lengths=[invalid_attn_mask.size(1)],
            padding_value=0.0,
        )
        return y, cache_states


class FuXi(GeneralizedInteractionModule):
    """
    Top-level FuXi-alpha encoder that glues embedding, attention blocks, and
    similarity module together.  It implements the AMS attention described in
    the paper and exposes utility functions (`encode`, `generate_user_embeddings`)
    consumed by both the standalone trainer and the ReChorus wrapper.
    """

    def __init__(
        self,
        max_sequence_len: int,
        max_output_len: int,
        embedding_dim: int,
        num_blocks: int,
        num_heads: int,
        linear_dim: int,
        attention_dim: int,
        normalization: str,
        linear_config: str,
        linear_activation: str,
        linear_dropout_rate: float,
        attn_dropout_rate: float,
        ffn_multiply: int,
        ffn_single_stage: bool,
        embedding_module: EmbeddingModule,
        similarity_module: NDPModule,
        input_features_preproc_module: InputFeaturesPreprocessorModule,
        output_postproc_module: OutputPostprocessorModule,
        enable_relative_attention_bias: bool = True,
        verbose: bool = True,
    ) -> None:
        super().__init__(ndp_module=similarity_module)

        self._embedding_dim: int = embedding_dim
        self._item_embedding_dim: int = embedding_module.item_embedding_dim
        self._max_sequence_length: int = max_sequence_len
        self._embedding_module: EmbeddingModule = embedding_module
        self._input_features_preproc: InputFeaturesPreprocessorModule = (
            input_features_preproc_module
        )
        self._output_postproc: OutputPostprocessorModule = output_postproc_module
        self._num_blocks: int = num_blocks
        self._num_heads: int = num_heads
        self._dqk: int = attention_dim
        self._dv: int = linear_dim
        self._linear_activation: str = linear_activation
        self._linear_dropout_rate: float = linear_dropout_rate
        self._attn_dropout_rate: float = attn_dropout_rate
        self._enable_relative_attention_bias: bool = enable_relative_attention_bias
        self._fuxi = FuXiJagged(
            modules=[
                FuXiBlockJagged(
                    embedding_dim=self._embedding_dim,
                    linear_hidden_dim=linear_dim,
                    attention_dim=attention_dim,
                    normalization=normalization,
                    linear_config=linear_config,
                    linear_activation=linear_activation,
                    num_heads=num_heads,
                    relative_attention_bias_module=(
                        SeperatedRelativeBucketedTimeAndPositionBasedBias(
                            max_seq_len=max_sequence_len
                            + max_output_len,  # accounts for next item.
                            num_buckets=128,
                            bucketization_fn=lambda x: (
                                torch.log(torch.abs(x).clamp(min=1)) / 0.301
                            ).long(),
                        )
                        if enable_relative_attention_bias
                        else None
                    ),
                    dropout_ratio=linear_dropout_rate,
                    attn_dropout_ratio=attn_dropout_rate,
                    ffn_multiply=ffn_multiply,
                    ffn_single_stage=ffn_single_stage,
                )
                for _ in range(num_blocks)
            ],
            autocast_dtype=None,
        )
        # causal forward, w/ +1 for padding.
        self.register_buffer(
            "_attn_mask",
            torch.triu(
                torch.ones(
                    (
                        self._max_sequence_length + max_output_len,
                        self._max_sequence_length + max_output_len,
                    ),
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
        )
        self._verbose: bool = verbose
        self.reset_params()

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if ("_fuxi" in name) or ("_embedding_module" in name):
                if self._verbose:
                    print(f"Skipping init for {name}")
                continue
            try:
                torch.nn.init.xavier_normal_(params.data)
                if self._verbose:
                    print(
                        f"Initialize {name} as xavier normal: {params.data.size()} params"
                    )
            except:
                if self._verbose:
                    print(f"Failed to initialize {name}: {params.data.size()} params")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._embedding_module.get_item_embeddings(item_ids)

    def debug_str(self) -> str:
        debug_str = (
            f"FuXi-alpha-b{self._num_blocks}-h{self._num_heads}-dqk{self._dqk}-dv{self._dv}"
            + f"-l{self._linear_activation}d{self._linear_dropout_rate}"
            + f"-ad{self._attn_dropout_rate}"
        )
        if not self._enable_relative_attention_bias:
            debug_str += "-norab"
        return debug_str

    def generate_user_embeddings(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[FuXiCacheState]] = None,
        return_cache_states: bool = False,
    ) -> Tuple[torch.Tensor, List[FuXiCacheState]]:
        """
        [B, N] -> [B, N, D].
        """
        device = past_lengths.device
        float_dtype = past_embeddings.dtype
        B, N, _ = past_embeddings.size()

        past_lengths, user_embeddings, _ = self._input_features_preproc(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )

        float_dtype = user_embeddings.dtype
        user_embeddings, cached_states = self._fuxi(
            x=user_embeddings,
            x_offsets=torch.ops.fbgemm.asynchronous_complete_cumsum(past_lengths),
            all_timestamps=(
                past_payloads[TIMESTAMPS_KEY]
                if TIMESTAMPS_KEY in past_payloads
                else None
            ),
            invalid_attn_mask=1.0 - self._attn_mask.to(float_dtype),
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
        )
        return self._output_postproc(user_embeddings), cached_states

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        batch_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Runs the main encoder.

        Args:
            past_lengths: (B,) x int64
            past_ids: (B, N,) x int64 where the latest engaged ids come first. In
                particular, past_ids[i, past_lengths[i] - 1] should correspond to
                the latest engaged values.
            past_embeddings: (B, N, D) x float or (\sum_b N_b, D) x float.
            past_payloads: implementation-specific keyed tensors of shape (B, N, ...).

        Returns:
            encoded_embeddings of [B, N, D].
        """
        encoded_embeddings, _ = self.generate_user_embeddings(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )
        return encoded_embeddings

    def _encode(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]],
        cache: Optional[List[FuXiCacheState]],
        return_cache_states: bool,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[FuXiCacheState]]]:
        """
        Args:
            past_lengths: (B,) x int64.
            past_ids: (B, N,) x int64.
            past_embeddings: (B, N, D,) x float.
            past_payloads: implementation-specific keyed tensors of shape (B, N, ...).
            return_cache_states: bool.

        Returns:
            (B, D) x float, representing embeddings for the current state.
        """
        encoded_seq_embeddings, cache_states = self.generate_user_embeddings(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
        )  # [B, N, D]
        current_embeddings = get_current_embeddings(
            lengths=past_lengths, encoded_embeddings=encoded_seq_embeddings
        )
        if return_cache_states:
            return current_embeddings, cache_states
        else:
            return current_embeddings

    def encode(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[FuXiCacheState]] = None,
        return_cache_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[FuXiCacheState]]]:
        """
        Runs encoder to obtain the current hidden states.

        Args:
            past_lengths: (B,) x int.
            past_ids: (B, N,) x int.
            past_embeddings: (B, N, D) x float.
            past_payloads: implementation-specific keyed tensors of shape (B, N, ...).

        Returns:
            (B, D,) x float, representing encoded states at the most recent time step.
        """
        return self._encode(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
        )


# --- ReChorus SequentialModel 封装 ----------------------------------------------------------
FuXiAlphaCore = FuXi  # Keep reference to the original FuXi-alpha encoder class


class FuXi(SequentialModel):
    """ReChorus SequentialModel wrapper around the FuXi-alpha encoder."""

    reader = "SeqReader"
    runner = "BaseRunner"
    extra_log_args = ["emb_size", "dropout", "history_max"]

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument(
            "--emb_size",
            type=int,
            default=240,
            help="Item embedding / FuXi hidden size.",
        )
        parser.add_argument("--fuxi_blocks", type=int, default=2, help="# of FuXi blocks.")
        parser.add_argument("--fuxi_heads", type=int, default=4, help="# of attention heads.")
        parser.add_argument("--fuxi_linear_dim", type=int, default=64, help="Hidden dim in linear projections.")
        parser.add_argument("--fuxi_attention_dim", type=int, default=64, help="Attention q/k dim per head.")
        parser.add_argument("--fuxi_linear_dropout", type=float, default=0.1, help="Linear dropout rate.")
        parser.add_argument("--fuxi_attn_dropout", type=float, default=0.1, help="Attention dropout rate.")
        parser.add_argument("--fuxi_linear_activation", type=str, default="silu", help="Activation for projection MLP.")
        parser.add_argument("--fuxi_ffn_multiply", type=float, default=1.0, help="FFN expansion factor.")
        parser.add_argument("--fuxi_ffn_single_stage", type=int, default=0, help="Use single-stage FFN.")
        parser.add_argument("--fuxi_enable_rel_bias", type=int, default=1, help="Enable relative bias buckets.")
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.max_his = args.history_max
        self.dropout = args.dropout

        self.num_blocks = args.fuxi_blocks
        self.num_heads = args.fuxi_heads
        self.linear_dim = args.fuxi_linear_dim
        self.attn_dim = args.fuxi_attention_dim
        self.linear_dropout = args.fuxi_linear_dropout
        self.attn_dropout = args.fuxi_attn_dropout
        self.linear_activation = args.fuxi_linear_activation
        self.ffn_multiply = args.fuxi_ffn_multiply
        self.ffn_single_stage = bool(args.fuxi_ffn_single_stage)
        self.enable_rel_bias = bool(args.fuxi_enable_rel_bias)

        self.embedding_module = LocalEmbeddingModule(
            num_items=self.item_num,
            item_embedding_dim=self.emb_size,
        )
        self.interaction_module, _ = get_similarity_function(
            module_type="DotProduct",
            query_embedding_dim=self.emb_size,
            item_embedding_dim=self.emb_size,
        )
        self.input_preproc = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
            max_sequence_len=self.max_his + 1,
            embedding_dim=self.emb_size,
            dropout_rate=self.dropout,
        )
        self.output_postproc = L2NormEmbeddingPostprocessor(
            embedding_dim=self.emb_size,
            eps=1e-6,
        )
        self.encoder = FuXiAlphaCore(
            max_sequence_len=self.max_his,
            max_output_len=1,
            embedding_dim=self.emb_size,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            linear_dim=self.linear_dim,
            attention_dim=self.attn_dim,
            normalization="rel_bias",
            linear_config="uvqk",
            linear_activation=self.linear_activation,
            linear_dropout_rate=self.linear_dropout,
            attn_dropout_rate=self.attn_dropout,
            ffn_multiply=self.ffn_multiply,
            ffn_single_stage=self.ffn_single_stage,
            embedding_module=self.embedding_module,
            similarity_module=self.interaction_module,
            input_features_preproc_module=self.input_preproc,
            output_postproc_module=self.output_postproc,
            enable_relative_attention_bias=self.enable_rel_bias,
            verbose=False,
        )

    def forward(self, feed_dict):
        item_ids = feed_dict["item_id"].to(self.device)
        history = feed_dict["history_items"].to(self.device)
        timestamps = feed_dict["history_times"].to(self.device)
        lengths = feed_dict["lengths"].to(self.device)

        max_len = self.max_his + 1
        if history.size(1) < max_len:
            pad_len = max_len - history.size(1)
            history = F.pad(history, (0, pad_len))
            timestamps = F.pad(timestamps, (0, pad_len))
        elif history.size(1) > max_len:
            history = history[:, -max_len:]
            timestamps = timestamps[:, -max_len:]
            lengths = torch.clamp(lengths, max=max_len)

        past_embs = self.embedding_module.get_item_embeddings(history)
        user_emb = self.encoder.encode(
            past_lengths=lengths,
            past_ids=history,
            past_embeddings=past_embs,
            past_payloads={TIMESTAMPS_KEY: timestamps},
        )
        item_emb = self.embedding_module.get_item_embeddings(item_ids)
        logits = (user_emb[:, None, :] * item_emb).sum(-1)  # standard dot-product scoring
        return {"prediction": logits}

    class Dataset(SequentialModel.Dataset):
        pass
