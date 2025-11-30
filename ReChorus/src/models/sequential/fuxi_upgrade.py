"""
FuXi-beta style encoder that removes the semantic (q/k) attention branch and
relies solely on positional and temporal streams.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BaseModel import SequentialModel

# Reuse utilities and path setup from the original FuXi implementation.
from . import fuxi as fuxi_alpha

import fbgemm_gpu  # noqa: F401

RelativeAttentionBiasModule = fuxi_alpha.RelativeAttentionBiasModule
SeperatedRelativeBucketedTimeAndPositionBasedBias = (
    fuxi_alpha.SeperatedRelativeBucketedTimeAndPositionBasedBias
)
MultistageFeedforwardNeuralNetwork = fuxi_alpha.MultistageFeedforwardNeuralNetwork
FuXiJagged = fuxi_alpha.FuXiJagged
GeneralizedInteractionModule = fuxi_alpha.GeneralizedInteractionModule
get_current_embeddings = fuxi_alpha.get_current_embeddings
get_similarity_function = fuxi_alpha.get_similarity_function
LocalEmbeddingModule = fuxi_alpha.LocalEmbeddingModule
LearnablePositionalEmbeddingInputFeaturesPreprocessor = (
    fuxi_alpha.LearnablePositionalEmbeddingInputFeaturesPreprocessor
)
L2NormEmbeddingPostprocessor = fuxi_alpha.L2NormEmbeddingPostprocessor
TIMESTAMPS_KEY = fuxi_alpha.TIMESTAMPS_KEY


def _two_stream_attention(
    num_heads: int,
    linear_dim: int,
    v: torch.Tensor,
    x_offsets: torch.Tensor,
    all_timestamps: torch.Tensor,
    invalid_attn_mask: torch.Tensor,
    rel_attn_bias: RelativeAttentionBiasModule,
) -> torch.Tensor:
    """
    Compute attention outputs using only positional and temporal relative biases.
    """
    B: int = x_offsets.size(0) - 1
    n: int = invalid_attn_mask.size(-1)

    padded_v = torch.ops.fbgemm.jagged_to_padded_dense(v, [x_offsets], [n]).reshape(
        B, n, num_heads, linear_dim
    )
    pos_attn, ts_attn = rel_attn_bias(all_timestamps)
    pos_attn = pos_attn * invalid_attn_mask.unsqueeze(0)
    ts_attn = ts_attn * invalid_attn_mask.unsqueeze(0)

    output_pos = torch.einsum("bnm,bmhd->bnhd", pos_attn, padded_v)
    output_ts = torch.einsum("bnm,bmhd->bnhd", ts_attn, padded_v)
    combined_output = torch.cat([output_pos, output_ts], dim=-1).reshape(
        B, n, num_heads * linear_dim * 2
    )
    attn_output = torch.ops.fbgemm.dense_to_jagged(combined_output, [x_offsets])[0]
    return attn_output, None, None


class SemanticFreeFuXiBlock(nn.Module):
    """
    FuXi block that drops the semantic q/k attention branch and keeps
    only positional/temporal channels.
    """

    def __init__(
        self,
        embedding_dim: int,
        linear_hidden_dim: int,
        attention_dim: int,
        dropout_ratio: float,
        attn_dropout_ratio: float,
        num_heads: int,
        linear_activation: str,
        relative_attention_bias_module: RelativeAttentionBiasModule,
        normalization: str = "rel_bias",
        epsilon: float = 1e-6,
        ffn_multiply: float = 1.0,
        ffn_single_stage: bool = False,
    ) -> None:
        super().__init__()
        self._embedding_dim: int = embedding_dim
        self._linear_dim: int = linear_hidden_dim
        self._attention_dim: int = attention_dim
        self._dropout_ratio: float = dropout_ratio
        self._attn_dropout_ratio: float = attn_dropout_ratio
        self._num_heads: int = num_heads
        self._rel_attn_bias: RelativeAttentionBiasModule = relative_attention_bias_module
        self._normalization: str = normalization
        self._linear_activation: str = linear_activation
        self._uv = nn.Parameter(
            torch.empty(
                (
                    embedding_dim,
                    linear_hidden_dim * num_heads * 3,  # u (2 streams) + v
                )
            ).normal_(mean=0, std=0.02)
        )
        self._mffn = MultistageFeedforwardNeuralNetwork(
            ams_output_size=linear_hidden_dim * num_heads * 2,
            input_size=embedding_dim,
            hidden_size=int(embedding_dim * ffn_multiply),
            output_size=embedding_dim,
            dropout_ratio=dropout_ratio,
            single_stage=ffn_single_stage,
            epsilon=epsilon,
        )
        self._mffn.init()
        self._eps: float = epsilon

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, normalized_shape=[self._embedding_dim], eps=self._eps)

    def _norm_attn_output(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(
            x, normalized_shape=[self._linear_dim * self._num_heads * 2], eps=self._eps
        )

    def forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: torch.Tensor,
        invalid_attn_mask: torch.Tensor,
        delta_x_offsets: torch.Tensor | None = None,
        cache: tuple | None = None,
        return_cache_states: bool = False,
    ):
        n: int = invalid_attn_mask.size(-1)
        if delta_x_offsets is not None:
            assert cache is not None
            x = x[delta_x_offsets[0], :]
            cached_v, _, _, cached_outputs = cache

        normed_x = self._norm_input(x)
        projected = torch.mm(normed_x, self._uv)
        if self._linear_activation == "silu":
            projected = F.silu(projected)
        elif self._linear_activation == "none":
            projected = projected
        u, v = torch.split(
            projected,
            [
                self._linear_dim * self._num_heads * 2,
                self._linear_dim * self._num_heads,
            ],
            dim=1,
        )
        if delta_x_offsets is not None:
            v = cached_v.index_copy_(dim=0, index=delta_x_offsets[0], source=v)

        if self._normalization == "rel_bias":
            attn_output, padded_q, padded_k = _two_stream_attention(
                num_heads=self._num_heads,
                linear_dim=self._linear_dim,
                v=v,
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


class SemanticFreeFuXi(GeneralizedInteractionModule):
    """
    FuXi-beta encoder that keeps positional + temporal branches and drops
    semantic attention.
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
        ffn_multiply: float,
        ffn_single_stage: bool,
        embedding_module: fuxi_alpha.EmbeddingModule,
        similarity_module: fuxi_alpha.NDPModule,
        input_features_preproc_module: fuxi_alpha.InputFeaturesPreprocessorModule,
        output_postproc_module: fuxi_alpha.OutputPostprocessorModule,
        enable_relative_attention_bias: bool = True,
        verbose: bool = True,
    ) -> None:
        super().__init__(ndp_module=similarity_module)

        self._embedding_dim: int = embedding_dim
        self._item_embedding_dim: int = embedding_module.item_embedding_dim
        self._max_sequence_length: int = max_sequence_len
        self._embedding_module: fuxi_alpha.EmbeddingModule = embedding_module
        self._input_features_preproc: fuxi_alpha.InputFeaturesPreprocessorModule = (
            input_features_preproc_module
        )
        self._output_postproc: fuxi_alpha.OutputPostprocessorModule = (
            output_postproc_module
        )
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
                SemanticFreeFuXiBlock(
                    embedding_dim=self._embedding_dim,
                    linear_hidden_dim=linear_dim,
                    attention_dim=attention_dim,
                    normalization=normalization,
                    linear_activation=linear_activation,
                    num_heads=num_heads,
                    relative_attention_bias_module=(
                        SeperatedRelativeBucketedTimeAndPositionBasedBias(
                            max_seq_len=max_sequence_len + max_output_len,
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
            except Exception:
                if self._verbose:
                    print(f"Failed to initialize {name}: {params.data.size()} params")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._embedding_module.get_item_embeddings(item_ids)

    def debug_str(self) -> str:
        debug_str = (
            f"FuXi-beta-nosm-b{self._num_blocks}-h{self._num_heads}-dv{self._dv}"
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
        past_payloads: dict[str, torch.Tensor],
        delta_x_offsets: torch.Tensor | None = None,
        cache: list | None = None,
        return_cache_states: bool = False,
    ):
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
        past_payloads: dict[str, torch.Tensor],
        batch_id: int | None = None,
    ) -> torch.Tensor:
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
        past_payloads: dict[str, torch.Tensor],
        delta_x_offsets: torch.Tensor | None,
        cache: list | None,
        return_cache_states: bool,
    ):
        encoded_seq_embeddings, cache_states = self.generate_user_embeddings(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
        )
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
        past_payloads: dict[str, torch.Tensor],
        delta_x_offsets: torch.Tensor | None = None,
        cache: list | None = None,
        return_cache_states: bool = False,
    ):
        return self._encode(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
        )


class FuXiUpgrade(SequentialModel):
    """
    ReChorus SequentialModel wrapper for the semantic-free FuXi-beta encoder.
    """

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
        parser.add_argument("--fuxi_blocks", type=int, default=2, help="FuXi blocks.")
        parser.add_argument("--fuxi_heads", type=int, default=4, help="Attention heads.")
        parser.add_argument(
            "--fuxi_linear_dim", type=int, default=64, help="Hidden dim in linear projections."
        )
        parser.add_argument(
            "--fuxi_attention_dim",
            type=int,
            default=64,
            help="Attention q/k dim per head (kept for compatibility, not used).",
        )
        parser.add_argument(
            "--fuxi_linear_dropout", type=float, default=0.1, help="Linear dropout rate."
        )
        parser.add_argument(
            "--fuxi_attn_dropout", type=float, default=0.1, help="Attention dropout rate."
        )
        parser.add_argument(
            "--fuxi_linear_activation", type=str, default="silu", help="Activation for projection MLP."
        )
        parser.add_argument(
            "--fuxi_ffn_multiply", type=float, default=1.0, help="FFN expansion factor."
        )
        parser.add_argument(
            "--fuxi_ffn_single_stage", type=int, default=0, help="Use single-stage FFN."
        )
        parser.add_argument(
            "--fuxi_enable_rel_bias", type=int, default=1, help="Enable relative bias buckets."
        )
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
        self.encoder = SemanticFreeFuXi(
            max_sequence_len=self.max_his,
            max_output_len=1,
            embedding_dim=self.emb_size,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            linear_dim=self.linear_dim,
            attention_dim=self.attn_dim,
            normalization="rel_bias",
            linear_config="uv",
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
        logits = (user_emb[:, None, :] * item_emb).sum(-1)
        return {"prediction": logits}

    class Dataset(SequentialModel.Dataset):
        pass


fuxi_upgrade = FuXiUpgrade
