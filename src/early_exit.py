"""真正逐层执行的 Early-Exit 引擎（P4）。

关键原则（§6.1）：不能 ``bert(..., output_hidden_states=True)`` 后再挑某个头——
那已经跑完了全部 12 层。真正的 Early-Exit 必须每执行一层就决定是否继续。

流程（§6.2，优先复用当前 Transformers 的 mask/encoder 约定）：
1. ``bert.embeddings`` 计算 embedding；
2. ``create_bidirectional_mask`` 构造 extended attention mask
   （transformers 5.15：无 padding 返回 None，有 padding 返回 [B,1,S,S] float mask）；
3. 循环 ``bert.encoder.layer[i]``（5.15 中 layer 直接返回 Tensor）；
4. 在候选层对 hidden state 做相同 pooling，调用对应 head 得到 logits；
5. 计算退出分数；满足条件立即返回，不再执行后续层；
6. encoder layer 11 强制兜底。

执行追踪：每个 encoder layer 挂 forward hook 计数，证明退出后确实不再调用后续层。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .heads import N_ENCODER_LAYERS
from .pooling import apply_pooling


class CacheVersionError(Exception):  # re-export for convenience (unused here)
    pass


@dataclass
class ExitResult:
    """单次 Early-Exit 前向的结果与执行追踪。"""

    logits: torch.Tensor                 # [B, 2]
    probabilities: torch.Tensor          # [B, 2]
    exit_layer: int                      # 退出的 encoder layer
    executed_layer_count: int            # 实际执行的 encoder layer 数
    exit_reason: str                     # fixed_layer / max_prob_threshold / ... / fallback
    latency_ms: float
    layer_call_counts: List[int] = field(default_factory=lambda: [0] * N_ENCODER_LAYERS)
    all_layer_logits: Optional[torch.Tensor] = None   # 若请求，[B, 12, 2]


def _build_extended_mask(bert, embeds: torch.Tensor, attention_mask: Optional[torch.Tensor]):
    """复用 transformers 5.15 原生 mask 构造；无法导入时回退到手工 4D mask。"""
    try:
        from transformers.masking_utils import create_bidirectional_mask
        return create_bidirectional_mask(config=bert.config, inputs_embeds=embeds,
                                         attention_mask=attention_mask)
    except Exception:  # noqa: BLE001  # 回退：BERT 经典 2D->4D 扩展
        if attention_mask is None:
            return None
        extended = attention_mask[:, None, None, :]
        extended = extended.to(dtype=embeds.dtype)
        extended = (1.0 - extended) * torch.finfo(embeds.dtype).min
        return extended


class EarlyExitEngine(nn.Module):
    """真正逐层执行的 Early-Exit 引擎。

    head_type 支持：copied_layer_heads / random_layer_heads / normalized_layer_heads
    （逐层独立 Linear）；shared_frozen_head（同一 fc 作用于每层）；
    original_final_head（仅最终层兜底，其余层不产生退出判定）。
    """

    def __init__(
        self,
        bert: nn.Module,
        heads_model: nn.Module,
        head_type: str = "copied_layer_heads",
        pooling: str = "masked_mean",
        fallback_layer: int = N_ENCODER_LAYERS - 1,
        candidate_layers: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.bert = bert
        self.bert.eval()
        self.bert.requires_grad_(False)
        self.heads_model = heads_model
        self.head_type = head_type
        self.pooling = pooling
        self.fallback_layer = fallback_layer
        self.candidate_layers = sorted(set(candidate_layers or [2, 4, 6, 8, 10, fallback_layer]))

        # 每层 forward hook：独立证明退出后不再调用后续层
        self._counters: List[int] = [0] * N_ENCODER_LAYERS
        self._hooks: List = []
        for i, layer in enumerate(self.bert.encoder.layer):
            self._hooks.append(layer.register_forward_hook(self._make_counter_hook(i)))

    # ------------------------------------------------------------------ #
    def _make_counter_hook(self, i: int):
        def hook(module, inp, out):
            self._counters[i] += 1
        return hook

    def reset_counters(self) -> None:
        self._counters = [0] * N_ENCODER_LAYERS

    def layer_call_counts(self) -> List[int]:
        return list(self._counters)

    # ------------------------------------------------------------------ #
    def _head_logits(self, layer_idx: int, pooled: torch.Tensor) -> Optional[torch.Tensor]:
        """在指定 encoder layer 应用 head，得到 [B, 2] logits。

        对仅最终层适用的 head（original_final_head），非最终层返回 None。
        """
        ht = self.head_type
        if ht in ("copied_layer_heads", "random_layer_heads", "normalized_layer_heads"):
            module = getattr(self.heads_model, ht)
            return module.heads[layer_idx](pooled)
        if ht == "shared_frozen_head":
            return self.heads_model.shared_frozen_head.head(pooled)
        if ht == "original_final_head":
            if layer_idx != self.fallback_layer:
                return None
            return self.heads_model.original_final_head(pooled)
        raise ValueError(f"未知 head_type: {ht!r}")

    def _pool(self, hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        return apply_pooling(self.pooling, hidden, attention_mask=attention_mask,
                             pooler=self.bert.pooler)

    # ------------------------------------------------------------------ #
    def _run_loop(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        token_type_ids: Optional[torch.Tensor],
        exit_cond: Callable[[int, torch.Tensor], Optional[str]],
        evaluate_layers: Optional[Sequence[int]] = None,
        record_all_logits: bool = False,
    ) -> ExitResult:
        """通用逐层循环。

        exit_cond(i, logits) -> Optional[str]：在层 i 执行后调用，返回退出原因
        或 None（继续）。最终层恒强制退出。

        evaluate_layers：只在指定层计算 pooling + head（生产 Early-Exit 只会在
        候选退出层评估头，其余层只执行不加头），None 表示每层都评估（调试）。
        """
        self.reset_counters()
        t0 = time.perf_counter()

        embeds = self.bert.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        ext_mask = _build_extended_mask(self.bert, embeds, attention_mask)
        h = embeds

        executed = 0
        exit_layer = self.fallback_layer
        exit_reason = "fallback"
        final_logits: Optional[torch.Tensor] = None
        all_logits: List[torch.Tensor] = []

        eval_set = None if evaluate_layers is None else set(evaluate_layers)
        for i in range(self.fallback_layer + 1):
            h = self.bert.encoder.layer[i](h, attention_mask=ext_mask)
            executed += 1
            if eval_set is not None and i not in eval_set:
                continue  # 只执行层，不在该层评估头（降低无关开销）
            pooled = self._pool(h, attention_mask)
            logits = self._head_logits(i, pooled)
            if logits is None:
                # head 在该层不适用（如 original_final_head），继续下一层
                if i == self.fallback_layer:
                    raise RuntimeError(f"head_type={self.head_type} 在最终层应产出 logits")
                continue
            if record_all_logits:
                all_logits.append(logits)
            reason = exit_cond(i, logits)
            if reason is not None:
                final_logits = logits
                exit_layer = i
                exit_reason = reason
                break
            final_logits = logits  # 兜底：最后一轮循环必然触发 break

        assert final_logits is not None
        latency_ms = (time.perf_counter() - t0) * 1000.0
        probs = torch.softmax(final_logits, dim=-1)
        return ExitResult(
            logits=final_logits,
            probabilities=probs,
            exit_layer=exit_layer,
            executed_layer_count=executed,
            exit_reason=exit_reason,
            latency_ms=latency_ms,
            layer_call_counts=self.layer_call_counts(),
            all_layer_logits=torch.stack(all_logits, dim=1) if record_all_logits else None,
        )

    # ------------------------------------------------------------------ #
    # v1：固定层退出（无数据阶段唯一正式策略）
    # ------------------------------------------------------------------ #
    def run_fixed(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        exit_layer: int = N_ENCODER_LAYERS - 1,
    ) -> ExitResult:
        """整个 batch 在第 exit_layer 层固定退出。"""
        if not (0 <= exit_layer <= self.fallback_layer):
            raise ValueError(f"exit_layer={exit_layer} 超出 [0, {self.fallback_layer}]")

        def cond(i, logits):
            if i == exit_layer:
                return "fixed_layer"
            return None

        # 生产成本模型：只在退出层评估 head；最终兜底层恒评估（保证必有输出）
        result = self._run_loop(input_ids, attention_mask, token_type_ids, cond,
                                evaluate_layers=sorted({exit_layer, self.fallback_layer}))
        # 若 head 在该层不适用而实际退到兜底层，则如实标记 fallback，不冒充 fixed_layer
        if result.exit_layer == exit_layer:
            result.exit_reason = "fixed_layer"
        return result

    # ------------------------------------------------------------------ #
    # v1：动态阈值退出（仅控制流 smoke test，未校准，不得用于正式部署）
    # ------------------------------------------------------------------ #
    def run_dynamic(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        strategy: str = "max_prob",
        threshold: float = 0.9,
        candidate_layers: Optional[Sequence[int]] = None,
    ) -> ExitResult:
        """batch size 1（或整批）动态退出。

        策略：
        - max_prob: max(softmax(logits)) >= threshold（整批 all 满足才退出）
        - margin:   |logit_1 - logit_0| >= threshold（logit 差）
        未校准的 softmax/margin 不能用于正式部署（§6.3）。
        """
        if input_ids.shape[0] != 1:
            raise ValueError("v1 动态退出仅支持 batch size 1（active-set 见 run_active_set）")
        cands = sorted(set(candidate_layers or self.candidate_layers))
        eval_layers = sorted(set(cands) | {self.fallback_layer})  # 兜底层恒评估

        if strategy == "max_prob":
            def cond(i, logits):
                if i in cands and logits.softmax(dim=-1).max(dim=-1).values.item() >= threshold:
                    return "max_prob_threshold"
                return None
        elif strategy == "margin":
            def cond(i, logits):
                margin = (logits[:, 1] - logits[:, 0]).abs().item()
                if i in cands and margin >= threshold:
                    return "margin_threshold"
                return None
        else:
            raise ValueError(f"未知动态策略: {strategy!r}")

        return self._run_loop(input_ids, attention_mask, token_type_ids, cond,
                              evaluate_layers=eval_layers)

    # ------------------------------------------------------------------ #
    # v2：active-set batching（保持样本索引映射，恢复原顺序）
    # ------------------------------------------------------------------ #
    def run_active_set(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        strategy: str = "max_prob",
        threshold: float = 0.9,
        candidate_layers: Optional[Sequence[int]] = None,
    ) -> Tuple[ExitResult, List[str]]:
        """active-set 动态退出：已退出样本从 batch 移除，剩余继续下一层。

        返回 (ExitResult, exit_reasons)。ExitResult.logits 等已按**原输入顺序**排列。
        """
        cands = sorted(set(candidate_layers or self.candidate_layers))
        n = input_ids.shape[0]
        device = input_ids.device

        active = list(range(n))                     # 仍活跃的原始样本索引
        logits_out = torch.zeros(n, 2, dtype=torch.float32, device=device)
        reasons_out = [""] * n

        self.reset_counters()
        t0 = time.perf_counter()

        # embeddings 只算一次；h 随层前进，并在每次退出后按活跃子集切片，
        # 从而既省后续层计算，又保证已计算层的结果不被重算。
        embeds = self.bert.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        h = embeds
        max_exec = 0
        max_exit = 0          # 任意样本实际退出的最大层
        for layer_idx in range(self.fallback_layer + 1):
            if not active:
                break
            mask_sub = attention_mask[active] if attention_mask is not None else None
            ext_mask = _build_extended_mask(self.bert, h, mask_sub)
            h = self.bert.encoder.layer[layer_idx](h, attention_mask=ext_mask)
            max_exec = layer_idx + 1
            pooled = self._pool(h, mask_sub)
            logits = self._head_logits(layer_idx, pooled)
            if logits is None:
                continue  # head 在该层不适用
            still_active: List[int] = []
            kept_indices: List[int] = []
            exited_here = False
            for j, orig_idx in enumerate(active):
                if strategy == "max_prob":
                    met = logits[j].softmax(dim=-1).max().item() >= threshold
                elif strategy == "margin":
                    met = (logits[j][1] - logits[j][0]).abs().item() >= threshold
                else:
                    raise ValueError(f"未知策略: {strategy!r}")
                if layer_idx in cands and met:
                    logits_out[orig_idx] = logits[j].detach()
                    reasons_out[orig_idx] = f"{strategy}_threshold"
                    exited_here = True
                elif layer_idx == self.fallback_layer:
                    logits_out[orig_idx] = logits[j].detach()
                    reasons_out[orig_idx] = "fallback"
                    exited_here = True
                else:
                    still_active.append(orig_idx)
                    kept_indices.append(j)
            if exited_here:
                max_exit = layer_idx
            if kept_indices:
                h = h[kept_indices]
            active = still_active

        latency_ms = (time.perf_counter() - t0) * 1000.0
        probs_out = torch.softmax(logits_out, dim=-1)
        result = ExitResult(
            logits=logits_out,
            probabilities=probs_out,
            exit_layer=max_exit,
            executed_layer_count=max_exec,
            exit_reason="active_set",
            latency_ms=latency_ms,
            layer_call_counts=self.layer_call_counts(),
        )
        return result, reasons_out

    # ------------------------------------------------------------------ #
    def full_forward_baseline(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """普通完整前向（全部 12 层），用于最后层等价性对照。"""
        with torch.no_grad():
            return self.heads_model(input_ids=input_ids, attention_mask=attention_mask,
                                    token_type_ids=token_type_ids).results[self.head_type][:, -1, :]


def build_early_exit_engine(
    base_model_dir: str,
    checkpoint_path: str,
    head_type: str = "copied_layer_heads",
    pooling: str = "masked_mean",
    heads_enabled: Optional[Sequence[str]] = None,
    model_hash: Optional[str] = None,
    device: str = "cpu",
) -> EarlyExitEngine:
    """装配 Early-Exit 引擎（加载 backbone + heads 模型）。"""
    from .heads import build_layer_heads_model
    if heads_enabled is None:
        heads_enabled = ["shared_frozen_head", "copied_layer_heads",
                         "random_layer_heads", "normalized_layer_heads"]
    heads_model = build_layer_heads_model(
        base_model_dir, checkpoint_path, heads_enabled=heads_enabled, pooling=pooling,
        pooling_confirmed=False, model_hash=model_hash, device=device,
    ).eval()
    engine = EarlyExitEngine(heads_model.bert, heads_model, head_type=head_type,
                             pooling=pooling)
    return engine.to(device=device)
