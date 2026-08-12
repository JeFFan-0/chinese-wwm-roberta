#!/usr/bin/env python
"""P5 无数据固定层性能基准。

测量矩阵：seq_len {32,64,128,256} × batch {1,8,16,32} × 固定退出层 {2,4,6,8,10,11}
+ 完整模型基线（同一进程、同一配置测量）。

方法（§7.2）：每配置预热 >=10 次、正式 >=50-100 次；GPU 测量前后 cuda.synchronize；
eval + no_grad；固定随机输入（固定 seed）；不混入加载/tokenizer 时间。

结果只能表述为：
    "如果未来证明可以在 layer k 退出，则在硬件 H、batch B、长度 L 下，测得速度为 X。"
不能表述为 "layer k 已足够完成情绪任务"。

输出：
    reports/tables/fixed_exit_benchmark.csv
    reports/figures/latency_exit_layer_*.png
    reports/figures/speedup_vs_theory_*.png

用法:
    conda activate 26intern
    python scripts/benchmark_fixed_exit.py [--quick] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import load_yaml_config  # noqa: E402
from src.early_exit import build_early_exit_engine  # noqa: E402
from src.heads import build_layer_heads_model  # noqa: E402

# dataviz 参考调色板（categorical 固定顺序，不用循环色）
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK = "#0b0b0b"
SECONDARY = "#52514e"
GRID = "#e1e0d9"

ALL_HEADS = ["original_final_head", "shared_frozen_head", "copied_layer_heads",
             "random_layer_heads", "normalized_layer_heads"]


def _setup_cn_font() -> None:
    """注册中文字体（与 check.ipynb 一致），失败时退回英文标签。"""
    candidates = [
        "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
        "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc",
        "/usr/share/fonts/wenquanyi/wqy-microhei/wqy-microhei.ttc",
    ]
    for c in candidates:
        if os.path.isfile(c):
            from matplotlib import font_manager
            font_manager.fontManager.addfont(c)
            name = font_manager.FontProperties(fname=c).get_name()
            plt.rcParams["font.family"] = [name, "DejaVu Sans"]
            return
    plt.rcParams["font.family"] = ["DejaVu Sans"]


def _measure(fn, device: str, warmup: int, runs: int) -> tuple:
    """预热 + 测量，返回 (latencies_ms, peak_allocated_bytes, p50, p95, mean)。"""
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        lat = []
        for _ in range(runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000.0)
        peak = torch.cuda.max_memory_allocated()
    else:
        for _ in range(warmup):
            fn()
        lat = []
        for _ in range(runs):
            t0 = time.perf_counter()
            fn()
            lat.append((time.perf_counter() - t0) * 1000.0)
        peak = 0
    lat_sorted = sorted(lat)
    p50 = statistics.median(lat_sorted)
    p95 = lat_sorted[min(len(lat_sorted) - 1, int(0.95 * len(lat_sorted)))]
    return lat_sorted, peak, p50, p95, statistics.mean(lat_sorted)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quick", action="store_true", help="减少测量次数（smoke 用）")
    parser.add_argument("--lengths", default=None, help="覆盖 seq 长度，逗号分隔")
    parser.add_argument("--batches", default=None, help="覆盖 batch，逗号分隔")
    args = parser.parse_args()

    _setup_cn_font()
    bench = load_yaml_config(os.path.join(ROOT, "configs", "benchmark.yaml"))["benchmark"]
    model_cfg = load_yaml_config(os.path.join(ROOT, "configs", "model.yaml"))
    ckpt = os.path.join(ROOT, model_cfg["paths"]["checkpoint"])
    base_dir = os.path.join(ROOT, model_cfg["paths"]["base_model_dir"])

    lengths = [int(x) for x in (args.lengths or "").split(",") if x] or bench["sequence_lengths"]
    batches = [int(x) for x in (args.batches or "").split(",") if x] or bench["batch_sizes"]
    exit_layers = bench["exit_layers"]
    dtype_name = bench["dtypes"][0]
    warmup = 10 if not args.quick else 2
    runs = 50 if not args.quick else 3

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    is_cuda = device.startswith("cuda") and torch.cuda.is_available()

    print("=" * 88)
    print("P5 固定层性能基准")
    print(f"device={device}  lengths={lengths}  batches={batches}  exits={exit_layers}")
    print(f"warmup={warmup}  runs={runs}  dtype={dtype_name}")
    print("=" * 88)

    # ---- 构建引擎（CUDA）----
    t0 = time.time()
    engine = build_early_exit_engine(
        base_dir, ckpt, head_type="copied_layer_heads", pooling="masked_mean",
        device=device,
    ).eval()
    heads_model = engine.heads_model
    print(f"模型加载用时 {time.time() - t0:.1f}s")

    rows = []
    gen = torch.Generator().manual_seed(1234)
    vocab = heads_model.bert.config.vocab_size

    def full_forward(input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = heads_model(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids)
        return out.results["copied_layer_heads"][:, -1, :]

    for seq_len in lengths:
        for batch in batches:
            # 固定随机输入（seed 固定 → 可复现）
            g = torch.Generator().manual_seed(1000 * seq_len + batch)
            input_ids = torch.randint(1, vocab, (batch, seq_len), generator=g)
            attention_mask = torch.ones_like(input_ids)
            token_type_ids = torch.zeros_like(input_ids)
            inp = {
                "input_ids": input_ids.to(device),
                "attention_mask": attention_mask.to(device),
                "token_type_ids": token_type_ids.to(device),
            }

            # 1) 完整模型基线（同一进程、同一配置）
            lat, peak, p50, p95, mean = _measure(
                lambda: full_forward(**inp), device, warmup, runs)
            samples = batch * 1000.0 / p50
            rows.append(_row("model_full", 12, seq_len, batch, p50, p95, mean, samples,
                             peak, device, dtype_name, executed=12))

            # 2) 固定层退出
            for k in exit_layers:
                lat, peak, p50, p95, mean = _measure(
                    lambda k=k: engine.run_fixed(**inp, exit_layer=k), device, warmup, runs)
                samples = batch * 1000.0 / p50
                rows.append(_row("engine_exit", k, seq_len, batch, p50, p95, mean, samples,
                                 peak, device, dtype_name, executed=k + 1))

    df = pd.DataFrame(rows)

    # ---- 派生指标：相对完整 12 层基线（engine run_fixed(11)，同一代码路径）----
    base_rows = df[(df["path"] == "engine_exit") & (df["exit_layer"] == 11)]
    base_map = {(r.seq_len, r.batch): r.p50_latency_ms for r in base_rows.itertuples()}
    def derive(r):
        if r["path"] == "model_full":
            # model_full（全头全层）仅作参考，其与 engine_exit/11 的差为头评估开销
            base11 = base_map.get((r["seq_len"], r["batch"]), float("nan"))
            r = r.copy()
            r["ideal_speedup"] = float("nan")
            r["measured_speedup"] = round(base11 / r["p50_latency_ms"], 3) if base11 and r["p50_latency_ms"] else float("nan")
            r["speedup_gap"] = float("nan")
            r["theoretical_layer_savings"] = 0.0
            return r, None
        base = base_map[(r["seq_len"], r["batch"])]
        ideal = 12.0 / r["exit_layer"]  # 理论层数加速比（12/(k+1)）
        measured = base / r["p50_latency_ms"] if r["p50_latency_ms"] > 0 else float("nan")
        r = r.copy()
        r["ideal_speedup"] = round(ideal, 3)
        r["measured_speedup"] = round(measured, 3)
        r["speedup_gap"] = round(ideal - measured, 3)
        r["theoretical_layer_savings"] = round(1.0 - r["exit_layer"] / 12.0, 4)
        return r, None
    out_rows = [derive(r)[0] for _, r in df.iterrows()]
    df = pd.DataFrame(out_rows)

    os.makedirs(os.path.join(ROOT, "reports", "tables"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "reports", "figures"), exist_ok=True)
    csv_path = os.path.join(ROOT, "reports", "tables", "fixed_exit_benchmark.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n已写入 {csv_path}（{len(df)} 行）")

    # ---- 图 ----
    _plot_latency(df, exit_layers)
    _plot_speedup(df, exit_layers)

    # ---- 环境说明 ----
    env = {
        "hardware": torch.cuda.get_device_name(0) if is_cuda else platform.machine(),
        "device": device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda": torch.version.cuda,
        "warmup_runs": warmup,
        "measure_runs": runs,
        "dtype": dtype_name,
        "pooling": "masked_mean",
        "head_type": "copied_layer_heads",
        "input_source": "fixed random ids, attention_mask=all-ones, seed fixed",
        "disclaimer": ("结果只表述为'若未来证明可在 layer k 退出，则在硬件/批/长度下测得速度 X'，"
                       "不代表 layer k 已足够完成情绪任务。"),
    }
    env_path = os.path.join(ROOT, "reports", "tables", "benchmark_env.json")
    with open(env_path, "w", encoding="utf-8") as f:
        json.dump(env, f, ensure_ascii=False, indent=2)
    print(f"已写入 {env_path}")

    print("\n结果: OK")
    return 0


def _row(path, exit_layer, seq_len, batch, p50, p95, mean, samples, peak, device,
         dtype_name, executed):
    return {
        "path": path, "exit_layer": exit_layer, "executed_layer_count": executed,
        "seq_len": seq_len, "batch": batch, "dtype": dtype_name,
        "p50_latency_ms": round(p50, 4), "p95_latency_ms": round(p95, 4),
        "mean_latency_ms": round(mean, 4), "samples_per_s": round(samples, 2),
        "peak_allocated_bytes": int(peak), "device": device,
    }


# --------------------------------------------------------------------------- #
# 图
# --------------------------------------------------------------------------- #
def _plot_latency(df, exit_layers):
    """p50 latency vs exit_layer：固定 batch 按长度分线；固定长度按 batch 分线。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, fix, vary, col in (
        (axes[0], "batch", "seq_len", "长度"),
        (axes[1], "seq_len", "batch", "batch"),
    ):
        ax.set_facecolor("#fcfcfb")
        for idx, key in enumerate(sorted(df[fix].unique())):
            sub = df[(df[fix] == key) & (df["path"] == "engine_exit")]
            series = sub.groupby("exit_layer")["p50_latency_ms"].first()
            color = CAT[idx % len(CAT)]
            ax.plot(series.index, series.values, marker="o", ms=5, lw=2, color=color,
                    label=f"{col}={key}")
        # 完整模型基线
        base = df[df["path"] == "model_full"].groupby(fix)["p50_latency_ms"].first()
        ax.axhline(base.mean(), color=INK, ls="--", lw=1.5, label="完整 12 层基线(均值)")
        ax.set_xlabel("固定退出层 (encoder_layer)")
        ax.set_ylabel("p50 latency (ms)")
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
        ax.set_title(f"固定退出层 p50 延迟（按 {col} 分组）")
    fig.tight_layout()
    out = os.path.join(ROOT, "reports", "figures", "latency_exit_layer.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"图: {out}")


def _plot_speedup(df, exit_layers):
    """代表配置（len=128, batch=16）的理论 vs 实测加速比。"""
    sub = df[(df["seq_len"] == 128) & (df["batch"] == 16) & (df["path"] == "engine_exit")]
    if sub.empty:
        sub = df[(df["path"] == "engine_exit")].groupby("exit_layer").first().reset_index()
    sub = sub.sort_values("exit_layer")
    x = sub["exit_layer"]
    ideal = sub["ideal_speedup"]
    measured = sub["measured_speedup"]
    gap = sub["speedup_gap"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_facecolor("#fcfcfb")
    width = 0.35
    x_pos = range(len(x))
    ax.bar([p - width / 2 for p in x_pos], ideal, width, color="#1baf7a",
           label="理论层数加速比 12/(k+1)")
    ax.bar([p + width / 2 for p in x_pos], measured, width, color="#2a78d6",
           label="实测加速比 (完整基线/退出)")
    # 差距标注
    for p, (ex, g) in enumerate(zip(x, gap)):
        ax.text(p, measured.iloc[p] + 0.05, f"Δ{g:.2f}", ha="center", fontsize=8, color=SECONDARY)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels([f"{int(k)}" for k in x])
    ax.set_xlabel("固定退出层 (encoder_layer)")
    ax.set_ylabel("相对完整 12 层的速度比 (×)")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, frameon=False)
    ax.set_title("理论层数节省 vs 实测加速（len=128, batch=16 代表配置）")
    fig.tight_layout()
    out = os.path.join(ROOT, "reports", "figures", "speedup_vs_theory.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"图: {out}")


if __name__ == "__main__":
    sys.exit(main())
