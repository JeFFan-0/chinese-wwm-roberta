"""对比微调 ckpt 与官方底座：逐层、逐组件 余弦相似度 + 相对 L2 位移

用法:
    python compare_layers.py            # 用 26intern 环境: /home/intern_fjq_2026/miniconda3/envs/26intern/bin/python compare_layers.py

输出两张表:
    1. 余弦相似度   —— 微调后 vs 底座，同一层同一组件的权重方向变化（越接近 1 = 几乎没动）
    2. 相对 L2 位移 —— ||ΔW|| / ||W_base||（越大 = 动得越狠）
"""
import torch
import pandas as pd

BASE_DIR = "chinese-roberta-wwm-ext"
CKPT_PATH = "chinese-wwm-roberta.ckpt"
N_LAYERS = 12

comps = [
    "attention.self.query.weight",
    "attention.self.key.weight",
    "attention.self.value.weight",
    "attention.output.dense.weight",
    "intermediate.dense.weight",
    "output.dense.weight",
]


def cos(a, b):
    """余弦相似度：只看方向，不看幅度"""
    a, b = a.flatten().double(), b.flatten().double()
    return float(a @ b / (a.norm() * b.norm()))


def rel_shift(a, b):
    """相对 L2 位移：||ΔW|| / ||W_base||，同时体现方向和幅度"""
    a, b = a.flatten().double(), b.flatten().double()
    return float((a - b).norm() / b.norm())


def main():
    # 两份权重都 load 到 CPU
    base = torch.load(f"{BASE_DIR}/pytorch_model.bin", map_location="cpu")
    ft = torch.load(CKPT_PATH, map_location="cpu")

    # 若未来 ckpt 包了一层 "model_state_dict"，自动解开
    for d in (base, ft):
        if "model_state_dict" in d:
            d = d["model_state_dict"]

    # 键对齐检查
    base_keys = set(base)
    ft_keys = set(ft)
    only_ft = sorted(ft_keys - base_keys)          # 只在 ckpt 里（如 fc.*）
    only_base = sorted(base_keys - ft_keys)        # 只在底座里（如 cls.* MLM 头）
    print(f"[对齐检查] 底座 {len(base_keys)} 个键, ckpt {len(ft_keys)} 个键")
    if only_ft:
        print(f"  仅在 ckpt: {only_ft[:6]}{' ...' if len(only_ft) > 6 else ''}")
    if only_base:
        print(f"  仅底座(MLM头等): {only_base[:6]}{' ...' if len(only_base) > 6 else ''}")

    rows_cos, rows_shift = {}, {}
    for c in comps:
        rows_cos[c], rows_shift[c] = [], []
        for i in range(N_LAYERS):
            a = ft[f"bert.encoder.layer.{i}.{c}"]
            b = base[f"bert.encoder.layer.{i}.{c}"]
            rows_cos[c].append(round(cos(a, b), 6))
            rows_shift[c].append(round(rel_shift(a, b), 4))

    idx = [f"layer{i}" for i in range(N_LAYERS)]
    cos_sim = pd.DataFrame(rows_cos, index=idx)
    shift = pd.DataFrame(rows_shift, index=idx)

    print("\n===== 1. 余弦相似度（微调后 vs 官方底座）=====")
    print(cos_sim.to_string())
    print("\n===== 2. 相对 L2 位移 ||ΔW|| / ||W_base|| =====")
    print(shift.to_string())

    # 附加：embedding 与 pooler（若有）
    extra = {}
    for k in ["bert.embeddings.word_embeddings.weight",
              "bert.embeddings.position_embeddings.weight",
              "bert.embeddings.token_type_embeddings.weight",
              "bert.pooler.dense.weight"]:
        if k in ft and k in base:
            extra[k.split("bert.")[1]] = (round(cos(ft[k], base[k]), 3),
                                          round(rel_shift(ft[k], base[k]), 4))
    if extra:
        print("\n===== 3. Embedding / Pooler（余弦, 相对L2）=====")
        for name, (c, s) in extra.items():
            print(f"  {name:<42} cos={c:<6} rel_L2={s}")

    # 逐层平均，看变化趋势
    avg_cos = cos_sim.mean(axis=1)
    avg_shift = shift.mean(axis=1)
    print("\n===== 4. 逐层平均趋势 =====")
    trend = pd.DataFrame({"avg_cos": avg_cos.round(6), "avg_rel_L2": avg_shift.round(4)}, index=idx)
    print(trend.to_string())
    print("\n底层(0-3) vs 顶层(8-11):")
    print(f"  avg_cos    底层={avg_cos[:4].mean():.6f}  顶层={avg_cos[-4:].mean():.6f}")
    print(f"  avg_rel_L2 底层={avg_shift[:4].mean():.4f}  顶层={avg_shift[-4:].mean():.4f}")


if __name__ == "__main__":
    main()
