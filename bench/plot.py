"""Plot benchmark CSVs and print a Markdown table for the README.

    python bench/plot.py                       # every CSV in bench/results/
    python bench/plot.py bench/results/X.csv

Writes docs/img/<gpu>.png (2x2 grid: d in {64,128} x causal in {0,1}, TFLOPS vs N).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OURS = "fa (ours)"
FA2 = "sdpa-flash (FA2)"
STYLE = {
    OURS: dict(color="tab:red", marker="o", linewidth=2.5),
    FA2: dict(color="tab:blue", marker="s"),
    "sdpa-cudnn": dict(color="tab:green", marker="^"),
    "sdpa-efficient": dict(color="tab:purple", marker="v"),
    "torch naive": dict(color="tab:gray", marker="x"),
}


def plot(df: pd.DataFrame, gpu: str) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for (i, d), (j, causal) in [((i, d), (j, c)) for i, d in enumerate([64, 128]) for j, c in enumerate([0, 1])]:
        ax = axes[i][j]
        sub = df[(df.d == d) & (df.causal == causal) & (df.status == "ok")]
        for impl, g in sub.groupby("impl", sort=False):
            ax.plot(g.N, g.tflops, label=impl, **STYLE.get(impl, {}))
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(df.N.unique()), [str(n) for n in sorted(df.N.unique())])
        ax.set_title(f"head dim {d}, {'causal' if causal else 'non-causal'}")
        ax.set_xlabel("sequence length N")
        ax.set_ylabel("TFLOPS")
        ax.grid(alpha=0.3)
    axes[0][0].legend()
    fig.suptitle(f"Attention forward, fp16, B*N = 16k, H*d = 2048 — {gpu}")
    fig.tight_layout()
    out = ROOT / "docs" / "img" / f"{gpu.replace(' ', '_')}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def table(df: pd.DataFrame) -> str:
    ok = df[df.status == "ok"]
    piv = ok.pivot_table(index=["d", "causal", "N"], columns="impl", values="tflops")
    impls = [c for c in STYLE if c in piv.columns]
    lines = ["| d | causal | N | " + " | ".join(impls) + " | ours / FA2 |",
             "|---|---|---|" + "---|" * len(impls) + "---|"]
    for (d, causal, n), row in piv.iterrows():
        cells = [f"{row[c]:.1f}" if pd.notna(row[c]) else "OOM" for c in impls]
        ratio = row.get(OURS) / row.get(FA2) if FA2 in row and pd.notna(row.get(FA2)) else float("nan")
        cells = [f"**{c}**" if impls[k] == OURS else c for k, c in enumerate(cells)]
        lines.append(f"| {d} | {causal} | {n} | " + " | ".join(cells) + f" | {ratio:.2f}x |")
    return "\n".join(lines)


def main():
    paths = [Path(p) for p in sys.argv[1:]] or sorted((ROOT / "bench" / "results").glob("*.csv"))
    for path in paths:
        df = pd.read_csv(path)
        gpu = df.gpu.iloc[0]
        print(f"### {gpu}\n\nTFLOPS (higher is better), from `{path.relative_to(ROOT)}`\n")
        print(table(df))
        print(f"\nplot: {plot(df, gpu).relative_to(ROOT)}\n")


if __name__ == "__main__":
    main()
