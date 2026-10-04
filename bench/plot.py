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
OURS = "fa opt (ours)"
OURS_FP16 = "fa fp16-acc (ours)"
OURS_BASE = "fa baseline (ours)"
FA2 = "sdpa-flash (FA2)"
STYLE = {
    OURS_FP16: dict(color="darkred", marker="*", linewidth=2.5, markersize=10),
    OURS: dict(color="tab:red", marker="o", linewidth=2.5),
    OURS_BASE: dict(color="salmon", marker="o", linestyle="--"),
    FA2: dict(color="tab:blue", marker="s"),
    "sdpa-cudnn": dict(color="tab:green", marker="^"),
    "sdpa-efficient": dict(color="tab:purple", marker="v"),
    "torch naive": dict(color="tab:gray", marker="x"),
}


def plot(df: pd.DataFrame, gpu: str) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.8), sharey=True)
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
    # One shared legend under the panels, so it never covers a line.
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(f"Attention forward, fp16, B*N = 16k, H*d = 2048 — {gpu}")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = ROOT / "docs" / "img" / f"{gpu.replace(' ', '_')}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def table(df: pd.DataFrame) -> str:
    ok = df[df.status == "ok"]
    piv = ok.pivot_table(index=["d", "causal", "N"], columns="impl", values="tflops")
    impls = [c for c in STYLE if c in piv.columns]
    lines = ["| d | causal | N | " + " | ".join(impls) + " | opt / FA2 | fp16-acc / FA2 |",
             "|---|---|---|" + "---|" * len(impls) + "---|---|"]
    for (d, causal, n), row in piv.iterrows():
        cells = [f"{row[c]:.1f}" if pd.notna(row[c]) else "OOM" for c in impls]
        def ratio(name):
            if name in row and FA2 in row and pd.notna(row.get(name)) and pd.notna(row.get(FA2)):
                return f"{row[name] / row[FA2]:.2f}x"
            return "-"

        cells = [f"**{c}**" if impls[k] in (OURS, OURS_FP16) else c for k, c in enumerate(cells)]
        lines.append(f"| {d} | {causal} | {n} | " + " | ".join(cells)
                     + f" | {ratio(OURS)} | {ratio(OURS_FP16)} |")
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
