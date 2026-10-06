"""Plots for the report (static PNGs; every plot has a CSV table with the same numbers next to it)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical palette (fixed order) + chart chrome.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e6e5e0"
MARKERS = ["o", "s", "^", "D"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "lines.linewidth": 2,
})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  wrote {path}")


def plot_ladder(rows, path):
    """Small multiples: one horizontal bar chart per metric, one bar per ladder step."""
    names = [r["step"] for r in rows][::-1]
    metrics = ["R@50", "NDCG@50", "MRR"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.45 * len(rows) + 1.6), sharey=True)
    for ax, m in zip(axes, metrics):
        vals = [r[m] for r in rows][::-1]
        colors = [MUTED if n.startswith("x") else (ORANGE if "BM25" in n else BLUE) for n in names]
        ax.barh(names, vals, color=colors, height=0.6, edgecolor=SURFACE, linewidth=2)
        for y, v in enumerate(vals):
            ax.text(v, y, f" {v:.3f}", va="center", fontsize=8.5, color=INK2)
        ax.set_title(m)
        ax.set_xlim(0, max(vals) * 1.22)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Ablation ladder, 500 held-out test papers (blue = cosine ladder, orange = BM25, grey = extras)",
                 x=0.01, ha="left", fontsize=10, color=INK2)
    _save(fig, path)


def plot_recall_at_k(rows, path):
    """Recall@k curves for four key systems."""
    pick = ["1 baseline", "3 + year filter", "5 + co-citation", "6 BM25"]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ks = [5, 10, 20, 50]
    for i, p in enumerate(pick):
        r = next(r for r in rows if r["step"].startswith(p))
        ys = [r[f"R@{k}"] for k in ks]
        ax.plot(ks, ys, color=[BLUE, AQUA, ORANGE, YELLOW][i], marker=MARKERS[i], markersize=6,
                label=r["step"][2:])
        ax.annotate(r["step"][2:], (ks[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK2)
    ax.set_xticks(ks)
    ax.set_xlabel("k (results inspected)")
    ax.set_ylabel("recall@k")
    ax.set_xlim(3, 85)
    ax.set_ylim(0, None)
    ax.set_title("Recall@k: share of the paper's true references found")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    _save(fig, path)


def _tradeoff_axes(rows, xkey, xlabel, title, highlight_first=False):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    xs = list(range(len(rows)))
    labels = [str(r[xkey]) for r in rows]
    a1.plot(xs, [r["ms_per_query"] for r in rows], color=BLUE, marker="o", markersize=6)
    a1.set_title("Speed: median ms per query")
    a1.set_ylabel("ms / query")
    a2.plot(xs, [r["R@50"] for r in rows], color=BLUE, marker="o", markersize=6, label="recall@50")
    a2.plot(xs, [r["NDCG@50"] for r in rows], color=ORANGE, marker="s", markersize=6, label="NDCG@50")
    for key, c in (("R@50", BLUE), ("NDCG@50", ORANGE)):
        a2.annotate({"R@50": "recall@50", "NDCG@50": "NDCG@50"}[key], (xs[-1], rows[-1][key]),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    a2.set_title("Quality")
    a2.set_ylim(0, None)
    a2.legend(frameon=False, fontsize=8.5)
    for a in (a1, a2):
        a.set_xticks(xs, labels, rotation=30 if len(labels) > 7 else 0)
        a.set_xlabel(xlabel)
        a.set_ylim(0, None)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    return fig


def plot_tradeoff(rows, xkey, xlabel, path, title):
    _save(_tradeoff_axes(rows, xkey, xlabel, title), path)


def plot_champion(rows, path):
    """Champion lists. Left: recall@50 vs postings touched (machine-independent cost).
    Right: wall-clock text-scoring time vs list size r. Exhaustive scoring = dashed reference."""
    ex = next(r for r in rows if r["variant"] == "exhaustive")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    for i, (variant, label) in enumerate((("approx", "champion postings only (approximate)"),
                                          ("rescored", "champion union, exact rescoring"))):
        rs = [r for r in rows if r["variant"] == variant]
        c, mk = (BLUE, "o") if i == 0 else (ORANGE, "s")
        a1.plot([r["postings_touched"] for r in rs], [r["R@50"] for r in rs], color=c, marker=mk, markersize=6, label=label)
        for r in rs:
            if r["champion_r"] in (5, 50, 500):
                a1.annotate(f"r={r['champion_r']}", (r["postings_touched"], r["R@50"]), xytext=(4, -12 if i else 6),
                            textcoords="offset points", fontsize=8, color=INK2)
        a2.plot([str(r["champion_r"]) for r in rs], [r["text_scoring_ms"] for r in rs], color=c, marker=mk,
                markersize=6, label=label)
    a1.scatter([ex["postings_touched"]], [ex["R@50"]], color=INK, marker="*", s=120, zorder=3, label="exhaustive")
    a1.axhline(ex["R@50"], color=MUTED, linestyle="--", linewidth=1)
    a1.set_xscale("log")
    a1.set_xlabel("postings touched per query (log scale)")
    a1.set_ylabel("recall@50")
    a1.set_ylim(0, None)
    a1.set_title("Recall vs work done")
    a1.legend(frameon=False, fontsize=8.5, loc="lower right")
    a2.axhline(ex["text_scoring_ms"], color=MUTED, linestyle="--", linewidth=1)
    a2.annotate(f"exhaustive {ex['text_scoring_ms']:.1f} ms", (0, ex["text_scoring_ms"]), xytext=(0, 5),
                textcoords="offset points", fontsize=8.5, color=INK2)
    a2.set_xlabel("champion list size r (docs per term)")
    a2.set_ylabel("ms per query (text scoring only)")
    a2.set_ylim(0, None)
    a2.set_title("Wall-clock time (numpy, 48k docs)")
    a2.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Champion lists: speed vs recall, 500 test queries", x=0.01, ha="left", fontsize=11, fontweight="bold")
    _save(fig, path)


def plot_faithfulness(rows, path):
    """Share of explained results pushed out of the top 10 when we delete (a) the explanation
    terms vs (b) the same number of random other matching terms."""
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    configs = [r["config"] for r in rows]
    y = range(len(rows))
    h = 0.36
    a = [r["pct_dropped_out_of_top10_explained"] for r in rows]
    b = [r["pct_dropped_out_of_top10_random"] for r in rows]
    ax.barh([i + h / 2 for i in y], a, height=h, color=BLUE, label="remove top-3 explanation terms",
            edgecolor=SURFACE, linewidth=2)
    ax.barh([i - h / 2 for i in y], b, height=h, color=ORANGE, label="remove 3 random matching terms (control)",
            edgecolor=SURFACE, linewidth=2)
    for i in y:
        ax.text(a[i], i + h / 2, f" {a[i]:.0%}", va="center", fontsize=8.5, color=INK2)
        ax.text(b[i], i - h / 2, f" {b[i]:.0%}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(list(y), configs)
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("share of top-10 results that fall out of the top 10")
    ax.set_title("Faithfulness: do the explanation terms really drive the rank?")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.45, -0.18), ncol=2)
    _save(fig, path)
