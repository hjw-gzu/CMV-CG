"""Plot the numerical evidence for Fig. 3 and Supplementary Figs. S1–S7."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "results/figures"
DATASETS = [("coco", "MSCOCO"), ("flickr", "Flickr30k")]
COLORS = ["#32719a", "#bd6952", "#4e836b", "#826694"]


def read(path):
    with (gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")) as f:
        return json.load(f)


def plot_selectors(data):
    """Draw the three main-figure panels using the archived adjusted interval bounds."""
    from matplotlib.patches import Patch
    BLUE, PEACH = "#4E83AD", "#E6B39C"
    EDGES = ["#355B77", "#A2745F"]
    panels = [
        ('Single scores',
         ['original_clip_only', 'multiview_clip_only', 'probability_only', 'consensus_only'],
         ['Original\nCLIP', 'Multi-view\nCLIP', 'Probability', 'Consensus']),
        ('Equal-weight combinations',
         ['clip_consensus_equal', 'clip_probability_equal', 'all_equal'],
         ['CLIP +\nconsensus', 'CLIP +\nprobability', 'Four-score\nequal']),
        ('Full-score variants',
         ['without_original', 'oarr_fixed'],
         ['w/o original\nterm', 'Ours']),
    ]
    fig = plt.figure(figsize=(180 / 25.4, 132 / 25.4))
    grid = fig.add_gridspec(2, 2, left=.10, right=.98, bottom=.13,
                          top=.89, wspace=.28, hspace=.64)
    axes = [fig.add_subplot(grid[0, :]),
            fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    plotted = []
    for ax, (title, keys, labels), letter in zip(axes, panels, 'abc'):
        x = np.arange(len(keys))
        # Keep physical bar widths consistent despite the full-width upper panel.
        xspan = len(keys) + .16
        plot_width_mm = ax.get_position().width * fig.get_figwidth() * 25.4
        bar_width = 7.0 * xspan / plot_width_mm
        for offset, ds, color, edge in zip([-.53 * bar_width, .53 * bar_width], ['coco', 'flickr'],
                                           [BLUE, PEACH], EDGES):
            source = {r['selector']: r for r in data['cross_scorer'][ds]}
            rows = [source[k] for k in keys]
            y = np.array([r['delta'] for r in rows])
            bounds = np.array([r['adjusted'] for r in rows])
            assert np.isfinite(y).all() and np.isfinite(bounds).all()
            assert np.all(bounds[:, 0] <= y) and np.all(bounds[:, 1] >= y)
            ax.bar(x + offset, y, width=bar_width, color=color, edgecolor=edge,
                   linewidth=.45, zorder=3)
            ax.errorbar(x + offset, y,
                        yerr=[y - bounds[:, 0], bounds[:, 1] - y],
                        fmt='none', ecolor='#39444D', elinewidth=.72,
                        capsize=2.1, capthick=.72, zorder=4)
            plotted.extend({'dataset':ds, 'selector':k, 'delta':r['delta'],
                            'interval':r['adjusted']} for k, r in zip(keys, rows))
        ax.set_xlim(-.58, len(keys) - .42)
        ax.set_ylim(-2.2, 8.35)
        ax.set_yticks([-2, 0, 2, 4, 6, 8])
        ax.set_xticks(x, labels)
        ax.tick_params(axis='x', pad=7)
        ax.tick_params(axis='y', left=True, labelleft=True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.set_ylabel('CIDEr gain', labelpad=6)
        ax.set_xlabel('Scoring rule', labelpad=7)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#E4E8EB', linewidth=.42)
        ax.axhline(0, color='#68747D', linewidth=.75, zorder=2)
        ax.set_title(title, loc='left', pad=9)
        ax.annotate(letter, xy=(0, 1), xycoords='axes fraction',
                    xytext=(-13, 8), textcoords='offset points',
                    fontsize=10, fontweight='bold', va='bottom', annotation_clip=False)
    axes[2].get_xticklabels()[-1].set_fontweight('bold')
    fig.legend(handles=[Patch(facecolor=BLUE, edgecolor=EDGES[0], linewidth=.45, label='COCO'),
                        Patch(facecolor=PEACH, edgecolor=EDGES[1], linewidth=.45, label='Flickr30k')],
               loc='upper center', bbox_to_anchor=(.54, .995), ncol=2,
               handlelength=1.45, handletextpad=.55, columnspacing=2.2)

    assert len(plotted) == 18
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True, help="New directory for vector PDFs")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    f = read(ROOT/"selector-rotation-latency-pairs.json.gz")
    ext = read(ROOT/"near-budget-transfer.json")
    def save(fig, name):
        fig.savefig(a.output / (name + ".pdf"), bbox_inches="tight", pad_inches=.07)
        plt.close(fig)

    save(plot_selectors(f), "Fig3-selector-comparison")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5), constrained_layout=True)
    for ax, metric in zip(axs, ["CIDEr", "SPICE"]):
        for y, group in enumerate(ext["external"]):
            row = next(v for v in group["rows"] if v["metric"] == metric)
            ax.plot(row["interval"], [y, y], color=COLORS[0])
            ax.plot(row["delta"], y, "o", color=COLORS[0])
        ax.set_yticks(range(len(ext["external"])), [r["label"] for r in ext["external"]])
        ax.axvline(0, lw=.7, color="0.5")
        ax.set_xlabel(f"Paired {metric} difference")
    save(fig, "FigS1-transfer")
    strengths = read(ROOT/"rotation-strength.json")
    fig, axs = plt.subplots(2, 2, figsize=(6.6, 4.5), constrained_layout=True)
    for c, (_, title) in enumerate(DATASETS):
        data = sorted((v for v in strengths if v["dataset"] == title), key=lambda v:v["strength"])
        for row, metric in enumerate(("CIDEr", "SPICE")):
            ax = axs[row, c]
            ax.plot([v["strength"] for v in data], [v[metric] for v in data], "o-", color=COLORS[c])
            ax.set(title=title, ylabel=metric + " × 100", xlabel="Rotation multiplier")
    save(fig, "FigS2-auxiliary-rotation")
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.9), constrained_layout=True)
    for ax, dataset in zip(axs, read(ROOT/"weight-sensitivity.json")):
        data = dataset["results"]
        for color, coord in zip(COLORS, ("original_clip", "multiview_clip", "language_probability", "caption_consensus")):
            terms = sorted([(0., 0.)]+[(v["amplitude"]*v["direction"], v["delta_CIDEr"]) for v in data if v["coordinate"] == coord])
            ax.plot(*zip(*terms), ".-", label=coord.replace("_", " "), color=color)
        ax.set(title=dataset["dataset"], ylabel="Validation CIDEr change", xlabel="Relative weight change")
    axs[-1].legend(fontsize=7)
    save(fig, "FigS3-selector-weights")
    fig, axs = plt.subplots(2, 2, figsize=(7, 4.6), constrained_layout=True)
    for c, (key, title) in enumerate(DATASETS):
        for row, metric in enumerate(("CIDEr", "SPICE")):
            ax = axs[row, c]
            for i, rule in enumerate(("fixed_full", "fixed_removed")):
                pts = sorted((v["angle"],v[metric]) for v in f["rotation"][key] if v["selector"] == rule)
                zero = dict(pts)[0]
                ax.plot([x for x,y in pts], [y-zero for x,y in pts], ".-", color=COLORS[i],
                        label=("Full", "Original term removed")[i])
            ax.set(title=title, ylabel=metric+" change", xlabel="Input rotation (degrees)")
    axs[0,0].legend(fontsize=7)
    save(fig, "FigS4-input-rotation")
    fig, axs = plt.subplots(1, 2, figsize=(7.3, 3.1), constrained_layout=True)
    for ax, (key, title) in zip(axs, DATASETS):
        arms = f["latency"][key]["arms"]
        labels = ["single_beam_20", "multiview"]
        bottom = np.zeros(2)
        for term in ("load_s", "encode_s", "bridge_s", "generation_s", "rerank_s", "other_s"):
            values = [arms[m][term] for m in labels]
            ax.bar(["Single-view", "Five-view"], values, bottom=bottom, label=term[:-2])
            bottom += values
        ax.set(title=title, ylabel="Mean seconds per image")
    axs[1].legend(fontsize=7)
    save(fig, "FigS5-latency")
    with (ROOT/"quality_values.csv").open(encoding="utf-8-sig") as sf, (ROOT/"paired_latency.csv").open(encoding="utf-8-sig") as tf:
        scores, times = list(csv.DictReader(sf)), list(csv.DictReader(tf))
    fig, axs = plt.subplots(2, 2, figsize=(8, 4.8), constrained_layout=True)
    for col, ds in enumerate(("COCO", "Flickr30k")):
        q = [r for r in scores if r["dataset"]==ds and r["metric"] in ("Bleu_1","Bleu_4","METEOR","ROUGE_L","CIDEr","SPICE")]
        ax = axs[0, col]
        for i, r in enumerate(q):
            ax.bar(i, float(r["delta"]), color=COLORS[col])
            if r["lower"]:
                ax.plot([i,i], [float(r["lower"]),float(r["upper"])], color="black", lw=1)
        ax.set_xticks(range(len(q)), [v["metric"] for v in q], rotation=35, ha="right")
        ax.set(title=ds, ylabel="Five-view minus single-view")
        t = [r for r in times if r["dataset"]==ds]
        axs[1,col].scatter([float(r["single_s"]) for r in t], [float(r["ours_s"]) for r in t], s=2, alpha=.27, color=COLORS[col], rasterized=True)
        axs[1,col].set(xlabel="Single-view seconds", ylabel="Five-view seconds")
    save(fig, "FigS6-near-budget")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
    for ax,(key,title) in zip(axs,DATASETS):
        pairs=f["paired_images"][key]
        ax.scatter(pairs["single"],pairs["multi"],s=3,alpha=.3,color=COLORS[0],rasterized=True)
        hi=max(max(pairs["single"]),max(pairs["multi"]))
        ax.plot([0,hi],[0,hi], "--", color="0.5", lw=.8)
        ax.set(title=title, xlabel="Single-view CIDEr × 100", ylabel="Five-view CIDEr × 100")
    save(fig, "FigS7-paired-images")


if __name__ == "__main__":
    main()
