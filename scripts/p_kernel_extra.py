"""Extra panels for the k-Nearest Neighbors page: data illustration + grid search."""
import math
from dio import *
from l_common import *

TRAIN = [  # day, RSI, EMA gap %, label
    (1, 72, 1.8, "Up"), (2, 65, 1.1, "Up"), (3, 58, 0.6, "Up"), (4, 45, -0.4, "Down"),
    (5, 38, -1.2, "Down"), (6, 30, -1.9, "Down"), (7, 61, -0.2, "Down"), (8, 49, 0.9, "Up")]
QUERY = (55, 0.3)


def minmax(v, lo, hi):
    return (v - lo) / (hi - lo)


def build(pg: Page, cat):
    y0 = pg.max_y() + 60
    data_illustration(pg, 400, y0)
    tuning(pg, 1440, y0)


# ---------------------------------------------------------------------------------------- ⑥
def data_illustration(pg, x0, y0):
    pg.panel(x0, y0, 1000, 760, 6, "Data Illustration: classify a new trading day with k = 3",
             "Training table → scaling → distance to the query row → vote")
    rl, rh = min(t[1] for t in TRAIN), max(t[1] for t in TRAIN)
    el, eh = min(t[2] for t in TRAIN), max(t[2] for t in TRAIN)
    sc = [(d, minmax(r, rl, rh), minmax(e, el, eh), lab) for d, r, e, lab in TRAIN]
    q = (minmax(QUERY[0], rl, rh), minmax(QUERY[1], el, eh))
    dists = sorted([(math.dist((a, b), q), d, lab, a, b) for d, a, b, lab in sc])
    nn = dists[:3]
    votes = {"Up": sum(1 for x in nn if x[2] == "Up"), "Down": sum(1 for x in nn if x[2] == "Down")}
    pred = max(votes, key=votes.get)

    pg.text("Step 1 · Stored training rows + new query day", x0 + 20, y0 + 62, 380, 20, 12, bold=True)
    t = [["Day", "RSI", "EMA gap %", "Label"]] + [[d, r, f"{e:+.1f}", lab] for d, r, e, lab in TRAIN]
    t.append(['<b>Q</b>', f"<b>{QUERY[0]}</b>", f"<b>{QUERY[1]:+.1f}</b>", "<b>?</b>"])
    pg.table(x0 + 20, y0 + 86, 260, t, size=11, h=250, highlight_rows=(9,))
    arrow_label(pg, x0 + 290, y0 + 200, x0 + 330, y0 + 200, "")
    pg.text("Step 2 · Min-max scale, Euclidean distance to Q, rank", x0 + 340, y0 + 62, 420, 20, 12, bold=True)
    rank = {d: i + 1 for i, (_, d, *_r) in enumerate(dists)}
    o = [["Day", "RSI'", "EMA'", "\\(d(x, Q)\\)", "rank", "Label"]]
    for d, a, b, lab in sc:
        dd = math.dist((a, b), q)
        top = rank[d] <= 3
        o.append([d, f"{a:.2f}", f"{b:.2f}", f"<b>{dd:.3f}</b>" if top else f"{dd:.3f}",
                  f"<b>{rank[d]}</b>" if top else rank[d], lab])
    o.append(["Q", f"{q[0]:.2f}", f"{q[1]:.2f}", "", "", ""])
    pg.table(x0 + 340, y0 + 86, 340, o, size=11, h=250,
             highlight_rows=tuple(i + 1 for i, (d, *_r) in enumerate(sc) if rank[d] <= 3))
    pg.math("x'=\\frac{x-x_{min}}{x_{max}-x_{min}}", x0 + 700, y0 + 95, 280, 44, 14)
    pg.math("d=\\sqrt{(RSI'-RSI'_Q)^2+(EMA'-EMA'_Q)^2}", x0 + 690, y0 + 150, 300, 40, 12)
    pg.html(f"<b>Step 3 · Vote of the 3 nearest</b><br>days {', '.join(str(x[1]) for x in nn)} → "
            f"Up = {votes['Up']}, Down = {votes['Down']}<br><br>"
            f'<span style="background:{HL}"><b>Prediction for Q: {pred}</b></span><br>'
            f"P(Up) = {votes['Up']}/3 = {votes['Up'] / 3:.2f}", x0 + 700, y0 + 205, 280, 120, 12)
    # scaled feature space
    p = pg.axes(x0 + 90, y0 + 400, 330, 250, (0, 1.05), (0, 1.05), "RSI'", "EMA'", label_size=12,
                xticks=(0, 0.5, 1), yticks=(0.5, 1))
    rad = nn[-1][0]
    cx, cy = p.P(*q)
    rx = rad / 1.05 * 330
    ry = rad / 1.05 * 250
    pg.ellipse(cx - rx, cy - ry, 2 * rx, 2 * ry, stroke=GREEN, dashed=True, sw=1.5)
    for dd, d, lab, a, b in nn:
        pg.line([p.P(*q), p.P(a, b)], color=GREEN, width=1.5)
    for d, a, b, lab in sc:
        p.scatter([(a, b)], color=BLUE if lab == "Up" else RED, r=6)
        p.text(str(d), a, b, dx=10, dy=-10, size=10)
    p.scatter([q], color=BLACK, r=6)
    p.text("Q", *q, dx=-12, dy=-12, size=12, bold=True)
    pg.html('<font color="#3333FF">● Up</font> &nbsp; <font color="#FF0000">● Down</font> &nbsp; '
            '<font color="#009900">- - k = 3 radius</font>', x0 + 100, y0 + 378, 300, 18, 11)
    pg.bullets(x0 + 470, y0 + 410, 510, [
        "No training step: the 'model' <b>is</b> the stored table",
        "Without scaling, RSI (range 30–72) would dominate EMA gap (range −1.9…1.8) in the distance",
        "Each prediction costs n distance computations → use KD-tree / Ball-tree for large n",
        "Distance-weighted vote: weight = 1/d (closer neighbours count more)",
        "Regression version: ŷ = mean of the neighbours' targets"], head="Reading it:")
    pg.ask("Should the new day be traded as Up or Down?",
           f"{pred}: days {', '.join(str(x[1]) for x in nn)} are the 3 most similar rows → Up = {votes['Up']}, "
           f"Down = {votes['Down']} → P(Up) = {votes['Up'] / 3:.2f}",
           x0 + 470, y0 + 565, *p.P(*q), w=510)
    pg.bullets(x0 + 20, y0 + 690, 960, [
        "Scale with the <b>training</b> min/max (or mean/σ) and reuse them for every new query to avoid leakage"], size=11)


def knn_predict(train, q, k, metric, weights):
    ds = []
    for (a, b), lab in train:
        d = abs(a - q[0]) + abs(b - q[1]) if metric == "manhattan" else math.hypot(a - q[0], b - q[1])
        ds.append((d, lab))
    ds.sort()
    s = 0.0
    for d, lab in ds[:k]:
        w = 1.0 / (d + 1e-9) if weights == "distance" else 1.0
        s += w if lab == 1 else -w
    return 1 if s > 0 else 0


def tuning(pg, x0, y0):
    pg.panel(x0, y0, 960, 760, 7, "Finding the optimal hyperparameters: grid search + CV",
             "k × distance metric × vote weighting, scored by 5-fold cross-validated accuracy")
    r = seeded(77)
    data = []
    for _ in range(160):
        a, b = r.uniform(-1, 1), r.uniform(-1, 1)
        lab = 1 if a * a + b * b < 0.55 else 0
        if r.random() < 0.12: lab = 1 - lab
        data.append(((a, b), lab))
    ks = [1, 3, 5, 7, 9, 15, 25, 41]
    cfgs = [("euclidean", "uniform"), ("euclidean", "distance"), ("manhattan", "uniform"), ("manhattan", "distance")]
    folds = kfold(len(data), 5)
    grid = []
    for m, w in cfgs:
        row = []
        for k in ks:
            acc = 0
            for f in folds:
                fs = set(f)
                tr = [data[i] for i in range(len(data)) if i not in fs]
                acc += sum(knn_predict(tr, data[i][0], k, m, w) == data[i][1] for i in f)
            row.append(acc / len(data))
        grid.append(row)
    best = max(((i, j) for i in range(4) for j in range(len(ks))), key=lambda ij: grid[ij[0]][ij[1]])
    bi, bj = best
    pg.text("(a) Grid search: mean CV accuracy (160 rows, noisy circular boundary)", x0 + 20, y0 + 62, 600, 20, 12,
            bold=True)
    heatmap(pg, x0 + 150, y0 + 105, 58, 34, [f"{m[:6]}·{w[:4]}" for m, w in cfgs], ks, grid, fmt="{:.3f}",
            best=best, rtitle="metric · weights", ctitle="k (number of neighbours)", size=10)
    pg.html(f'<span style="background:{HL}"><b>Best: k = {ks[bj]}, {cfgs[bi][0]}, {cfgs[bi][1]} → '
            f'{grid[bi][bj]:.3f}</b></span>', x0 + 640, y0 + 110, 300, 40, 12)
    pg.ask("Which k, metric and weighting should I use?",
           f"the yellow cell: k = {ks[bj]}, {cfgs[bi][0]}, {cfgs[bi][1]} (CV accuracy {grid[bi][bj]:.3f}); "
           "k = 1 memorises noise, k = 41 over-smooths",
           x0 + 640, y0 + 160, x0 + 700, y0 + 150, w=300)
    # grid vs random search
    pg.text("(b) Grid search vs random search (same budget = 9 trials)", x0 + 20, y0 + 300, 440, 20, 12, bold=True)
    rr = seeded(5)
    for k, name in enumerate(["Grid", "Random"]):
        bx, by = x0 + 60 + k * 230, y0 + 340
        pg.rect(bx, by, 150, 150, stroke=BLACK)
        pts = [(0.17 + 0.33 * i, 0.17 + 0.33 * j) for i in range(3) for j in range(3)] if k == 0 else \
              [(rr.uniform(0.05, 0.95), rr.uniform(0.05, 0.95)) for _ in range(9)]
        for a, b in pts:
            pg.dot(bx + a * 150, by + (1 - b) * 150, r=4, color=BLUE)
            pg.line([(bx + a * 150, by + 150), (bx + a * 150, by + 160)], color=RED, width=1.5)
        pg.text(f"{name} search", bx, by - 20, 150, 18, 11, bold=True, align="center")
        pg.text("important param →", bx, by + 162, 150, 16, 9, align="center", color=RED)
        pg.text("unimportant ↑", bx - 80, by + 60, 76, 16, 9, align="right", color="#555555")
    pg.text("Random search tries 9 distinct values of the important parameter, grid only 3",
            x0 + 40, y0 + 525, 420, 32, 10, italic=True, align="center")
    # SVM / typical grids
    pg.text("(c) Typical search spaces", x0 + 520, y0 + 300, 420, 20, 12, bold=True)
    pg.table(x0 + 520, y0 + 325, 420, [
        ["Model", "Hyperparameters", "Grid (log scale where ×)"],
        ["kNN", "k, metric, weights", "k ∈ {1,3,5,…,√n}, L1/L2, uniform/distance"],
        ["SVM (RBF)", "C, γ", "C ∈ 10⁻²…10³ ×, γ ∈ 10⁻⁴…10¹ ×"],
        ["SVM (linear)", "C", "C ∈ 10⁻³…10² ×"],
        ["Kernel ridge", "α, γ", "α ∈ 10⁻⁴…10¹ ×"]], size=10, h=130)
    pg.text("SVM: good (C, γ) pairs lie on a diagonal ridge — large C with small γ ≈ small C with large γ",
            x0 + 520, y0 + 462, 420, 32, 10, italic=True, color="#555555")
    pg.bullets(x0 + 20, y0 + 600, 920, [
        "Always scale <b>inside</b> each CV fold (Pipeline: scaler → model) so validation folds stay unseen",
        "GridSearchCV for small spaces, RandomizedSearchCV / Bayesian optimization (Optuna) for large ones; use <b>nested CV</b> to report an unbiased score",
        "Pick the simplest setting within noise of the best (e.g. larger k if accuracy is equal) and refit on all training data",
        "Time series: replace K-fold by walk-forward splits"], head="How to choose:", size=11)
