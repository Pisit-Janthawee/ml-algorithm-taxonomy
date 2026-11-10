"""Extra panels for the Bagging page: data illustration + hyperparameter search (OOB grid, learning curve)."""
import math
from dio import *
from l_common import *

ROWS = [  # day, RSI, EMA gap %, volatility %, label (1 = Up)
    (1, 72, 1.8, 0.9, 1), (2, 65, 1.1, 1.4, 1), (3, 58, 0.6, 2.1, 0), (4, 45, -0.4, 1.0, 0),
    (5, 38, -1.2, 2.6, 0), (6, 30, -1.9, 1.7, 1), (7, 61, -0.2, 0.8, 0), (8, 49, 0.9, 1.2, 1)]
FEATS = ["RSI", "EMA gap", "Vol"]


def stump(sample):
    """Best single split (feature, threshold) by misclassification on the bootstrap sample."""
    best = None
    for f in range(3):
        vals = sorted(set(r[1 + f] for r in sample))
        for a, b in zip(vals, vals[1:]):
            t = (a + b) / 2
            for left_lab in (0, 1):
                err = sum((left_lab if r[1 + f] <= t else 1 - left_lab) != r[4] for r in sample)
                if best is None or err < best[0]:
                    best = (err, f, t, left_lab)
    if best is None:
        lab = round(sum(r[4] for r in sample) / len(sample))
        return (0, 0, 1e9, lab)
    return best


def stump_pred(s, row):
    _, f, t, ll = s
    return ll if row[1 + f] <= t else 1 - ll


def build(pg: Page, cat):
    y0 = pg.max_y() + 60
    data_illustration(pg, 400, y0)
    tuning(pg, 1440, y0)


# ---------------------------------------------------------------------------------------- data illustration
def data_illustration(pg, x0, y0):
    pg.panel(x0, y0, 1000, 760, 6, "Data Illustration: 5 bootstrap trees vote on every row",
             "8 trading days, base learner = decision stump (1 split) so each tree's rule is readable")
    r = seeded(4)
    B = 5
    samples = [[r.randrange(8) for _ in range(8)] for _ in range(B)]
    stumps = [stump([ROWS[i] for i in s]) for s in samples]

    pg.text("Step 1 · Training table", x0 + 20, y0 + 62, 300, 20, 12, bold=True)
    t = [["Day", "RSI", "EMA gap", "Vol %", "Up?"]] + [[d, a, f"{b:+.1f}", c, f"<b>{u}</b>"] for d, a, b, c, u in ROWS]
    pg.table(x0 + 20, y0 + 86, 280, t, size=11, h=230)
    pg.text("Step 2 · Bootstrap row indices & the stump learned from each sample", x0 + 330, y0 + 62, 650, 20, 12,
            bold=True)
    pal = ["#F8CECC", "#DAE8FC", "#D5E8D4", "#FFF2CC", "#E1D5E7", "#FFE6CC", "#F5F5F5", "#CCE5FF"]
    for b, (s, st) in enumerate(zip(samples, stumps)):
        yy = y0 + 92 + b * 46
        pg.text(f"Tree {b + 1}", x0 + 330, yy, 56, 28, 11, bold=True)
        for k, i in enumerate(sorted(s)):
            pg.rect(x0 + 390 + k * 30, yy + 2, 26, 24, str(i + 1), fill=pal[i], stroke="#666666", size=10)
        oob = sorted(set(range(8)) - set(s))
        _, f, th, ll = st
        rule = f"{FEATS[f]} ≤ {th:g} → {'Up' if ll else 'Down'}, else {'Down' if ll else 'Up'}"
        pg.html(f"<b>{rule}</b><br><font color='#888888' style='font-size:9px'>OOB days: "
                f"{', '.join(str(i + 1) for i in oob) or '—'}</font>", x0 + 640, yy - 2, 340, 34, 10, valign="middle")
    pg.text("duplicates = drawn twice; missing days are out-of-bag (OOB) for that tree", x0 + 390, y0 + 322, 560, 16,
            10, italic=True, color="#555555")

    # votes table
    pg.line([(x0 + 500, y0 + 345), (x0 + 500, y0 + 370)], arrow="block", width=2)
    pg.text("Step 3 · Every tree predicts every row → majority vote (and OOB vote using only trees that never saw the row)",
            x0 + 20, y0 + 378, 960, 20, 12, bold=True)
    head = ["Day", "actual"] + [f"T{b + 1}" for b in range(B)] + ["votes Up", "bagged ŷ", "OOB ŷ", ""]
    o = [head]
    bag_ok = oob_ok = oob_n = 0
    for i, row in enumerate(ROWS):
        preds = [stump_pred(st, row) for st in stumps]
        cells = []
        oob_votes = []
        for b, pr in enumerate(preds):
            is_oob = i not in samples[b]
            if is_oob: oob_votes.append(pr)
            txt = f"<u><b>{pr}</b></u>" if is_oob else str(pr)
            cells.append(txt)
        up = sum(preds)
        yh = 1 if up > B / 2 else 0
        bag_ok += yh == row[4]
        if oob_votes:
            oy = 1 if sum(oob_votes) * 2 > len(oob_votes) else (0 if sum(oob_votes) * 2 < len(oob_votes) else yh)
            oob_n += 1; oob_ok += oy == row[4]
            ostr = str(oy)
        else:
            ostr = "—"
        mark = '<font color="#009900">✔</font>' if yh == row[4] else '<font color="#FF0000">✘</font>'
        o.append([row[0], row[4]] + cells + [f"{up}/{B}", f"<b>{yh}</b>", ostr, mark])
    pg.table(x0 + 20, y0 + 404, 640, o, size=11, h=240)
    pg.html(f"<b>Training accuracy</b> (bagged): {bag_ok}/8<br>"
            f'<span style="background:{HL}"><b>OOB accuracy: {oob_ok}/{oob_n}</b></span><br>'
            f"<font color='#555555'>underlined = OOB prediction (tree did not see that day)</font><br><br>"
            f"OOB accuracy is an honest estimate — like a free validation set — because each row is judged only by "
            f"trees trained without it.<br><br><font color='#FF0000'><b>Gap {bag_ok}/8 vs {oob_ok}/{oob_n}</b>: "
            f"8 rows are far too few — the training score is optimistic, OOB exposes it.</font>",
            x0 + 680, y0 + 410, 300, 230, 11)
    pg.ask(f"Can I trust the {bag_ok}/8 training accuracy?",
           f"no: those rows were seen in training; the OOB vote ({oob_ok}/{oob_n}) judges each row only with trees "
           "that never saw it",
           x0 + 680, y0 + 560, x0 + 760, y0 + 440, w=300)
    pg.bullets(x0 + 20, y0 + 670, 960, [
        "Single stumps disagree (different bootstrap samples → different rules); the <b>vote</b> is more stable than any one tree",
        "Regression bagging averages the trees' numeric predictions instead of voting",
        "Real bagging uses deep trees (high variance) and 100+ estimators; stumps are used here only to keep each rule readable"],
        size=11)


def grow(X, y, idx, depth, maxf, rng):
    labs = [y[i] for i in idx]
    maj = 1 if sum(labs) * 2 >= len(labs) else 0
    if depth == 0 or len(idx) < 4 or sum(labs) in (0, len(labs)):
        return maj
    best = None
    p = len(X[0])
    for f in rng.sample(range(p), maxf):
        vals = sorted(X[i][f] for i in idx)
        for q in range(1, 8):
            t = vals[len(vals) * q // 8]
            L = [i for i in idx if X[i][f] <= t]
            R = [i for i in idx if X[i][f] > t]
            if not L or not R: continue
            g = 0
            for part in (L, R):
                pp = sum(y[i] for i in part) / len(part)
                g += len(part) * 2 * pp * (1 - pp)
            if best is None or g < best[0]:
                best = (g, f, t, L, R)
    if best is None:
        return maj
    _, f, t, L, R = best
    return (f, t, grow(X, y, L, depth - 1, maxf, rng), grow(X, y, R, depth - 1, maxf, rng))


def tpred(node, x):
    while isinstance(node, tuple):
        node = node[2] if x[node[0]] <= node[1] else node[3]
    return node


def bag_fit(X, y, idx, n_trees, frac, maxf, rng, depth=4):
    trees, samples = [], []
    m = max(4, int(frac * len(idx)))
    for _ in range(n_trees):
        s = [idx[rng.randrange(len(idx))] for _ in range(m)]
        trees.append(grow(X, y, s, depth, maxf, rng)); samples.append(set(s))
    return trees, samples


def bag_pred(trees, x):
    v = sum(tpred(t, x) for t in trees)
    return 1 if v * 2 > len(trees) else 0


def tuning(pg, x0, y0):
    pg.panel(x0, y0, 960, 760, 7, "Finding the optimal hyperparameters: OOB grid + learning curve",
             "max_samples × max_features scored by OOB error (no extra validation split needed)")
    r = seeded(12)
    X, y = [], []
    for _ in range(300):
        f = [r.gauss(0, 1) for _ in range(6)]
        z = math.sin(1.5 * f[0]) + 0.8 * f[1] * f[2] + 0.3 * r.gauss(0, 1)
        X.append(f); y.append(1 if z > 0 else 0)
    pool = list(range(200))
    fracs, maxfs = [0.3, 0.5, 0.7, 1.0], [1, 2, 4, 6]
    grid = []
    for fr in fracs:
        row = []
        for mf in maxfs:
            rng = seeded(int(fr * 100) + mf)
            trees, samples = bag_fit(X, y, pool, 15, fr, mf, rng)
            err = n = 0
            for i in pool:
                vs = [tpred(t, X[i]) for t, s in zip(trees, samples) if i not in s]
                if vs:
                    n += 1
                    err += (1 if sum(vs) * 2 > len(vs) else 0) != y[i]
            row.append(err / max(n, 1))
        grid.append(row)
    best = min(((i, j) for i in range(4) for j in range(4)), key=lambda ij: grid[ij[0]][ij[1]])
    pg.text("(a) OOB error grid (15 trees, depth 4, 200 rows, 6 features)", x0 + 20, y0 + 62, 460, 20, 12, bold=True)
    heatmap(pg, x0 + 120, y0 + 110, 70, 40, [f"{f:.0%}" for f in fracs], maxfs, grid, fmt="{:.3f}", best=best,
            high_good=False, rtitle="max_samples", ctitle="max_features (features tried per split)", size=11)
    bi, bj = best
    pg.html(f'<span style="background:{HL}"><b>Best: max_samples = {fracs[bi]:.0%}, max_features = {maxfs[bj]} → '
            f'OOB error {grid[bi][bj]:.3f}</b></span><br><font color="#555555">max_features &lt; p decorrelates the '
            f'trees = Random Forest; max_features = p = plain Bagging</font>', x0 + 20, y0 + 330, 440, 60, 11)
    pg.ask("Which bootstrap size and feature count work best?",
           f"{fracs[bi]:.0%} of rows and {maxfs[bj]} features per split (OOB error {grid[bi][bj]:.3f}, yellow cell) — "
           "no separate validation set needed",
           x0 + 20, y0 + 398, x0 + 120 + bj * 70 + 35, y0 + 110 + (bi + 1) * 40, w=460)
    # learning curve
    sizes = [20, 40, 80, 120, 160, 200]
    val = list(range(200, 300))
    tr_err, va_err = [], []
    for n_ in sizes:
        rng = seeded(n_)
        idx = pool[:n_]
        trees, _ = bag_fit(X, y, idx, 15, 0.7, maxfs[bj], rng)
        tr_err.append(sum(bag_pred(trees, X[i]) != y[i] for i in idx) / n_)
        va_err.append(sum(bag_pred(trees, X[i]) != y[i] for i in val) / len(val))
    pg.text("(b) Learning curve: error vs training-set size", x0 + 500, y0 + 62, 440, 20, 12, bold=True)
    top = max(va_err + tr_err) * 1.2 + 0.02
    p = pg.axes(x0 + 550, y0 + 125, 360, 190, (0, 210), (0, top), "n_{train}", "error", label_size=11,
                xticks=sizes, yticks=(round(top / 2, 2),))
    p.poly(list(zip(sizes, tr_err)), color=BLUE, width=2)
    p.poly(list(zip(sizes, va_err)), color=RED, width=2)
    p.scatter(list(zip(sizes, tr_err)), color=BLUE, r=3)
    p.scatter(list(zip(sizes, va_err)), color=RED, r=3)
    p.text(f"{va_err[-1]:.2f}", 200, va_err[-1], dx=22, size=10, color=RED, w=40)
    p.text(f"{tr_err[-1]:.2f}", 200, tr_err[-1], dx=22, size=10, color=BLUE, w=40)
    pg.html('<font color="#3333FF"><b>— training error</b></font> &nbsp; <font color="#FF0000"><b>— validation error'
            '</b></font>', x0 + 640, y0 + 92, 300, 18, 10)
    falling = va_err[-1] < min(va_err[:-1]) - 0.01
    pg.bullets(x0 + 500, y0 + 340, 440, [
        f"Gap between the curves = variance (here {va_err[-1] - tr_err[-1]:.2f} at n = 200)",
        "Validation curve still falling → <b>more data will help</b>; flat → more data alone won't",
        f"Here: validation error {va_err[0]:.2f} → {va_err[-1]:.2f} "
        + ("(still improving)" if falling else "(already flat → improve features / tree depth instead)"),
        "Both curves high and close → high bias → deeper trees / better features"], head="Reading the learning curve:",
        size=11)
    pg.table(x0 + 20, y0 + 495, 920, [
        ["Hyperparameter", "Effect", "How to choose"],
        ["n_estimators", "more trees → lower variance, never overfits, slower", "increase until OOB error plateaus (panel ⑤)"],
        ["max_samples", "smaller samples → more diverse but weaker trees", "OOB grid (a): 50–100 % typical"],
        ["max_features", "fewer features per split → decorrelated trees (Random Forest)", "√p (classification), p/3 (regression), or grid"],
        ["base-tree depth", "deep trees = low bias, high variance (what bagging fixes)", "fully grown or tune max_depth / min_samples_leaf"]],
        size=11, h=150)
    pg.bullets(x0 + 20, y0 + 655, 920, [
        "OOB error ≈ cross-validation error for free → tune bagging / Random Forest without a separate validation split",
        "Time series: bootstrap breaks temporal order → use block bootstrap / sequential bootstrap (López de Prado) and purged CV"],
        head="How to choose:", size=11)
