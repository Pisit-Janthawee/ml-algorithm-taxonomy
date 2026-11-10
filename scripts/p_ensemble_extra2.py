"""Bagging page, part 3: the rest of the ensemble family (categorization, AdaBoost, Gradient Boosting,
Voting, Stacking, comparison) - mirrors the 'Ensemble Learning Categorization' of the original diagram."""
import math
from dio import *

ORANGE_L, GREEN_L, PURPLE_L, GREY_L = "#FFE6CC", "#D5E8D4", "#E1D5E7", "#F5F5F5"


# ---------------------------------------------------------------------- tiny learners (pure Python)
def stump_fit_weighted(xs, ys, ws):
    """1-D decision stump for labels ±1 with sample weights. Returns (thr, sign, err)."""
    best = None
    cands = sorted(set(xs))
    thrs = [cands[0] - 0.5] + [(a + b) / 2 for a, b in zip(cands, cands[1:])] + [cands[-1] + 0.5]
    for t in thrs:
        for s in (1, -1):
            err = sum(w for x, y, w in zip(xs, ys, ws) if (s if x > t else -s) != y)
            if best is None or err < best[2] - 1e-12:
                best = (t, s, err)
    return best


def stump_pred(x, t, s):
    return s if x > t else -s


def reg_stump(xs, rs):
    """Least-squares regression stump on residuals. Returns (thr, left_mean, right_mean)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    best = None
    for k in range(1, len(order)):
        L = [rs[i] for i in order[:k]]
        R = [rs[i] for i in order[k:]]
        ml, mr = sum(L) / len(L), sum(R) / len(R)
        sse = sum((v - ml) ** 2 for v in L) + sum((v - mr) ** 2 for v in R)
        if best is None or sse < best[0]:
            best = (sse, (xs[order[k - 1]] + xs[order[k]]) / 2, ml, mr)
    return best[1:]


def sigmoid(z):
    return 1 / (1 + math.exp(-max(-30, min(30, z))))


def logreg_fit(X, y, iters=400, lr=0.5):
    w = [0.0] * (len(X[0]) + 1)
    n = len(X)
    for _ in range(iters):
        g = [0.0] * len(w)
        for xi, yi in zip(X, y):
            p = sigmoid(w[0] + sum(a * b for a, b in zip(w[1:], xi)))
            e = p - yi
            g[0] += e
            for j, v in enumerate(xi): g[j + 1] += e * v
        w = [a - lr * b / n for a, b in zip(w, g)]
    return w


def logreg_prob(w, x):
    return sigmoid(w[0] + sum(a * b for a, b in zip(w[1:], x)))


def knn_prob(Xtr, ytr, x, k=7):
    d = sorted(((sum((a - b) ** 2 for a, b in zip(xi, x)), yi) for xi, yi in zip(Xtr, ytr)))[:k]
    return sum(yi for _, yi in d) / k


def tree_fit(X, y, depth=2):
    """Tiny Gini classification tree (axis-aligned). Leaf = P(y=1)."""
    p = sum(y) / len(y) if y else 0.5
    if depth == 0 or len(y) < 6 or p in (0, 1):
        return ("leaf", p)
    best = None
    for j in range(len(X[0])):
        vals = sorted(set(round(x[j], 3) for x in X))
        for a, b in zip(vals[::3], vals[3::3]):
            t = (a + b) / 2
            L = [yy for xx, yy in zip(X, y) if xx[j] <= t]
            R = [yy for xx, yy in zip(X, y) if xx[j] > t]
            if len(L) < 3 or len(R) < 3: continue
            g = sum(len(S) * (1 - (sum(S) / len(S)) ** 2 - (1 - sum(S) / len(S)) ** 2) for S in (L, R))
            if best is None or g < best[0]: best = (g, j, t)
    if best is None:
        return ("leaf", p)
    _, j, t = best
    Li = [i for i, xx in enumerate(X) if xx[j] <= t]
    Ri = [i for i, xx in enumerate(X) if xx[j] > t]
    return ("node", j, t, tree_fit([X[i] for i in Li], [y[i] for i in Li], depth - 1),
            tree_fit([X[i] for i in Ri], [y[i] for i in Ri], depth - 1))


def tree_prob(node, x):
    while node[0] == "node":
        node = node[3] if x[node[1]] <= node[2] else node[4]
    return node[1]


def mini(pg, x, y, label, fill="#FFFFFF", stroke="#666666", w=70, h=26, size=9):
    return pg.rect(x, y, w, h, label, fill=fill, stroke=stroke, size=size, raw=True)


def build(pg: Page, cat):
    Y = pg.max_y() + 60
    pg.title("The whole ensemble family: Boosting, Voting & Stacking", 400, Y, 1600, 22)
    pg.text("Bagging (above) is one of four ways to combine models. Same algorithm vs different algorithms, and "
            "parallel vs sequential training, as in the original 'Ensemble Learning Categorization'.",
            400, Y + 34, 2000, 20, 12, italic=True, color="#555555")
    y0 = Y + 70

    # ------------------------------------------------------------------ ⑧ categorization
    x0 = 400
    pg.panel(x0, y0, 1000, 780, 8, "Ensemble Learning Categorization",
             "Rows: same vs different algorithms · Columns: trained in parallel vs in sequence")
    gx, gy, cw, chh = x0 + 120, y0 + 100, 430, 300
    pg.rect(gx, gy - 34, cw, 30, "Parallel (independent models)", fill=GREY_L, bold=True, size=12)
    pg.rect(gx + cw, gy - 34, cw, 30, "Sequential (each model fixes the last)", fill=GREY_L, bold=True, size=12)
    pg.rect(x0 + 20, gy, 100, chh, "<b>Homogeneous</b><br><font style='font-size:10px'>same algorithm</font>",
            fill=GREY_L, raw=True)
    pg.rect(x0 + 20, gy + chh, 100, chh, "<b>Heterogeneous</b><br><font style='font-size:10px'>different "
            "algorithms</font>", fill=GREY_L, raw=True)
    for r in range(2):
        for c in range(2):
            pg.rect(gx + c * cw, gy + r * chh, cw, chh, stroke="#999999")
    # Homogeneous / parallel: Bagging
    cx, cy = gx + 15, gy + 15
    pg.text("Bagging → Random Forest, Extra Trees", cx, cy, 400, 18, 11, bold=True)
    d = mini(pg, cx, cy + 110, "Dataset", fill=LIGHTBLUE, stroke=BLUE)
    subs, mods = [], []
    for k in range(3):
        subs.append(mini(pg, cx + 110, cy + 40 + k * 70, f"Subset D{k + 1}", fill=LIGHTBLUE, stroke=BLUE))
        mods.append(mini(pg, cx + 220, cy + 40 + k * 70, "Model A", fill=ORANGE_L, stroke="#D79B00"))
        pg.connect(d, subs[-1], orth=False)
        pg.connect(subs[-1], mods[-1], orth=False)
    agg = mini(pg, cx + 320, cy + 105, "Average /<br>Vote", fill=HL, w=80, h=36)
    for m in mods: pg.connect(m, agg, orth=False)
    pg.text("bootstrap rows → same model → aggregate · ↓ variance", cx, cy + 250, 400, 18, 10, italic=True,
            color="#555555")
    # Homogeneous / sequential: Boosting
    cx = gx + cw + 15
    pg.text("Boosting → AdaBoost, GBM, XGBoost, LightGBM, CatBoost", cx, cy, 410, 18, 11, bold=True)
    prev = mini(pg, cx, cy + 110, "Dataset", fill=LIGHTBLUE, stroke=BLUE)
    for k in range(3):
        m = mini(pg, cx + 100 + k * 105, cy + 60, f"Model A{k + 1}", fill=ORANGE_L, stroke="#D79B00")
        wgt = mini(pg, cx + 100 + k * 105, cy + 150, "re-weight /<br>residuals", fill="#FFFFFF", h=34, size=8)
        pg.connect(prev, m, orth=False)
        pg.connect(m, wgt, orth=False, color=RED)
        prev = wgt
    pg.ellipse(cx + 170, cy + 210, 160, 34, "weighted sum of A1…An", fill=HL, size=9)
    pg.text("each model focuses on the previous mistakes · ↓ bias", cx, cy + 250, 400, 18, 10, italic=True,
            color="#555555")
    # Heterogeneous / parallel: Voting
    cx, cy = gx + 15, gy + chh + 15
    pg.text("Voting (hard / soft)", cx, cy, 400, 18, 11, bold=True)
    d = mini(pg, cx, cy + 110, "Dataset", fill=LIGHTBLUE, stroke=BLUE)
    mods = []
    for k, (nm, col) in enumerate([("Logistic", ORANGE_L), ("kNN", GREEN_L), ("Tree", PURPLE_L)]):
        mods.append(mini(pg, cx + 150, cy + 40 + k * 70, nm, fill=col))
        pg.connect(d, mods[-1], orth=False)
    agg = mini(pg, cx + 290, cy + 105, "Majority /<br>avg prob", fill=HL, w=90, h=36)
    for m in mods: pg.connect(m, agg, orth=False)
    pg.text("same full data → different models → vote", cx, cy + 250, 400, 18, 10, italic=True, color="#555555")
    # Heterogeneous / parallel + sequential: Stacking
    cx = gx + cw + 15
    pg.text("Stacking / Blending (meta-learning)", cx, cy, 410, 18, 11, bold=True)
    d = mini(pg, cx, cy + 110, "Dataset", fill=LIGHTBLUE, stroke=BLUE)
    mods = []
    for k, (nm, col) in enumerate([("Logistic", ORANGE_L), ("kNN", GREEN_L), ("Tree", PURPLE_L)]):
        mods.append(mini(pg, cx + 100, cy + 40 + k * 70, nm, fill=col))
        pg.connect(d, mods[-1], orth=False)
    mf = mini(pg, cx + 205, cy + 100, "meta-<br>features", fill="#FFFFFF", w=70, h=40)
    for m in mods: pg.connect(m, mf, orth=False)
    ml = mini(pg, cx + 310, cy + 105, "Meta-learner", fill=HL, w=90, h=30)
    pg.connect(mf, ml, orth=False)
    pg.text("Level-0", cx + 100, cy + 22, 70, 16, 9, align="center", color="#555555")
    pg.text("Level-1", cx + 310, cy + 85, 90, 16, 9, align="center", color="#555555")
    pg.text("parallel base models → sequential meta-model learns how to combine them", cx, cy + 250, 410, 18, 10,
            italic=True, color="#555555")
    pg.ask("Which ensemble should I reach for?",
           "high-variance model (deep tree) → Bagging; weak, biased model → Boosting; "
           "several good but different models → Voting / Stacking",
           x0 + 560, y0 + 706, gx + cw, gy + 2 * chh, w=420)
    pg.text("Single model = one iteration of any of these (the original 'Decision tree' column)", x0 + 20,
            y0 + 720, 520, 18, 10, italic=True, color="#555555")

    # ------------------------------------------------------------------ ⑨ AdaBoost worked
    x0 = 1440
    pg.panel(x0, y0, 960, 780, 9, "AdaBoost (worked, 10 points, 3 rounds of stumps)",
             "Misclassified points get heavier; each stump gets a say α by its accuracy")
    xs = list(range(1, 11))
    ys = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]
    ws = [0.1] * 10
    rounds = []
    for t in range(3):
        thr, s, err = stump_fit_weighted(xs, ys, ws)
        alpha = 0.5 * math.log((1 - err) / err)
        preds = [stump_pred(x, thr, s) for x in xs]
        rounds.append((list(ws), thr, s, err, alpha, preds))
        ws = [w * math.exp(-alpha * y * p) for w, y, p in zip(ws, ys, preds)]
        z = sum(ws)
        ws = [w / z for w in ws]
    final = [1 if sum(r[4] * r[5][i] for r in rounds) > 0 else -1 for i in range(10)]
    for t, (w_, thr, s, err, alpha, preds) in enumerate(rounds):
        ry = y0 + 70 + t * 150
        p = Plot(pg, x0 + 60, ry + 20, 420, 80, (0.5, 10.5), (0, 1))
        pg.line([(x0 + 50, ry + 100), (x0 + 490, ry + 100)], arrow="classic")
        for i, x in enumerate(xs):
            px, _ = p.P(x, 0)
            r = 3 + 40 * w_[i]
            col = BLUE if ys[i] == 1 else RED
            pg.ellipse(px - r, ry + 60 - r, 2 * r, 2 * r, fill=col if preds[i] == ys[i] else "#FFFFFF", stroke=col,
                       sw=2)
            pg.text(f"{w_[i]:.2f}", px - 15, ry + 104, 30, 12, 8, align="center", color="#555555")
        tx, _ = p.P(thr, 0)
        pg.line([(tx, ry + 15), (tx, ry + 100)], color=GREEN, width=2, dashed=True)
        side = "+1 right" if s == 1 else "+1 left"
        pg.text(f"Round {t + 1}", x0 + 14, ry + 50, 46, 18, 11, bold=True)
        pg.html(f"stump: x &gt; {thr:g} ? ({side})<br>ε = {err:.3f}<br>α = ½ ln((1−ε)/ε) = <b>{alpha:.3f}</b>",
                x0 + 510, ry + 25, 220, 60, 11)
    pg.text("dot size = sample weight · blue = +1, red = −1 · hollow = misclassified by that round's stump",
            x0 + 40, y0 + 515, 700, 16, 10, italic=True, color="#555555")
    pg.math(r"w_i \leftarrow \frac{w_i\, e^{-\alpha_t y_i h_t(x_i)}}{Z_t}", x0 + 740, y0 + 90, 200, 50, 15)
    pg.math(r"H(x) = \text{sign}\Big(\sum_t \alpha_t h_t(x)\Big)", x0 + 740, y0 + 170, 200, 44, 14)
    pg.text("wrong → weight × e^α (grows), right → × e^−α", x0 + 740, y0 + 140, 210, 16, 9, color=RED)
    rows = [["x"] + [str(x) for x in xs], ["y"] + [("+" if y > 0 else "−") for y in ys]]
    for t, r in enumerate(rounds):
        rows.append([f"h{t + 1}"] + [("+" if p > 0 else "−") for p in r[5]])
    rows.append(["H"] + [("+" if p > 0 else "−") for p in final])
    pg.table(x0 + 40, y0 + 540, 520, rows, size=10, highlight_rows=(5,), h=130)
    tr_err = [sum(1 for i in range(10) if (1 if sum(rr[4] * rr[5][i] for rr in rounds[:k + 1]) > 0 else -1)
                  != ys[i]) / 10 for k in range(3)]
    pg.html("<b>Training error of the ensemble</b><br>" +
            "<br>".join(f"after {k + 1} stump{'s' if k else ''}: {e:.0%}" for k, e in enumerate(tr_err)),
            x0 + 590, y0 + 545, 170, 80, 11)
    pg.ask("Why does adding weak stumps give a strong model?",
           f"no single stump beats {min(r[3] for r in rounds):.0%} weighted error, but their α-weighted vote "
           f"reaches {tr_err[-1]:.0%} training error",
           x0 + 590, y0 + 640, x0 + 560, y0 + 655, w=350)

    # ------------------------------------------------------------------ ⑩ gradient boosting
    y1, x0 = y0 + 820, 400
    pg.panel(x0, y1, 1320, 800, 10, "Gradient Boosting (worked regression, learning rate η = 0.5)",
             "Each new stump is fitted to the residuals (errors) that the ensemble still makes")
    r = seeded(11)
    gx_ = sorted(r.uniform(0, 10) for _ in range(40))
    gy_ = [2 * math.sin(v) + 0.3 * v + r.gauss(0, 0.4) for v in gx_]
    F0 = sum(gy_) / len(gy_)

    def run(eta, iters):
        models, F, hist = [], [F0] * len(gx_), [sum((a - b) ** 2 for a, b in zip(gy_, [F0] * len(gx_))) / len(gx_)]
        snaps = {0: (list(F), None)}
        for m in range(1, iters + 1):
            res = [a - b for a, b in zip(gy_, F)]
            st = reg_stump(gx_, res)
            models.append(st)
            F = [f + eta * (st[1] if v <= st[0] else st[2]) for f, v in zip(F, gx_)]
            hist.append(sum((a - b) ** 2 for a, b in zip(gy_, F)) / len(gx_))
            snaps[m] = (list(F), st)
        return models, hist, snaps

    def predict(models, eta, v):
        return F0 + eta * sum(st[1] if v <= st[0] else st[2] for st in models)

    models, hist, snaps = run(0.5, 50)
    stages = [0, 1, 3, 20]
    sw = 280
    for k, m in enumerate(stages):
        sx = x0 + 40 + k * (sw + 30)
        F, st = snaps[m]
        p = pg.axes(sx, y1 + 100, sw - 20, 150, (0, 10), (-3, 6), "x", "y" if k == 0 else "", label_size=11)
        p.scatter(list(zip(gx_, gy_)), r=2.5)
        grid_ = [i / 20 for i in range(201)]
        p.poly([(v, predict(models[:m], 0.5, v)) for v in grid_], color=RED, width=2)
        lab = "F₀ = mean of y" if m == 0 else f"F_{m} = F_{m - 1} + η·h_{m}" if m < 20 else "F₂₀ (20 stumps)"
        pg.text(lab, sx, y1 + 72, sw - 20, 18, 12, bold=True, align="center", color=RED)
        res = [a - b for a, b in zip(gy_, F)]
        q = pg.axes(sx, y1 + 290, sw - 20, 70, (0, 10), (-3.5, 3.5), "", "r" if k == 0 else "", origin=(0, 0),
                    arrow=False)
        for v, rr in zip(gx_, res):
            q.poly([(v, 0), (v, max(-3.4, min(3.4, rr)))], color=ORANGE, width=2)
        mse = hist[m]
        pg.text(f"residuals · MSE = {mse:.3f}", sx, y1 + 366, sw - 20, 16, 10, align="center", color="#555555")
    pg.math(r"F_m(x) = F_{m-1}(x) + \eta\, h_m(x),\qquad h_m \approx \arg\min_h \sum_i \big(r_{im} - h(x_i)\big)^2", x0 + 40,
            y1 + 400, 760, 44, 16)
    pg.math(r"r_{im} = y_i - F_{m-1}(x_i) = -\frac{\partial L}{\partial F}\ \ \text{(squared loss)}", x0 + 820,
            y1 + 400, 480, 44, 15)
    lc = pg.axes(x0 + 70, y1 + 490, 380, 200, (0, 50), (0, hist[0] * 1.05), "iteration", "train MSE",
                 label_size=11, xticks=(0, 10, 20, 30, 40, 50))
    curves = {}
    for eta, col in [(1.0, RED), (0.3, BLUE), (0.1, GREEN)]:
        _, h_, _ = run(eta, 50)
        curves[eta] = h_
        lc.poly(list(enumerate(h_)), color=col, width=2)
        lc.text(f"η = {eta:g}", 50, h_[-1], dx=34, size=10, color=col, w=60)
    pg.text("Learning rate η (shrinkage)", x0 + 170, y1 + 455, 300, 18, 12, bold=True)
    pg.ask("How fast should each tree correct the errors?",
           f"η = 1 drops training error fastest (MSE {curves[1.0][10]:.2f} at 10 trees) but overfits sooner; "
           f"η = 0.1 needs more trees ({curves[0.1][10]:.2f} at 10) yet usually generalises better",
           x0 + 560, y1 + 722, *lc.P(10, curves[0.1][10]), w=730)
    pg.table(x0 + 560, y1 + 470, 730, [
        ["", "GBM (classic)", "XGBoost", "LightGBM", "CatBoost"],
        ["Tree growth", "level-wise", "level-wise (depth)", "leaf-wise (best leaf first)", "symmetric (oblivious)"],
        ["Regularization", "shrinkage, subsampling", "L1 + L2 on leaf weights, γ split cost", "L1 / L2, min data in leaf",
         "ordered boosting"],
        ["Speed trick", "—", "sparsity-aware, parallel splits", "histogram bins + GOSS + EFB", "GPU, symmetric trees"],
        ["Categoricals", "encode yourself", "encode (native in new versions)", "native (integer codes)",
         "native target statistics"],
        ["Best for", "small data, teaching", "strong default, competitions", "large data, speed", "many categories"]],
        size=10, h=190)
    pg.bullets(x0 + 560, y1 + 668, 730, [
        "Boosting reduces <b>bias</b>: shallow trees (depth 3–6) + many rounds + small η + early stopping on validation",
        "Overfits if rounds grow without validation; noisy labels hurt boosting more than bagging"], size=11)

    # ------------------------------------------------------------------ ⑪ voting
    x0 = 1760
    pg.panel(x0, y1, 640, 800, 11, "Voting: hard vs soft", "Three different models vote on 6 trading days")
    P = [(1, .80, .70, .65), (0, .30, .20, .45), (1, .90, .40, .45),
         (0, .55, .30, .35), (1, .60, .55, .40), (0, .45, .52, .10)]
    rows = [["day", "true", "P_LR", "P_kNN", "P_Tree", "hard votes", "hard", "soft avg", "soft"]]
    acc = {"LR": 0, "kNN": 0, "Tree": 0, "hard": 0, "soft": 0}
    hl = []
    for i, (yt, a, b, c) in enumerate(P):
        votes = sum(1 for v in (a, b, c) if v > 0.5)
        hard = 1 if votes >= 2 else 0
        avg = (a + b + c) / 3
        soft = 1 if avg > 0.5 else 0
        for nm, v in zip(("LR", "kNN", "Tree"), (a, b, c)):
            acc[nm] += (1 if v > 0.5 else 0) == yt
        acc["hard"] += hard == yt
        acc["soft"] += soft == yt
        if hard != soft: hl.append(i + 1)
        mark = lambda pred: ("Up" if pred else "Dn") + (" ✔" if pred == yt else " ✘")
        rows.append([i + 1, "Up" if yt else "Dn", f"{a:.2f}", f"{b:.2f}", f"{c:.2f}", f"{votes}/3 Up",
                     mark(hard), f"{avg:.3f}", mark(soft)])
    rows.append(["acc.", "", f"{acc['LR']}/6", f"{acc['kNN']}/6", f"{acc['Tree']}/6", "", f"{acc['hard']}/6", "",
                 f"<b>{acc['soft']}/6</b>"])
    pg.table(x0 + 20, y1 + 60, 600, rows, size=10, highlight_rows=tuple(hl), h=200)
    pg.math(r"\text{hard: } \hat{y} = \text{mode}\{h_1(x), \dots, h_K(x)\}", x0 + 20, y1 + 280, 600, 34, 14)
    pg.math(r"\text{soft: } \hat{y} = \arg\max_c \frac{1}{K}\sum_k p_k(c \mid x)", x0 + 20, y1 + 320, 600, 40, 14)
    # day 3 bars
    d3 = P[2]
    b = pg.axes(x0 + 70, y1 + 400, 300, 170, (0, 5), (0, 1), "", "P(Up)", label_size=11, yticks=(0.5, 1))
    b.hline(0.5, color=RED)
    vals = [("LR", d3[1], ORANGE_L), ("kNN", d3[2], GREEN_L), ("Tree", d3[3], PURPLE_L),
            ("avg", round(sum(d3[1:]) / 3, 2), HL)]
    for k, (nm, v, col) in enumerate(vals):
        b.bars([(k + 1, v)], 0.6, fill=col, stroke="#666666", labels=True)
        b.text(nm, k + 1, 0, dy=12, size=10)
    pg.text("Day 3: only LR says Up, but it is very sure", x0 + 115, y1 + 368, 300, 18, 11, bold=True)
    pg.ask("Do I trust the majority or the confidence?",
           f"day 3: hard vote says Down (1 of 3), soft average {sum(d3[1:]) / 3:.2f} > 0.5 says Up — correct; "
           "soft wins when models are calibrated",
           x0 + 390, y1 + 400, *b.P(4, sum(d3[1:]) / 3), w=230)
    pg.bullets(x0 + 20, y1 + 610, 600, [
        "<b>Hard</b>: robust, needs only labels; ties need an odd number of models",
        "<b>Soft</b>: uses confidence; needs calibrated probabilities (Platt / isotonic)",
        "Weighted voting: give better models more weight (e.g. by validation score)",
        "Works best with <b>diverse</b> models whose errors are not correlated"], head="When to use:", size=11)

    # ------------------------------------------------------------------ ⑫ stacking
    y2, x0 = y1 + 840, 400
    pg.panel(x0, y2, 1100, 700, 12, "Stacking (meta-learning) with out-of-fold meta-features",
             "Level-0 models predict; a Level-1 model learns how much to trust each one")
    r = seeded(5)
    X, Y_ = [], []
    for _ in range(300):
        a, b = r.uniform(-1, 1), r.uniform(-1, 1)
        yy = 1 if a + 0.6 * math.sin(3 * b) + 0.4 * a * b > 0 else 0
        if r.random() < 0.08: yy = 1 - yy
        X.append((a, b)); Y_.append(yy)
    Xtr, ytr, Xte, yte = X[:200], Y_[:200], X[200:], Y_[200:]
    K = 5
    fold = [i % K for i in range(len(Xtr))]
    oof = [[0.0] * 3 for _ in Xtr]
    for k in range(K):
        tri = [i for i in range(len(Xtr)) if fold[i] != k]
        tei = [i for i in range(len(Xtr)) if fold[i] == k]
        A, B = [Xtr[i] for i in tri], [ytr[i] for i in tri]
        w = logreg_fit(A, B)
        tr = tree_fit(A, B, 3)
        for i in tei:
            oof[i] = [logreg_prob(w, Xtr[i]), knn_prob(A, B, Xtr[i]), tree_prob(tr, Xtr[i])]
    w_full, tr_full = logreg_fit(Xtr, ytr), tree_fit(Xtr, ytr, 3)
    meta_te = [[logreg_prob(w_full, x), knn_prob(Xtr, ytr, x), tree_prob(tr_full, x)] for x in Xte]
    wm = logreg_fit(oof, ytr, iters=800, lr=1.0)
    accs = {}
    for j, nm in enumerate(["Logistic", "kNN", "Tree"]):
        accs[nm] = sum((m[j] > 0.5) == bool(t) for m, t in zip(meta_te, yte)) / len(yte)
    accs["Soft vote"] = sum((sum(m) / 3 > 0.5) == bool(t) for m, t in zip(meta_te, yte)) / len(yte)
    accs["Stacked"] = sum((logreg_prob(wm, m) > 0.5) == bool(t) for m, t in zip(meta_te, yte)) / len(yte)
    # K-fold OOF diagram
    fx, fy = x0 + 30, y2 + 90
    pg.text("1. 5-fold out-of-fold (OOF) predictions on the training set", fx, fy - 28, 460, 18, 12, bold=True)
    for k in range(K):
        for c in range(K):
            pg.rect(fx + c * 56, fy + k * 30, 56, 24, "predict" if c == k else "fit",
                    fill=ORANGE_L if c == k else LIGHTBLUE, stroke="#666666", size=9)
        pg.text(f"fold {k + 1}", fx + 290, fy + k * 30 + 3, 50, 18, 10)
    pg.line([(fx + 345, fy + 70), (fx + 395, fy + 70)], arrow="classic")
    rows = [["row", "P_LR", "P_kNN", "P_Tree", "y"]] + \
           [[i + 1, f"{oof[i][0]:.2f}", f"{oof[i][1]:.2f}", f"{oof[i][2]:.2f}", ytr[i]] for i in range(6)] + \
           [["…", "…", "…", "…", "…"]]
    pg.text("2. meta-features (each row predicted by models that never saw it)", fx + 400, fy - 28, 460, 18, 12,
            bold=True)
    pg.table(fx + 400, fy, 300, rows, size=10, h=175)
    mlb = pg.rect(fx + 760, fy + 50, 240, 70,
                  f"<b>Level-1 meta-learner</b><br>logistic on [P_LR, P_kNN, P_Tree]<br>"
                  f"w = [{wm[1]:.2f}, {wm[2]:.2f}, {wm[3]:.2f}], b = {wm[0]:.2f}", fill=HL, raw=True, size=10)
    pg.line([(fx + 705, fy + 85), (fx + 755, fy + 85)], arrow="classic")
    pg.text("3. learn how to combine", fx + 760, fy + 22, 240, 18, 12, bold=True)
    # accuracy bars
    ac = pg.axes(x0 + 80, y2 + 340, 460, 200, (0, 6), (0.5, 1.0), "", "test acc.", label_size=11,
                 yticks=(0.6, 0.8, 1.0), origin=(0, 0.5))
    cols = [ORANGE_L, GREEN_L, PURPLE_L, LIGHTBLUE, HL]
    for k, (nm, v) in enumerate(accs.items()):
        ac.bars([(k + 1, round(v, 2))], 0.6, fill=cols[k], stroke="#666666", labels=True)
        ac.text(nm, k + 1, 0.5, dy=12, size=10, w=80)
    pg.text(f"Test accuracy on 100 unseen rows (200 train rows, 8 % label noise)", x0 + 60, y2 + 300, 520, 18, 12,
            bold=True)
    best_base = max(accs[n] for n in ("Logistic", "kNN", "Tree"))
    pg.ask("Is combining the models better than the best single one?",
           f"stacked {accs['Stacked']:.0%} vs best base model {best_base:.0%} and soft vote {accs['Soft vote']:.0%}; "
           "gains are usually small — use it when base models are diverse",
           x0 + 600, y2 + 360, *ac.P(5, accs["Stacked"]), w=460)
    pg.bullets(x0 + 600, y2 + 470, 470, [
        "<b>Never</b> train the meta-learner on in-sample predictions → it trusts overfitted models (leakage)",
        "<b>Blending</b>: meta-learner trained on one hold-out split instead of K-fold OOF (simpler, wastes data)",
        "Keep Level-1 simple (logistic / ridge); time series → purged / walk-forward folds",
        "The meta-weights show which base model is trusted most"], head="Rules:", size=11)
    pg.math(r"\hat{y} = g\big(h_1(x), h_2(x), \dots, h_K(x)\big)", x0 + 60, y2 + 590, 480, 40, 16)
    pg.text("g = Level-1 meta-learner · hₖ = Level-0 base models (trained on full data for prediction time)",
            x0 + 40, y2 + 640, 520, 32, 10, italic=True, color="#555555")

    # ------------------------------------------------------------------ ⑬ comparison
    x0 = 1540
    pg.panel(x0, y2, 860, 700, 13, "Bagging vs Boosting vs Voting vs Stacking")
    pg.table(x0 + 20, y2 + 50, 820, [
        ["", "<b>★ Bagging / RF</b>", "<b>Boosting</b>", "<b>Voting</b>", "<b>Stacking</b>"],
        ["Mainly reduces", "variance", "bias (and variance)", "variance", "bias + variance"],
        ["Training", "parallel", "sequential", "parallel", "parallel → sequential"],
        ["Base learners", "same, deep / unstable", "same, shallow / weak", "different", "different"],
        ["Combiner", "average / majority", "weighted sum (α, η)", "majority / avg prob", "learned meta-model"],
        ["Overfitting risk", "low", "medium–high (needs early stop)", "low", "medium (needs OOF)"],
        ["Key hyperparameters", "n_estimators, max_features", "n_rounds, η, depth", "weights, hard/soft",
         "base set, meta-model, K"],
        ["Typical models", "Random Forest, Extra Trees", "AdaBoost, XGBoost, LightGBM, CatBoost", "LR + kNN + RF",
         "LR + GBM + NN → ridge"],
        ["Trading use", "robust baseline signal", "strongest tabular model", "combine strategies",
         "combine model families"]], size=10, h=330)
    pg.ask("Which ensemble should be my first try on tabular data?",
           "start with Random Forest (few knobs, hard to overfit), then gradient boosting with early stopping; "
           "add voting / stacking only for the last few %",
           x0 + 20, y2 + 270, x0 + 210, y2 + 228, w=400)
    pg.bullets(x0 + 440, y2 + 262, 400, [
        "Diversity is the engine of every ensemble: correlated models add little",
        "Variance of an average of B models with correlation ρ: ρσ² + (1−ρ)σ²/B",
        "Ensembles cost interpretability → use permutation importance / SHAP (Explainability page)"],
        head="Remember:", size=11)
    return y2 + 800
